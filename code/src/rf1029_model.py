"""RF1029-M with validity-weighted cross-modal attention at P4."""
import torch
from torch import nn
from torch.nn import functional as F

from rfdetr.config import RFDETRMediumConfig
from rfdetr.main import populate_args
from rfdetr.models import build_model, build_criterion_and_postprocessors

from src.rf1029_data import SENSOR_STATS
from src.trimodal_rfdetr import NestedTensor, SensorBranch, extend_patch_embed


class ValidSensor(SensorBranch):
    def __init__(self, channels, pool):
        super().__init__(channels)
        self.pool = pool
        width = self.project.out_channels
        self.attn_in = nn.Conv2d(width, width, 1, bias=False)
        self.attn_q = nn.Conv2d(width, width, 1, bias=False)
        self.attn_kv = nn.Conv2d(width, 2 * width, 1, bias=False)
        self.attn_proj = nn.Conv2d(width, width, 1, bias=False)
        self.attn_norm = nn.GroupNorm(8, width)
        self.attn_gamma = nn.Parameter(torch.zeros(1))
        self.validity_gamma = nn.Parameter(torch.zeros(1))

    def global_attention(self, rgb, sensor):
        batch, channels, height, width = rgb.shape
        if self.pool > 1 and min(height, width) > self.pool:
            sensor = F.avg_pool2d(sensor, self.pool)
        queries = self.attn_q(rgb).flatten(2).transpose(1, 2)
        keys, values = self.attn_kv(sensor).flatten(2).chunk(2, dim=1)
        attention = torch.softmax(torch.bmm(queries, keys) / (channels ** .5), dim=-1)
        output = torch.bmm(attention, values.transpose(1, 2))
        output = output.transpose(1, 2).reshape(batch, channels, height, width)
        return self.attn_gamma * self.attn_proj(self.attn_norm(output))

    def masked_attention(self, rgb, sensor, fraction, content):
        batch, channels, height, width = rgb.shape
        if bool((fraction == 1).all()) and bool((content == 1).all()):
            return self.global_attention(rgb, sensor)
        mass = fraction.float()
        weighted = sensor * mass.to(sensor.dtype)
        if self.pool > 1 and min(height, width) > self.pool:
            mass = F.avg_pool2d(mass, self.pool)
            weighted = F.avg_pool2d(weighted, self.pool)
        pooled = weighted / mass.clamp_min(1e-6).to(weighted.dtype)
        queries = self.attn_q(rgb).flatten(2).transpose(1, 2)
        keys, values = self.attn_kv(pooled).flatten(2).chunk(2, dim=1)
        allowed = mass.flatten(2) > 1e-6
        with torch.autocast(device_type=rgb.device.type, enabled=False):
            scores = torch.bmm(queries.float(), keys.float()) / (channels ** .5)
            scores = scores + mass.flatten(2).clamp_min(1e-6).log()
            scores = scores.masked_fill(~allowed, -1e4)
            attention = scores.softmax(-1) * allowed
            attention = attention / attention.sum(-1, keepdim=True).clamp_min(1e-8)
        output = torch.bmm(attention.to(values.dtype), values.transpose(1, 2))
        output = output.transpose(1, 2).reshape(batch, channels, height, width)
        residual = self.attn_gamma * self.attn_proj(self.attn_norm(output))
        present = allowed.any(-1).reshape(batch, 1, 1, 1)
        return residual * content.to(residual.dtype) * present

    def forward(self, sensor, rgb, fraction, content):
        feature = F.interpolate(self.body(sensor), rgb.shape[-2:], mode="bilinear", align_corners=False)
        projected = self.attn_in(feature)
        original = self.global_attention(rgb, projected)
        masked = self.masked_attention(rgb, projected, fraction, content)
        return original + self.validity_gamma.tanh() * (masked - original)


class ValidFusion(nn.Module):
    def __init__(self, backbone, pool):
        super().__init__()
        self.encoder = backbone.encoder
        self.projector = backbone.projector
        extend_patch_embed(self.encoder, 6)
        self.ir_branch = ValidSensor(1, pool)
        self.depth_branch = ValidSensor(2, pool)

    def forward(self, samples):
        inputs = samples.tensors
        rgb, = self.projector(self.encoder(inputs))
        if hasattr(rgb, "tensors"):
            rgb = rgb.tensors
        shape = rgb.shape[-2:]
        content_mask = samples.content_valid[:, None].float()
        mean, std = SENSOR_STATS[2]
        depth_bits = (inputs[:, 5:6].float() * std + mean).clamp(0., 1.)
        content = F.adaptive_avg_pool2d(content_mask, shape)
        depth = F.adaptive_avg_pool2d((depth_bits > .5).float() * content_mask, shape)
        fused = (rgb + self.ir_branch(inputs[:, 3:4], rgb, content, content)
                 + self.depth_branch(inputs[:, 4:6], rgb, depth, content))
        mask = F.interpolate(samples.mask[:, None].float(), size=shape, mode="nearest")[:, 0].bool()
        return [NestedTensor(fused, mask)]


def build_detector(config, state, initialize=False):
    """Load the semifinal checkpoint or initialize it from the preceding v3 checkpoint."""
    torch.manual_seed(config["seed"])
    model_config = RFDETRMediumConfig().model_dump()
    model_config.update(num_classes=11, resolution=config["imgsz"], device="cpu",
                        gradient_checkpointing=True, num_select=100)
    args = populate_args(**model_config)
    model = build_model(args)
    model.backbone[0] = ValidFusion(model.backbone[0], config["attn_pool"])
    if initialize:
        # The two learned corrections start at zero when fine-tuning from v3.
        current = model.state_dict()
        state = {**state, **{key: value for key, value in current.items()
                             if key.endswith(".validity_gamma")}}
    model.load_state_dict(state, strict=True)
    criterion, _ = build_criterion_and_postprocessors(args)
    criterion.losses = ["labels", "boxes"]
    return model, criterion
