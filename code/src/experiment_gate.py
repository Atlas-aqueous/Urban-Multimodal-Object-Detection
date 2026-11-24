"""读取实验 JSON 并执行 60+ 计划的晋级门禁。"""
import argparse
import json
from pathlib import Path


def official_metrics(path: Path) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return payload.get("official_validation") or payload.get("best") or payload


def screen_gate(baseline: dict, candidate: dict) -> dict:
    total_gain = candidate["map50_95"] - baseline["map50_95"]
    base_pc = baseline["per_class_ap50_95"]
    cand_pc = candidate["per_class_ap50_95"]
    rare_gain = ((cand_pc["ball"] + cand_pc["tricycle"])
                 - (base_pc["ball"] + base_pc["tricycle"])) / 2
    passed = total_gain >= 0.005 or (rare_gain >= 0.03 and total_gain >= -0.002)
    return {"passed": passed, "total_gain": total_gain, "rare_mean_gain": rare_gain}


def cross_split_gate(gains: list[float]) -> dict:
    mean = sum(gains) / len(gains)
    passed = mean >= 0.015 and sum(g > 0 for g in gains) >= 2 and min(gains) >= -0.005
    return {"passed": passed, "mean_gain": mean,
            "positive_count": sum(g > 0 for g in gains), "worst_gain": min(gains)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline", required=True, type=Path)
    ap.add_argument("--candidate", required=True, type=Path)
    args = ap.parse_args()
    print(json.dumps(screen_gate(official_metrics(args.baseline), official_metrics(args.candidate)),
                     ensure_ascii=False, indent=2))
