"""清理前的完整性核验（只读）。

1. 评测阶段测试集已无压缩包，是唯一副本 —— 必须逐张核验可读性。
2. 基线阶段训练集 / 测试集的 ZIP 与已解压目录逐条比对大小，
   确认解压完整后才可删除 ZIP。
"""
import hashlib
import json
import os
import sys
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor

ROOT = "data"
P1 = os.path.join(ROOT, "基线阶段数据集-面向城市场景的多模态目标检测")
P2 = os.path.join(ROOT, "评测阶段测试集-面向城市场景的多模态目标检测")

REPORT = {}


def crc32_of(path, chunk=1 << 22):
    c = 0
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            c = zlib.crc32(b, c)
    return c & 0xFFFFFFFF


def verify_zip(name, zip_path, extract_root, sample_crc=40):
    """逐条比对 ZIP 内条目与磁盘文件；抽样做 CRC32。"""
    out = {"zip": zip_path, "exists": os.path.exists(zip_path)}
    if not out["exists"]:
        out["status"] = "ZIP_MISSING"
        return out
    with zipfile.ZipFile(zip_path) as z:
        infos = [i for i in z.infolist() if not i.is_dir()]
        out["entries"] = len(infos)
        out["zip_bytes"] = os.path.getsize(zip_path)
        missing, size_mismatch, ok = [], [], 0
        for i in infos:
            # ZIP 内路径形如 "visible/00000004.jpg" 或带一层前缀
            rel = i.filename
            parts = rel.split("/")
            for cut in range(len(parts)):
                cand = os.path.join(extract_root, *parts[cut:])
                if os.path.exists(cand):
                    rel = os.path.join(*parts[cut:])
                    break
            else:
                missing.append(i.filename)
                continue
            disk = os.path.join(extract_root, rel)
            if os.path.getsize(disk) != i.file_size:
                size_mismatch.append((rel, i.file_size, os.path.getsize(disk)))
            else:
                ok += 1
        out["matched_size"] = ok
        out["missing"] = missing[:20]
        out["n_missing"] = len(missing)
        out["size_mismatch"] = size_mismatch[:20]
        out["n_size_mismatch"] = len(size_mismatch)

        # 抽样 CRC 复核（含最大与最小的若干条目）
        srt = sorted(infos, key=lambda i: i.file_size)
        pick = srt[: sample_crc / 2] + srt[-(sample_crc / 2):]
        bad = []
        with ThreadPoolExecutor(8) as ex:
            def job(i):
                parts = i.filename.split("/")
                for cut in range(len(parts)):
                    cand = os.path.join(extract_root, *parts[cut:])
                    if os.path.exists(cand):
                        return i.filename, crc32_of(cand), i.CRC
                return i.filename, None, i.CRC
            for fn, got, want in ex.map(job, pick):
                if got is None:
                    bad.append((fn, "absent"))
                elif got != want:
                    bad.append((fn, hex(got), hex(want)))
        out["crc_checked"] = len(pick)
        out["crc_bad"] = bad
    out["status"] = (
        "OK" if not out["n_missing"] and not out["n_size_mismatch"] and not out["crc_bad"]
        else "PROBLEM"
    )
    return out


def verify_phase2():
    """评测阶段集：无 ZIP，唯一副本，逐张打开校验。"""
    out = {"root": P2, "dirs": {}}
    total = 0
    bad = []
    for sub in sorted(os.listdir(P2)):
        d = os.path.join(P2, sub)
        if not os.path.isdir(d):
            continue
        files = sorted(f for f in os.listdir(d) if not f.startswith("."))
        nbytes = 0
        for f in files:
            p = os.path.join(d, f)
            nbytes += os.path.getsize(p)
            try:
                with open(p, "rb") as fh:
                    head = fh.read(4)
                fh2 = open(p, "rb")
                fh2.seek(max(0, os.path.getsize(p) - 4))
                fh2.read()
                fh2.close()
                if len(head) < 4:
                    bad.append((sub, f, "too_short"))
            except Exception as e:  # noqa: BLE001
                bad.append((sub, f, repr(e)))
        out["dirs"][sub] = {"files": len(files), "bytes": nbytes}
        total += len(files)
    out["total_files"] = total
    out["unreadable"] = bad[:20]
    out["n_unreadable"] = len(bad)
    # 三模态 stem 一致性
    stems = {sub: {os.path.splitext(f)[0] for f in os.listdir(os.path.join(P2, sub))
                   if not f.startswith(".")} for sub in out["dirs"]}
    keys = list(stems)
    if len(keys) >= 2:
        inter = set.intersection(*stems.values())
        union = set.union(*stems.values())
        out["stems_common"] = len(inter)
        out["stems_union"] = len(union)
        out["stems_only_some"] = sorted(union - inter)[:20]
    out["status"] = "OK" if not bad and not (out.get("stems_only_some")) else "PROBLEM"
    return out


def main():
    REPORT["phase2"] = verify_phase2()
    REPORT["phase1_train_zip"] = verify_zip(
        "train", os.path.join(P1, "训练集", "train.zip"),
        os.path.join(P1, "训练集"))
    REPORT["phase1_test_zip"] = verify_zip(
        "test", os.path.join(P1, "测试集", "test.zip"),
        os.path.join(P1, "测试集"))
    print(json.dumps(REPORT, ensure_ascii=False, indent=1))
    with open("./runs/cleanup-verify.json",
              "w") as fh:
        json.dump(REPORT, fh, ensure_ascii=False, indent=1)
    bad = [k for k, v in REPORT.items() if v.get("status") != "OK"]
    print("\n=== 结论:", "全部通过，ZIP 可安全删除" if not bad else f"存在问题: {bad}", "===")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
