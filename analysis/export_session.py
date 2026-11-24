"""把 DSH 会话记录（zstd 压缩的 JSONL 事件流）导出为可读的 Markdown 记录。

原始文件：$DSH_HOME/sessions/<工作区>/<会话ID>/session.v2.jsonl.zstd
事件类型：user/message, assistant/message, tool/call, tool/result, compaction/prune ...

    python3 export_session.py                      # 导出本会话
    python3 export_session.py --no-reasoning       # 不含内部推理（体积小很多）
"""
import argparse
import json
import os
import zstandard
from pathlib import Path

DSH_HOME = Path(os.environ.get("DSH_HOME", Path.home() / ".dsh"))
SESSION_ID = os.environ.get("DSH_SESSION_ID", "")


def find_session(session_id=None):
    sid = session_id or SESSION_ID
    hits = list((DSH_HOME / "sessions").rglob(f"{sid}/session.v2.jsonl.zstd")) if sid else []
    if not hits:
        hits = sorted((DSH_HOME / "sessions").rglob("session.v2.jsonl.zstd"),
                      key=lambda p: p.stat().st_mtime, reverse=True)
    if not hits:
        raise SystemExit("找不到会话记录")
    return hits[0]


def text_of(content):
    """把 message.content 数组拼成纯文本。"""
    out = []
    if isinstance(content, str):
        return content
    for part in content or []:
        t = part.get("type")
        if t == "text":
            out.append(part.get("text", ""))
        elif t == "reasoning":
            out.append(("<<REASONING>>", part.get("text", "")))
        elif t == "tool-result":
            for sub in part.get("content", []):
                if sub.get("type") == "text":
                    out.append(sub.get("text", ""))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--session", default=None)
    ap.add_argument("--out", default="dsh-transcript.md")
    ap.add_argument("--no-reasoning", action="store_true")
    ap.add_argument("--max-tool-chars", type=int, default=4000)
    args = ap.parse_args()

    src = find_session(args.session)
    raw = zstandard.ZstdDecompressor().stream_reader(open(src, "rb")).read()
    recs = [json.loads(l) for l in raw.split(b"\n") if l.strip()]

    # ---- 先扫一遍，生成目录索引 ----
    toc, turn_no = [], 0
    t0 = t1 = None
    for r in recs:
        d = r.get("data") or {}
        if r.get("type") == "turn/start":
            turn_no = d.get("turn", turn_no)
        elif r.get("type") == "user/message":
            parts = text_of(d.get("content"))
            body = parts if isinstance(parts, str) else "".join(
                p for p in parts if isinstance(p, str))
            body = " ".join(body.split())
            if body.startswith("Current runtime context"):
                continue
            ts = r.get("time")
            if isinstance(ts, (int, float)):
                import datetime
                t0 = t0 or ts
                t1 = ts
                stamp = datetime.datetime.fromtimestamp(ts / 1000).strftime("%m-%d %H:%M")
            else:
                stamp = ""
            toc.append((turn_no, stamp, body[:90]))
    import datetime as _dt
    span = ""
    if t0 and t1:
        a = _dt.datetime.fromtimestamp(t0 / 1000).strftime("%Y-%m-%d %H:%M")
        b = _dt.datetime.fromtimestamp(t1 / 1000).strftime("%Y-%m-%d %H:%M")
        span = f"{a} → {b}"

    lines = ["# DSH 会话完整记录", "",
             f"- 会话 ID: `{src.parent.name}`",
             f"- 时间跨度: {span}（北京时间）",
             f"- 轮数: {max(toc[i][0] for i in range(len(toc))) if toc else 0}",
             f"- 原始文件: `{src}`",
             f"- 记录条数: {len(recs)}，解压后 {len(raw):,} 字节", "",
             "## 目录", ""]
    for turn, stamp, preview in toc:
        lines.append(f"- **第 {turn} 轮** {stamp} — {preview}")
    lines += ["", "---", ""]


    stats = {"user": 0, "assistant": 0, "tool": 0, "reasoning": 0}
    for r in recs:
        t, d = r.get("type"), r.get("data") or {}
        if t == "turn/start":
            lines += ["", "---", "", f"## 第 {d.get('turn', '?')} 轮", ""]
        elif t == "user/message":
            stats["user"] += 1
            parts = text_of(d.get("content"))
            body = parts if isinstance(parts, str) else "".join(
                p for p in parts if isinstance(p, str))
            lines += ["### 👤 用户", "", body, ""]
        elif t == "assistant/message":
            stats["assistant"] += 1
            parts = text_of((d.get("message") or {}).get("content"))
            if isinstance(parts, str):
                parts = [parts]
            txt, rea = [], []
            i = 0
            while i < len(parts):
                p = parts[i]
                if isinstance(p, tuple) and p[0] == "<<REASONING>>":
                    rea.append(p[1]); stats["reasoning"] += 1
                elif isinstance(p, str) and p.strip():
                    txt.append(p)
                i += 1
            if rea and not args.no_reasoning:
                lines += ["<details><summary>内部推理</summary>", "",
                          "```", "\n\n".join(rea)[:20000], "```", "", "</details>", ""]
            if txt:
                lines += ["### 🤖 助手", "", "\n\n".join(txt), ""]
        elif t == "tool/call":
            stats["tool"] += 1
            name = d.get("name", "?")
            try:
                a = json.loads(d.get("arguments") or "{}")
            except Exception:
                a = d.get("arguments")
            pretty = json.dumps(a, ensure_ascii=False, indent=1)
            if len(pretty) > args.max_tool_chars:
                pretty = pretty[:args.max_tool_chars] + "\n…（已截断）"
            lines += [f"**🔧 调用 `{name}`**", "", "```json", pretty, "```", ""]
        elif t == "tool/result":
            msg = (d.get("message") or {})
            parts = text_of(msg.get("content"))
            if isinstance(parts, str):
                parts = [parts]
            body = "\n".join(p for p in parts if isinstance(p, str))
            if len(body) > args.max_tool_chars:
                body = (body[: args.max_tool_chars // 2] + "\n…（中间省略）…\n"
                        + body[-args.max_tool_chars // 2:])
            lines += ["```", body.rstrip(), "```", ""]
        elif t == "compaction/prune":
            lines += ["> ⚠️ 此处发生一次上下文压缩", ""]

    out = Path(args.out)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"已导出: {out}  ({out.stat().st_size:,} 字节)")
    print(f"  用户消息 {stats['user']} 条 / 助手回复 {stats['assistant']} 条 / "
          f"工具调用 {stats['tool']} 次 / 推理块 {stats['reasoning']} 个")


if __name__ == "__main__":
    main()
