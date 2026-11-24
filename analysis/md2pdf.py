"""Markdown -> HTML -> PDF（Chrome headless），用于生成《技术方案.PDF》。

只用本机已有工具：python-markdown + Google Chrome，中文字体走系统 PingFang/STHeiti。
"""
import argparse
import html
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import markdown

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

CSS = """
@page { size: A4; margin: 18mm 16mm 18mm 16mm; }
body { font-family: "PingFang SC", "STHeiti", "Heiti SC", "Songti SC", serif;
       font-size: 10.5pt; line-height: 1.65; color: #111; }
h1 { font-size: 19pt; border-bottom: 2px solid #222; padding-bottom: 6px;
     margin: 0 0 14px 0; }
h2 { font-size: 14pt; margin: 20px 0 8px 0; border-left: 4px solid #444;
     padding-left: 8px; page-break-after: avoid; }
h3 { font-size: 12pt; margin: 14px 0 6px 0; page-break-after: avoid; }
h4 { font-size: 11pt; margin: 12px 0 4px 0; page-break-after: avoid; }
p, li { margin: 4px 0; text-align: justify; }
code { font-family: "SF Mono", Menlo, monospace; font-size: 9pt;
       background: #f4f4f4; padding: 1px 3px; border-radius: 3px; }
pre { background: #f6f8fa; border: 1px solid #ddd; border-radius: 4px;
      padding: 8px 10px; font-size: 8.5pt; line-height: 1.4;
      white-space: pre-wrap; word-break: break-all; page-break-inside: avoid; }
pre code { background: none; padding: 0; }
table { border-collapse: collapse; width: 100%; margin: 8px 0;
        font-size: 9pt; page-break-inside: avoid; }
th, td { border: 1px solid #bbb; padding: 4px 6px; text-align: left;
         word-break: break-word; }
th { white-space: nowrap; }
th:first-child, td:first-child { white-space: nowrap; }
th { background: #eee; font-weight: 600; }
blockquote { border-left: 3px solid #bbb; margin: 8px 0; padding: 2px 10px;
             color: #444; background: #fafafa; }
img { max-width: 100%; }
hr { border: none; border-top: 1px solid #ccc; margin: 14px 0; }
"""

TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>{title}</title><style>{css}</style></head>
<body>{body}</body></html>
"""


def md_to_html(md_text, title):
    body = markdown.markdown(
        md_text,
        extensions=["tables", "fenced_code", "toc", "sane_lists", "attr_list"],
    )
    return TEMPLATE.format(title=html.escape(title), css=CSS, body=body)


def html_to_pdf(html_text, out_pdf, timeout=180):
    out_pdf = Path(out_pdf).resolve()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "doc.html"
        src.write_text(html_text, encoding="utf-8")
        profile = Path(td) / "chrome-profile"
        profile.mkdir()
        cmd = [CHROME, "--headless=new", "--disable-gpu", "--no-sandbox",
               f"--user-data-dir={profile}",
               "--no-pdf-header-footer", "--run-all-compositor-stages-before-draw",
               "--virtual-time-budget=20000",
               f"--print-to-pdf={out_pdf}", src.as_uri()]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            stderr = proc.stderr
        except subprocess.TimeoutExpired as exc:
            # Chrome 写完 PDF 后偶尔不退出：只要文件已生成就当作成功。
            subprocess.run(["pkill", "-f", f"user-data-dir={profile}"],
                           capture_output=True)
            stderr = (exc.stderr or b"").decode("utf-8", "replace")[-600:]
        if not out_pdf.exists() or out_pdf.stat().st_size == 0:
            # 退一步：用 cupsfilter（系统自带）走纯文本路径
            print(f"[warn] Chrome 失败，改用 cupsfilter:\n{stderr}", file=sys.stderr)
            rtf = Path(td) / "doc.txt"
            rtf.write_text(re.sub(r"<[^>]+>", "", html_text), encoding="utf-8")
            with out_pdf.open("wb") as fh:
                subprocess.run(["/usr/sbin/cupsfilter", "-m", "application/pdf", str(rtf)],
                               stdout=fh, stderr=subprocess.DEVNULL, timeout=timeout)
    if not out_pdf.exists() or out_pdf.stat().st_size == 0:
        raise RuntimeError(f"Chrome 未能生成 PDF:\n{proc.stderr[-2000:]}")
    return out_pdf


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--md", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default=None)
    args = ap.parse_args()
    md_path = Path(args.md)
    title = args.title or md_path.stem
    pdf = html_to_pdf(md_to_html(md_path.read_text(encoding="utf-8"), title), args.out)
    print(f"{pdf}  {pdf.stat().st_size} bytes")


if __name__ == "__main__":
    sys.exit(main())
