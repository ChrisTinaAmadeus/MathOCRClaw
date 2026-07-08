from __future__ import annotations

import argparse
import json
from pathlib import Path

from proofread.md_utils import split_page_into_blocks

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--page-md", required=True)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    md = Path(args.page_md).read_text(encoding="utf-8", errors="ignore")
    blocks = split_page_into_blocks(md)
    obj = [{"i": i, "kind": b.kind, "qno": b.qno, "text": b.text} for i, b in enumerate(blocks)]
    s = json.dumps(obj, ensure_ascii=False, indent=2)
    if args.out:
        Path(args.out).write_text(s, encoding="utf-8")
    else:
        print(s)

if __name__ == "__main__":
    main()

