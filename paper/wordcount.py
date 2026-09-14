#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Count the Paper Track Writeup against Kaggle's 1,500-word cap.

The competition page states verbatim: "Your Writeup should not exceed 1,500 words.
Submissions over this limit may be subject to penalty." Two open questions the organizers
never answered (topic 696513, open since 2026-05-02): whether a bibliography counts, and
whether the cap applies to a PDF attached in Project Links. We therefore budget against the
strict reading -- every whitespace-separated token, tables and code included.

Run:  python paper/wordcount.py [path]
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

CAP = 1500


def counts(text: str) -> dict:
    toks = lambda s: len(re.findall(r"\S+", s))
    no_code = re.sub(r"```.*?```", "", text, flags=re.S)
    no_tables = re.sub(r"^\|.*$", "", no_code, flags=re.M)
    no_html = re.sub(r"<!--.*?-->", "", no_tables, flags=re.S)
    # Markdown syntax is not prose: strip heading hashes, emphasis and link targets.
    prose = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", no_html)
    prose = re.sub(r"[*_`#>]", "", prose)
    return {
        "everything": toks(text),
        "no_code": toks(no_code),
        "no_code_or_tables": toks(no_tables),
        "prose_only": toks(prose),
    }


def main() -> int:
    p = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name(
        "ARC_PRIZE_2026_PAPER.md")
    if not p.exists():
        print(f"missing {p}")
        return 1
    c = counts(p.read_text(encoding="utf-8"))
    print(f"{p}   (cap = {CAP} words, strict reading)\n")
    for k, v in c.items():
        over = v - CAP
        flag = f"OVER by {over}" if over > 0 else f"{abs(over)} to spare"
        print(f"  {k:20} {v:6}   {flag}")
    strict = c["everything"]
    print(f"\nstrict count {strict} vs cap {CAP}: "
          + ("PASS" if strict <= CAP else f"FAIL -- must cut {strict - CAP} words"))
    print("\nWe budget against the strict reading (every token, tables and code included),")
    print("because the two questions that would relax it are unanswered by the organizers.")
    return 0 if strict <= CAP else 2


if __name__ == "__main__":
    raise SystemExit(main())
