#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Render the Writeup to a PDF for the optional `Public Project Link` field.

Why this exists: that field asks for "a PDF version of your paper ... publicly accessible and
not requiring a login or paywall". The long draft (`ARC_PRIZE_2026_PAPER.md`) is **superseded**
and contains two claims now known to be wrong, so it must not be attached. Rather than maintain a
second document by hand -- which is how the two would drift apart -- this renders the *submission
artefact itself*, `ARC_PRIZE_2026_WRITEUP.md`, so the PDF and the Kaggle Writeup are by
construction the same words.

Deliberately minimal markdown: headings, paragraphs, bold, and the numbered reference list. No
tables (the Writeup has none) and no images.

Run:  python paper/make_pdf.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "ARC_PRIZE_2026_WRITEUP.md"
OUT = HERE / "ARC_PRIZE_2026_WRITEUP.pdf"

# fpdf's built-in fonts are latin-1 only, and this document uses arrows, relations and dashes.
FONT_CANDIDATES = [
    ("DejaVu", Path(r"C:\Windows\Fonts\DejaVuSans.ttf"),
     Path(r"C:\Windows\Fonts\DejaVuSans-Bold.ttf")),
    ("Arial", Path(r"C:\Windows\Fonts\arial.ttf"), Path(r"C:\Windows\Fonts\arialbd.ttf")),
    ("Calibri", Path(r"C:\Windows\Fonts\calibri.ttf"), Path(r"C:\Windows\Fonts\calibrib.ttf")),
]


def pick_font():
    for name, regular, bold in FONT_CANDIDATES:
        if regular.exists() and bold.exists():
            return name, regular, bold
    return None, None, None


def main() -> int:
    if not SRC.exists():
        print(f"missing {SRC}")
        return 1
    try:
        from fpdf import FPDF
    except ImportError:
        print("need fpdf2:  python -m pip install fpdf2")
        return 1

    text = SRC.read_text(encoding="utf-8")
    # Strip an optional leading H1-free preamble; keep everything else verbatim.
    lines = text.splitlines()

    name, regular, bold = pick_font()
    pdf = FPDF(format="A4", unit="mm")
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.set_margins(20, 18, 20)
    pdf.add_page()
    if name:
        pdf.add_font(name, "", str(regular))
        pdf.add_font(name, "B", str(bold))
        family = name
    else:
        family = "Helvetica"
        # No Unicode font: fall back and degrade the few non-latin-1 glyphs rather than crash.
        text = (text.replace("\u2192", "->").replace("\u2264", "<=")
                    .replace("\u00d7", "x").replace("\u2014", "--").replace("\u2013", "-"))
        lines = text.splitlines()

    def clean(s: str) -> str:
        # fpdf2 understands **bold** via markdown=True; pass it through untouched.
        return s

    for raw in lines:
        line = raw.rstrip()
        if not line.strip():
            pdf.ln(2.5)
            continue
        # fpdf2 leaves the cursor where the previous cell ended, so w=0 ("to the right margin")
        # can collapse to zero width and raise "Not enough horizontal space". Reset explicitly.
        pdf.set_x(pdf.l_margin)
        m = re.match(r"^(#{1,6})\s+(.*)$", line)
        if m:
            level, body = len(m.group(1)), m.group(2).strip()
            sizes = {1: 16, 2: 12.5, 3: 11, 4: 10.5}
            pdf.ln(2 if level > 1 else 0)
            pdf.set_font(family, "B", sizes.get(level, 11))
            pdf.multi_cell(0, 6.0, clean(body), markdown=True)
            pdf.ln(1.5)
            continue
        if line.strip() == "---":
            pdf.ln(1)
            continue
        pdf.set_font(family, "", 9.5)
        pdf.multi_cell(0, 4.9, clean(line.strip()), markdown=True)

    pdf.output(str(OUT))
    words = len(re.findall(r"\S+", text))
    print(f"wrote {OUT}")
    print(f"  source : {SRC.name}  ({words} words -- must stay <= 1500)")
    print(f"  font   : {family}{'' if name else '  (no unicode font found; glyphs degraded)'}")
    print(f"  pages  : {pdf.pages_count}")
    if words > 1500:
        print(f"  !! OVER THE CAP by {words - 1500}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
