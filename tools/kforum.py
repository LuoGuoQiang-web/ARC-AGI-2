#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""kforum -- read a Kaggle competition's discussion forum into files you can actually grep.

Why this exists: Kaggle's web pages are JS-only, so `web_fetch` returns a 5 KB shell with no
content. The CLI, however, talks to the real API. This wraps it, handles the pagination token
that the CLI prints to stdout (which corrupts `--format json` if you do not strip it), and
saves one file per topic so that later passes can search the corpus instead of re-fetching it.

Discussions are where the actual methods live -- the difference between a leaderboard score and
an explanation of it is almost always a forum post.

Usage
-----
    python tools/kforum.py list  arc-prize-2026-arc-agi-2
    python tools/kforum.py dump  arc-prize-2026-arc-agi-2            # every topic's messages
    python tools/kforum.py dump  arc-prize-2026-arc-agi-2 --top 20   # only the top-voted 20
    python tools/kforum.py grep  "test-time training"                # search what you saved
    python tools/kforum.py show  arc-prize-2026-arc-agi-2 740808     # one topic to stdout
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_OUT = HERE.parent / "_forums"
TOKEN_RE = re.compile(r"^\s*Next Page Token\s*=\s*(\S+)\s*$")


def cli(args: list[str]) -> str:
    """Run the kaggle CLI and return stdout with the pagination trailer removed."""
    p = subprocess.run(["kaggle", *args], capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    out = (p.stdout or "") + (p.stderr or "")
    lines = [ln for ln in out.splitlines() if not TOKEN_RE.match(ln)]
    return "\n".join(lines).strip()


def next_token(raw: str) -> str | None:
    for ln in raw.splitlines():
        m = TOKEN_RE.match(ln)
        if m:
            return m.group(1)
    return None


def cmd_list(args) -> int:
    raw_all, token, page = [], None, 0
    while True:
        a = ["competitions", "topics", "list", "-c", args.competition, "--format", "json"]
        if token:
            a += ["--page-token", token]
        p = subprocess.run(["kaggle", *a], capture_output=True, text=True,
                           encoding="utf-8", errors="replace")
        raw = (p.stdout or "") + (p.stderr or "")
        token = next_token(raw)
        body = "\n".join(ln for ln in raw.splitlines() if not TOKEN_RE.match(ln)).strip()
        try:
            chunk = json.loads(body)
        except Exception:
            break
        if not chunk:
            break
        raw_all.extend(chunk)
        page += 1
        print(f"  page {page}: +{len(chunk)} (total {len(raw_all)})")
        if not token or page > 40:
            break

    out = DEFAULT_OUT / f"{args.competition}_topics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(raw_all, indent=1, ensure_ascii=False), encoding="utf-8")
    raw_all.sort(key=lambda t: -(t.get("votes") or 0))
    print(f"\n{len(raw_all)} topics -> {out}\n")
    print(f"{'id':>8} {'votes':>6} {'msgs':>5}  title")
    for t in raw_all:
        print(f"{t.get('id'):>8} {t.get('votes') or 0:>6} {t.get('commentCount') or 0:>5}  "
              f"{(t.get('title') or '')[:110]}")
    return 0


def cmd_dump(args) -> int:
    src = DEFAULT_OUT / f"{args.competition}_topics.json"
    if not src.exists():
        print(f"run `list` first ({src} missing)")
        return 1
    topics = json.loads(src.read_text(encoding="utf-8"))
    topics.sort(key=lambda t: -(t.get("votes") or 0))
    if args.top:
        topics = topics[:args.top]
    if args.min_comments:
        topics = [t for t in topics if (t.get("commentCount") or 0) >= args.min_comments]

    dest = DEFAULT_OUT / args.competition
    dest.mkdir(parents=True, exist_ok=True)
    ok = 0
    for t in topics:
        tid = t.get("id")
        f = dest / f"{tid}.json"
        if f.exists() and not args.force:
            ok += 1
            continue
        body = cli(["competitions", "topic-messages", args.competition, str(tid),
                    "-n", "-1", "--format", "json"])
        try:
            msgs = json.loads(body) if body else []
        except Exception:
            msgs = []
        f.write_text(json.dumps({"topic": t, "messages": msgs}, indent=1, ensure_ascii=False),
                     encoding="utf-8")
        print(f"  {tid}: {len(msgs)} message(s)  {(t.get('title') or '')[:80]}")
        ok += 1
    print(f"\n{ok} topic file(s) under {dest}")
    return 0


def _walk_text(obj) -> str:
    """Flatten a message payload to plain text so grep sees prose, not JSON escaping."""
    if isinstance(obj, str):
        return obj
    if isinstance(obj, dict):
        return "\n".join(_walk_text(v) for v in obj.values())
    if isinstance(obj, list):
        return "\n".join(_walk_text(v) for v in obj)
    return str(obj)


def cmd_grep(args) -> int:
    pat = re.compile(args.pattern, re.I)
    hits = 0
    for f in sorted(DEFAULT_OUT.glob("*/*.json")):
        data = json.loads(f.read_text(encoding="utf-8"))
        text = _walk_text(data.get("messages") or [])
        title = (data.get("topic") or {}).get("title", "")
        for para in re.split(r"\n{2,}", text):
            if pat.search(para):
                hits += 1
                if hits <= args.limit:
                    print(f"\n--- {f.name}  [{title[:70]}] ---")
                    print(" ".join(para.split())[:1400])
    print(f"\n{hits} matching paragraph(s)")
    return 0


def cmd_show(args) -> int:
    f = DEFAULT_OUT / args.competition / f"{args.topic_id}.json"
    if not f.exists():
        print(f"not dumped: {f}")
        return 1
    print(_walk_text(json.loads(f.read_text(encoding="utf-8"))))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("list"); p.add_argument("competition"); p.set_defaults(fn=cmd_list)

    p = sub.add_parser("dump"); p.add_argument("competition")
    p.add_argument("--top", type=int, default=0, help="only the N most-voted topics")
    p.add_argument("--min-comments", type=int, default=0)
    p.add_argument("--force", action="store_true", help="re-fetch topics already on disk")
    p.set_defaults(fn=cmd_dump)

    p = sub.add_parser("grep"); p.add_argument("pattern")
    p.add_argument("--limit", type=int, default=40); p.set_defaults(fn=cmd_grep)

    p = sub.add_parser("show"); p.add_argument("competition"); p.add_argument("topic_id", type=int)
    p.set_defaults(fn=cmd_show)

    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
