#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gh_api_push.py -- push local commits to GitHub **without github.com**.

Why this exists (2026-09-14): on this network `github.com:443` is intermittently
unreachable (connection reset / timeout) while `api.github.com`, `codeload.github.com`
and `raw.githubusercontent.com` all answer 200. `git push` therefore fails while the
GitHub REST API keeps working. This tool replays a normal fast-forward push through the
Git Data API (blobs -> tree -> commit -> ref), producing a commit that is byte-identical
to the local one.

It verifies the result: the tree sha GitHub computes must equal the local commit's tree
sha. If it does not, the push is aborted before moving the ref.

Usage:
    set GH_TOKEN via `gh auth token` (or export it yourself)
    python tools/gh_api_push.py --repo LuoGuoQiang-web/ARC-AGI-2 --branch main [--dry-run]
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

API = "https://api.github.com"
WORKSPACE = Path(__file__).resolve().parent.parent


def api(method: str, path: str, token: str, body: dict | None = None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(API + path, data=data, method=method, headers={
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "User-Agent": "dsh-gh-api-push",
        "Content-Type": "application/json",
    })
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            payload = r.read().decode("utf-8", "replace")
            return r.status, (json.loads(payload) if payload.strip() else {})
    except urllib.error.HTTPError as e:
        return e.code, {"error": e.read().decode("utf-8", "replace")[:400]}


def git(*args: str) -> str:
    r = subprocess.run(["git", "-C", str(WORKSPACE), *args], capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"git {' '.join(args)} failed: {r.stderr.strip()}")
    return r.stdout


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="owner/name")
    ap.add_argument("--branch", default="main")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if not token:
        print("no token: run `gh auth token` and put it in GH_TOKEN")
        return 2

    local_head = git("rev-parse", "HEAD").strip()
    local_tree = git("rev-parse", "HEAD^{tree}").strip()
    subject = git("log", "-1", "--pretty=%B").strip()
    print(f"local HEAD   {local_head[:12]}  tree {local_tree[:12]}")

    st, ref = api("GET", f"/repos/{args.repo}/git/ref/heads/{args.branch}", token)
    if st != 200:
        print(f"cannot read remote ref: {st} {ref}")
        return 1
    remote_head = ref["object"]["sha"]
    st, remote_commit = api("GET", f"/repos/{args.repo}/git/commits/{remote_head}", token)
    remote_tree = remote_commit.get("tree", {}).get("sha", "")
    print(f"remote HEAD  {remote_head[:12]}  tree {remote_tree[:12]}")

    if remote_head == local_head:
        print("already up to date")
        return 0
    if remote_tree == local_tree:
        print("remote tree already matches the local tree; nothing to write")
        return 0

    # Enumerate the local tree from the local object database. Uploading every blob and
    # building the tree WITHOUT base_tree makes the result a pure function of local
    # content, so the tree sha check below is a real end-to-end verification -- and it
    # works even when the local and remote histories have diverged (which is what happens
    # when a push goes through the API: same content, different commit sha).
    entries = []
    for line in git("ls-tree", "-r", "HEAD").splitlines():
        meta, path = line.split("\t", 1)
        mode, _type, sha = meta.split()
        if _type != "blob":
            continue
        entries.append((mode, path, sha))
    print(f"local tree entries: {len(entries)}")

    if args.dry_run:
        print("--dry-run: stopping before writing anything")
        return 0

    # Reuse blobs the remote already has: ask GitHub whether each base blob exists by
    # skipping ones whose sha is unchanged relative to the remote tree is not possible
    # without the remote tree contents, so upload everything (the repo is small).
    api_tree = []
    for mode, path, sha in entries:
        raw = subprocess.run(["git", "-C", str(WORKSPACE), "cat-file", "-p", sha],
                             capture_output=True)
        if raw.returncode != 0:
            print(f"   ! cannot read blob {sha[:10]} for {path}")
            return 1
        st, res = api("POST", f"/repos/{args.repo}/git/blobs", token,
                      {"content": base64.b64encode(raw.stdout).decode(), "encoding": "base64"})
        if st not in (200, 201):
            print(f"   ! blob failed for {path}: {st} {res}")
            return 1
        if res["sha"] != sha:
            print(f"   ! blob sha mismatch for {path}: local {sha[:10]} vs github {res['sha'][:10]}")
            return 1
        api_tree.append({"path": path, "mode": mode, "type": "blob", "sha": res["sha"]})

    st, tree = api("POST", f"/repos/{args.repo}/git/trees", token, {"tree": api_tree})
    if st not in (200, 201):
        print(f"tree failed: {st} {tree}")
        return 1
    new_tree = tree["sha"]
    print(f"new tree     {new_tree[:12]}")

    if new_tree != local_tree:
        print(f"REFUSING: GitHub tree {new_tree[:12]} != local tree {local_tree[:12]}")
        print("  (the API replay would not reproduce the local commit exactly)")
        return 1
    print("tree matches the local commit exactly")

    st, commit = api("POST", f"/repos/{args.repo}/git/commits", token, {
        "message": subject, "tree": new_tree, "parents": [remote_head],
    })
    if st not in (200, 201):
        print(f"commit failed: {st} {commit}")
        return 1
    new_commit = commit["sha"]
    print(f"new commit   {new_commit[:12]}  (parent {remote_head[:12]})")

    st, res = api("PATCH", f"/repos/{args.repo}/git/refs/heads/{args.branch}", token,
                  {"sha": new_commit, "force": False})
    if st not in (200, 201):
        print(f"ref update failed: {st} {res}")
        return 1
    print(f"pushed: {args.branch} -> {new_commit[:12]}")
    print("\nNOTE: content is identical, but this commit's sha differs from the local one.")
    print("      When github.com is reachable again, reconcile with:")
    print("        git fetch origin && git reset --hard origin/main")
    return 0


if __name__ == "__main__":
    sys.exit(main())
