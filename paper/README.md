# Paper Track — submission guide and reproduction

## Files

| file | role |
|---|---|
| **`ARC_PRIZE_2026_WRITEUP.md`** | **The submission artefact.** 1,445 words, within Kaggle's hard 1,500-word cap. This is what goes into the Kaggle Writeup editor. |
| `ARC_PRIZE_2026_PAPER.md` | ⚠️ **Superseded — do not attach.** The longer draft the Writeup was condensed from. Its title and its "memory-infeasible on a 14.56 GiB T4" claim are now known to be wrong (banner at the top of that file). Attaching it would contradict §4.4 of the Writeup. |
| `PAPER_TRACK_SUBMISSION_MECHANISM.md` | How the submission actually works, with sources. |
| `wordcount.py` | Checks the cap mechanically. Run before every edit. |

```bash
python paper/wordcount.py paper/ARC_PRIZE_2026_WRITEUP.md
```

We budget against the **strict** reading — every whitespace token, tables and code included —
because the two questions that would relax it (does a bibliography count; does the cap apply to an
attached PDF) have been unanswered by the organizers since 2026-05-02.

## How the submission works (verified, not assumed)

It is a **Kaggle "Hackathon Writeup"** — not a PDF upload, not a discussion post, not a form. 2024
and 2025 used a Google Form on a hidden page named `"Paper Award "` (trailing space); that route
does not exist for 2026.

1. Go to **<https://www.kaggle.com/competitions/arc-prize-2026-paper-track/projects>**
2. Click **"New Writeup"**.
3. Fill in **title**, **subtitle**, and the body (paste `ARC_PRIZE_2026_WRITEUP.md`; ≤ 1,500 words).
4. **Media Gallery** — a **cover image is required** before the Submit button appears.
5. **Attach the public notebook** in the **`Project Links`** field. It must be publicly accessible
   without a login or paywall. A private Kaggle notebook is auto-published after the deadline, but
   do not rely on that.
6. **`Public Project Link`** — *optional*, and it is a **link field, not a file picker**. The page
   says: *"instead of using the Kaggle Writeup, you can upload a PDF version of your paper using the
   Public Project Link feature. It should be publicly accessible and not require a login or
   paywall."* So it takes a **publicly reachable URL**. If we attach a PDF, host it somewhere public
   (e.g. `raw.githubusercontent.com` of our public repo) and paste that URL.
7. **Select a Track** — required, or the Writeup cannot be submitted.
8. **Save**, then **Submit** (top right).

### Three traps

- **One submission per team.** The CLI reports 5/day; the competition rules say one. **Never
  test-submit.**
- **Saved ≠ submitted.** "Any un-submitted or draft Writeups by the competition deadline will not
  be considered by the Judges." Afterwards, verify that
  `kaggle competitions submissions -c arc-prize-2026-paper-track` no longer returns
  "No submissions found".
- **Deadline conflict.** Kaggle's Timeline and API say **2026-11-09T23:59Z**; arcprize.org's Key
  Dates say **"November 8, 2026 — Papers due"**. Unresolved in the forum. **Target Nov 8.**

## Where the fields and the submission ID live

**`Public Project Link`, `Project Links` and the Media Gallery** are all fields **inside the Writeup
editor**, reachable only via **New Writeup** at the `/projects` URL above. They are not on the
Overview, Data or Rules tabs. The page is client-side rendered, so `web_fetch` returns an empty
shell — a human has to open it in a browser.

**The ARC-AGI-2 submission ID** is the `ref` column of the submissions list:

```bash
kaggle competitions submissions -c arc-prize-2026-arc-agi-2
```

which currently reports:

| ref (submission ID) | date | status | score |
|---|---|---|---|
| `56221023` | 2026-09-14 04:01 | COMPLETE, no error | **0.42** |
| `56199696` | 2026-09-13 05:17 | COMPLETE, **format error** | — |

The same list is on the web at
<https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/submissions>.

The Paper Track's Submission Requirements **do not ask for a submission ID field** — the linkage to
the code submission is the attached public notebook. The ID matters for the ARC-AGI-2 **Innovation
Prize** solution writeup, not for the Paper Track.

## The second shot

The host has stated *"A unified paper is allowed — it will be judged according to the rubric as
normal"*, and the ARC-AGI-2 Innovation Prize ($275k) uses **the same six-criterion rubric** as the
Paper Prize. The same content can go to both: two independent chances, no extra cost. There is also
**no prohibition on arXiv or prior publication** — the 2025 first-place paper was on arXiv about
four weeks before its deadline.

## Reproducing every number (no GPU needed)

```bash
cd work/arc_w1
pip install numpy
kaggle competitions download -c arc-prize-2026-arc-agi-2 -p comp_data   # then unzip

python tools/evidence_ledger.py          # every cited value + the file it came from
python tools/recompute_pool_recall.py    # §4.1, from the committed run reports
python tools/verify_leak.py comp_data    # §4.5
python tests/test_model_free.py          # 59 checks
python tests/test_merge_shards.py        # 20
python tests/test_engine_integration.py  # 11
```

If a number is not in `evidence_ledger.py`'s output, it should not be in the paper.
