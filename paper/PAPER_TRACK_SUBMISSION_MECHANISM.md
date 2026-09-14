# ARC Prize 2026 Paper Track — the literal submission mechanism

**Verified on 2026-09-14** against the live Kaggle API/CLI and arcprize.org.
Every claim below carries either a command (+ its output) or a URL we actually fetched.
Anything not backed that way is marked **UNVERIFIED** or **INFERRED**.

---

## 0. Headline

The 2026 Paper Track is **not** a PDF upload, **not** a discussion post, and **not** a Google Form.
It is a **Kaggle "Hackathon Writeup"**: a first-class submission object created *inside* the
`arc-prize-2026-paper-track` competition, which you then **Submit** with a button on that Writeup.

There is no file to upload and no leaderboard. `kaggle competitions submit` is **not** the route.

---

## 1. The literal submission mechanism — VERIFIED

### 1.1 Primary source: the competition's own "Submission Requirements" page

```powershell
kaggle competitions pages -c arc-prize-2026-paper-track --content --page-name "Submission Requirements"
```

Verbatim content:

> **Submission Requirements**
> A valid submission must contain the following:
>
> 1. Kaggle Writeup
>    1. Media Gallery
>    2. Attached Public Notebook
>    3. (Optional) Attached Project Link
>
> **Your final Submission must be made prior to the deadline. Any un-submitted or draft Writeups by the competition deadline will not be considered by the Judges.**
>
> To create a new Writeup, click on the "New Writeup" button [here](https://www.kaggle.com/competitions/arc-prize-2026-paper-track/projects). After you have saved your Writeup, you should see a "Submit" button in the top right corner.
>
> Note: If you attach a private Kaggle Resource to your public Kaggle Writeup, your private Resource will automatically be made public after the deadline.
>
> ### 1. Kaggle Writeup
>
> The Kaggle Writeup serves as your project report. This should include a title, subtitle, and a detailed analysis of your submission to ARC-AGI-2 or ARC-AGI-3. You must select a Track for your Writeup in order to submit.
>
> Your Writeup should not exceed 1,500 words. Submissions over this limit may be subject to penalty.
>
> The below assets must be attached to the Writeup to be eligible.
>
> #### a. Media Gallery
> This is where you should attach any images and/or videos associated with your submission. A cover image is required to submit your Writeup.
>
> #### b. Public Notebook
> Your code should be submitted as a public notebook in the `Project Links` field. Your notebook should be publicly accessible and not require a login or paywall. If you use a private Kaggle Notebook, it will automatically be made public after the deadline.
>
> #### c. (Optional) Public Project Link
> Optionally, instead of using the Kaggle Writeup, you can upload a PDF version of your paper using the Public Project Link feature. It should be publicly accessible and not require a login or paywall.

**This is the complete answer to "the literal submission mechanism".**
The page list that contains it:

```powershell
kaggle competitions pages -c arc-prize-2026-paper-track
# rules / Description / Timeline / Submission Requirements / data-description
# abstract / Evaluation / tracks-and-awards / Bonus Prize / judges
```

### 1.2 What a "Hackathon Writeup" is — VERIFIED

Kaggle's product announcement (fetched via `kaggle forums topics show product-announcements 582328`):

> **Submissions to Kaggle Hackathons take the form of Hackathon Writeups. A Writeup is a type of
> discussion post that contains a rich multimedia gallery and a section for Project Links.**
> With a Hackathon Writeup you can link to and describe any project, including ones that were built
> or deployed off of Kaggle.

Source: <https://www.kaggle.com/discussions/product-announcements/582328> (Addison Howard, Kaggle, 2025-05-30)

The competition rules confirm it is a Hackathon and not a scored competition
(`kaggle competitions pages -c arc-prize-2026-paper-track --content --page-name rules`, line 4):

> For Competitions designated as hackathons by the Competition Sponsor ("Hackathons"), your
> Submissions will be judged by the Competition Sponsor based on the evaluation rubric set forth on
> the Competition Website ("Evaluation Rubric").

and line 135:

> There will be no leaderboards for Hackathon Competitions.

### 1.3 Machine-checkable confirmation — VERIFIED

```powershell
kaggle competitions list --search "arc-prize"
```
```
ref                                                             deadline             category   reward        teamCount  userHasEntered
https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3    2026-11-02 23:59:00  Featured   850,000 Usd        3033  False
https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2    2026-11-02 23:59:00  Featured   700,000 Usd        2007  True
https://www.kaggle.com/competitions/arc-prize-2026-paper-track  2026-11-09 23:59:00  Featured   450,000 Usd         188  True
```

Raw API (`/api/v1/competitions/list?search=arc-prize-2026`) for id `133724`:

| field | value | meaning |
|---|---|---|
| `isKernelsSubmissionsOnly` | `false` | *(see §1.4 — this does NOT mean "upload a file")* |
| `hasEvaluationMetric` | `false` | no metric → no prediction-file scoring |
| `evaluationMetric` | `""` | |
| `awardsPoints` | `false` | Hackathons award no Kaggle points |
| `submissionsDisabled` | `false` | |
| `maxDailySubmissions` | `5` | conflicts with §1.5 |
| `maxTeamSize` | `8` | |
| `licenseName` | `Subject to Competition Rules` | |
| `mergerDeadline` | `2026-11-09T23:59:00Z` | mergers open to the very end |
| `deadline` | `2026-11-09T23:59:00Z` | |
| `teamCount` | `188` | |
| `userHasEntered` | `true` | we are entered |

Supporting read-only checks:

```powershell
kaggle competitions files -c arc-prize-2026-paper-track -v
# name,size,creationDate
# NOTE.md,150,2026-03-13 19:58:56.343000

kaggle competitions submissions -c arc-prize-2026-paper-track
# No submissions found

kaggle competitions leaderboard -c arc-prize-2026-paper-track --show
# No results found

kaggle competitions submission-limits -c arc-prize-2026-paper-track
# Submissions today: 0
# Lifetime submissions: 0
# Remaining today: 5
```

### 1.4 Is the Kaggle "Submit" button functional?

**Yes — but it lives on the Writeup, not on a file-upload form.** The official text says:
*"After you have saved your Writeup, you should see a 'Submit' button in the top right corner."*
There is no evaluation metric and no dataset, so a conventional file/notebook submission is not
the documented route. `isKernelsSubmissionsOnly: false` means only "this competition is not
restricted to Code-Competition notebook submissions", **not** "a CSV upload is expected".

### 1.5 How many submissions?

Two sources disagree:

- Competition-Specific Rules §2 (authoritative), `--page-name rules` line 33:
  > a. For Hackathons, each Team may submit one (1) Submission only.
- The Kaggle API counter says `Remaining today: 5`, and paper-track forum topic 735361 assumes
  "We're also allowed multiple submissions per day."

**Practical rule: treat it as exactly ONE submission. Do not test-submit.**

### 1.6 What is required at deadline — the mandatory "Track" selection and submission ID

`--page-name Evaluation`:

> To be eligible to win the ARC 2026 Paper Award Prize, you must join this competition and submit a
> Writeup that documents your solution for either ARC-AGI-2 or ARC-AGI-3.

> |**Accuracy** | How accurate is the submission based on its performance on the leaderboard?
> (Note: you will be asked to provide your submission ID to match your notebook to your submission) |

So the Writeup must **select a Track** (pick ARC-AGI-2) and the **ARC-AGI-2 submission ID** will be
requested, to bind the paper to the scored entry.

---

## 2. How papers were submitted in 2024 and 2025 — VERIFIED

The mechanism changed completely in 2026. Both prior years used a **Google Form**, and both years'
competitions carry a hidden page named `"Paper Award "` (**note the trailing space** — this is why
`--page-name "Paper Award"` returns "No pages found"; dump all pages and filter instead).

### 2025 — `kaggle competitions pages -c arc-prize-2025 --content` → page `"Paper Award "`

> To be eligible for a Paper Award, you must separately submit a paper (**Kaggle Notebook, PDF,
> arXiv, txt, etc.**) documenting and describing the conceptual approach of your eligible ARC Prize
> 2025 Kaggle submission. Paper submissions must be submitted **within 6 days** of the competition
> ending and **must be public (thus, also open-sourced)**.
> …
> ### Click here to go to the [Submission form](https://docs.google.com/forms/d/e/1FAIpQLSfhmYE6AMYfMBkxj5_G8QHyWTWWEe1wUg98LM1UQsUR8ci-1w/viewform)

### 2024 — `kaggle competitions pages -c arc-prize-2024 --content` → page `"Paper Award "`

Same text except: "**within 48 hours** of the competition ending", and no "must be public" clause.
Same Google Form URL.

### 2026 — the form is gone

```powershell
kaggle competitions pages -c arc-prize-2026-arc-agi-2
# name: rules / data-description / Description / Evaluation / Timeline / Prizes
#       Code Requirements / abstract / Upgraded Accelerators
```
**There is no `Paper Award` page on the 2026 ARC-AGI-2 competition.** The Paper Prize moved to its
own competition (`arc-prize-2026-paper-track`, id 133724) and its submission object is a Writeup.
Note `arc-prize-2025` and `arc-prize-2024` *do* list `Paper Award `; 2026 does not.

### What the winning papers actually looked like

All 2024/2025 paper-award winners are **external links** — arXiv, OpenReview, GitHub PDF, Google
Drive, or a personal site. Only one is a Kaggle writeup.

| Year | Place | Artifact | Form |
|---|---|---|---|
| 2024 | 1st | <https://arxiv.org/abs/2411.02272> | arXiv |
| 2024 | 3rd | `github.com/clement-bonnet/lpn/.../paper.pdf` | GitHub PDF |
| 2025 | 1st | <https://arxiv.org/abs/2510.04871> | arXiv |
| 2025 | 2nd | <https://openreview.net/pdf?id=z4IG090qt2> | OpenReview PDF |
| 2025 | 3rd | <https://iliao2345.github.io/.../ARC_AGI_Without_Pretraining.pdf> | personal site PDF |
| 2025 | 4th (score) | <https://www.kaggle.com/competitions/arc-prize-2025/writeups/arc-prize-2025-competition-writeup-5th-place> | **Kaggle writeup** |

Source: <https://arcprize.org/competitions/2025> and <https://arcprize.org/competitions/2024>.
Confirmed on <https://arcprize.org/blog/arc-prize-2025-results-analysis>: "We also had **90 papers
submitted, up from 47 last year**".

**So: historically a PDF/arXiv link was the whole submission. In 2026 it must be a Kaggle Writeup.**

---

## 3. Page limit and format — PARTLY VERIFIED, PARTLY UNRESOLVED

**VERIFIED (limit):** on the Submission Requirements page:

> Your Writeup should not exceed **1,500 words**. Submissions over this limit may be subject to penalty.

**VERIFIED (no page limit, no template):** No page count, no NeurIPS/ICML template, no "PDF only"
rule appears anywhere on the competition pages, the rules page, or arcprize.org. Asked directly in
paper-track topic 724819 "Paper Writing template", the host (Greg Kamradt, ARC Prize President)
answered only:

> Thank you! You can see previous winning submissions here: <https://arcprize.org/competitions/2025>
> <https://arcprize.org/competitions/2024> — Those should be a good source of inspiration

**VERIFIED (structure required):** `https://arcprize.org/competitions/2026/paper`:

> **What to Include** — Abstract / Intro / Prior work / Approach / Results / Conclusion
> "Shorter and clearer is always better. No filler, no unnecessary equations."

**UNRESOLVED (open questions the organizers never answered).** Paper-track topic 696513
"Bibliography? 1500-word limit?" (asked 2026-05-02) and its follow-up comment (2026-08-15) ask:
does a bibliography count toward the 1,500 words, and does the limit apply to a PDF uploaded via
the Public Project Link instead of writing it in the Writeup editor? **No organizer reply exists
as of the dump.**

**UNRESOLVED (ambiguity in the primary source).** Submission Requirements §c says *"Optionally,
**instead of** using the Kaggle Writeup, you can upload a PDF version of your paper using the
Public Project Link feature"*, while the same page says a valid submission **must** contain
"1. Kaggle Writeup". The two readings contradict each other.

**Safe reading:** write the full paper in the Kaggle Writeup editor, ≤1,500 words, and *additionally*
attach the PDF via Public Project Link. That satisfies both readings.

---

## 4. The "official competition Solution Writeup" for the ARC-AGI-2 Grand Prize

### 4.1 What the rules say — VERIFIED

`https://arcprize.org/competitions/2026/arc-agi-2`:

> **ARC-AGI-2 Grand Prize: $275,000** — The Grand Prize will be awarded to the highest scoring
> Solution Writeup based on the below criteria. **All artifacts should be open sourced and attached
> to an official competition Solution Writeup within seven days of the competition's submission
> deadline** to be considered eligible.

The Kaggle side calls the same prize the **Innovation Prize** and carries identical text
(`kaggle competitions pages -c arc-prize-2026-arc-agi-2 --content --page-name Prizes`, line 33):

> The Innovation Prize will be awarded to the highest scoring Solution Writeup based on the below
> criteria. All artifacts should be open sourced and attached to an official competition
> [Solution Writeup](https://www.kaggle.com/discussions/product-feedback/373153) within seven days
> of the competition's submission deadline to be considered eligible.

`kaggle competitions pages -c arc-prize-2026-arc-agi-2 --content --page-name rules` lines 15–17:

> - Progress Prizes (guaranteed): $275,000
> - Innovation Prize (guaranteed): $275,000
> - Bonus Prize: $150,000

**Deadline arithmetic:** ARC-AGI-2 submission deadline `2026-11-02T23:59:00Z` + 7 days
= **`2026-11-09T23:59:00Z`**, which is *exactly* the Paper Track deadline. The two deadlines coincide.

### 4.2 Where it lives — VERIFIED that "Solution Writeup" is a named Kaggle feature

The linked discussion is the Kaggle feature announcement
(`kaggle forums topics show product-feedback 373153`):

> **[Product Update] Competition Solution Write-Ups: Improving the Way Insights Are Gathered on Kaggle** (DJ Sterling, 2022-12-19)
> …
> **1. Ability to Set Your Team's Official Solution Write-Up**
> Anyone on a competition's leaderboard can **set a forum post as their team's official solution
> write-up post**, making it quicker for readers to find these valuable summaries. Once the
> competition closes, the team leader can set their solution write-up by **copying the URL of your
> forum post** (written by anyone on the team) and **pasting it into their team page**.

So per Kaggle's own definition it is **a forum post on the ARC-AGI-2 competition whose URL is
registered as the team's official solution write-up** — not a PDF and not a separate file upload.

**UNRESOLVED (exact 2026 UI).** ARC Prize 2025's 4th-place team's link used the newer URL shape
`/competitions/arc-prize-2025/writeups/<slug>`, so Kaggle may now expose this through a
**Writeups** tab rather than the 2022 team-page field. Both routes are plausible for 2026 and we
could not disambiguate them without an authenticated browser session (§8).

### 4.3 What must be attached within 7 days of 2026-11-02

- The **solver's code**, open sourced.
- **All artifacts** (weights/parameters per the ARC-AGI-2 Winner License, which requires
  "open source system, open source model, and open source weights/parameters" per the OSI
  Open Source AI checklist — `--page-name rules` line 88).
- Attached to a **Solution Writeup** on the ARC-AGI-2 competition.
- Plus the reproduction link: rules line 94 requires winners to provide "a link to a code repository
  with complete and detailed instructions so that the results obtained can be reproduced."

Deadline: **2026-11-09T23:59Z**.

---

## 5. Can we submit the same content to BOTH the Paper Track and the ARC-AGI-2 Grand Prize?

**VERIFIED FACTS**

- The two are **separate Kaggle competitions** (id 133724 paper track vs id 133469 ARC-AGI-2).
  A Paper Track Writeup is created under `arc-prize-2026-paper-track`; a Solution Writeup is a post
  on `arc-prize-2026-arc-agi-2`. They are different objects in different competitions.
- The Paper Track exists *in order to* document an ARC-AGI-2/3 submission —
  `--page-name Description`: "To participate in the ARC Prize 2026 Paper Track, you need to submit a
  Writeup where you document the submission that you provided for either ARC-AGI-2 or ARC-AGI-3."
- Paper Track rules require team alignment: "Team must match the team making a submission to either
  ARC-AGI-2 or ARC-AGI-3." (`--page-name rules` line 29)
- Both prizes use the **identical 6-dimension rubric**, averaged (Accuracy / Universality /
  Progress / Theory / Completeness / Novelty, each 0–5).
- On unifying ARC-AGI-2 and ARC-AGI-3 into one paper, the host said (paper-track topic 694752,
  Greg Kamradt, 2026-04-27):
  > A unified paper is allowed - it will be judged according to the rubric as normal. However, if
  > you think your paper could more clearly communicate it's finding and impact using just one
  > ARC-AGI version that is fine too.
  > Yes, please use 1 notebook as your primary one when submitting the paper but also link to any
  > other assets you'd like to as well

**NOT FOUND:** any rule, on Kaggle or arcprize.org, prohibiting the same content from serving both.
We read the full paper-track rules page (36,239 chars, 211 lines) and ARC Prize 2026's general rules
and found no anti-double-dipping clause.

**INFERRED (not verified):** submitting the same paper as both (a) the Paper Track Writeup and
(b) the ARC-AGI-2 Solution Writeup appears permitted, because they are submissions to two different
competitions with two different prizes, and the Paper Track is explicitly *for* documenting the
ARC-AGI-2 entry. Kaggle's general tie/duplication language does not address cross-competition reuse.

Practical consequence: doing both is nearly free once the content exists, and it roughly doubles
the number of prize pools we are eligible for (Paper: $75k + share of $375k; ARC-AGI-2 Innovation:
$275k). The cost is that the Paper Track Writeup must be ≤1,500 words in the Kaggle editor while the
Solution Writeup has no stated word limit — write once, then trim for the Paper Track.

---

## 6. arXiv preprints and prior publication

**VERIFIED: no rule forbids it.** The full paper-track rules page (211 lines, 36,239 chars) contains
**zero** occurrences of `arXiv`, `preprint`, or `prior publication`. ARC Prize 2026's general rules
(`https://arcprize.org/competitions/2026`) contain none either. The only relevant clauses are
pro-openness:

- `https://arcprize.org/competitions/2026`: "In order for a submission to be eligible, all code and
  methods authored by the submitter must be made open source under a permissive public domain
  license (eg. CC0 or MIT-0)."
- `--page-name rules` line 19: "In line with the spirit of the competition, participants eligible
  for a prize will be removed from the competition if they do not open source their solutions."

**VERIFIED strong precedent.** The 2025 1st-place Paper Prize winner was posted publicly on arXiv
**before** the competition deadlines and still won $50k:

- <https://arxiv.org/abs/2510.04871> — "Less is More: Recursive Reasoning with Tiny Networks",
  **`[v1] Mon, 6 Oct 2025 14:58:08 UTC`**.
- ARC Prize 2025 submission deadline: 2025-11-03. Paper deadline: 2025-11-09.
- The 2025 Paper Award page explicitly required papers to be public: *"must be public (thus, also
  open-sourced)"*.

**CAVEAT — one real, low-probability tension.** Kaggle's General Competition Rules §3.6.b
(`--page-name rules` line 131), which applies to the paper track:

> **Public Code Sharing.** You are permitted to publicly share Competition Code… If you do choose to
> share Competition Code or other such code, **you are required to share it on Kaggle.com on the
> discussion forum or notebooks associated specifically with the Competition** for the benefit of all
> competitors.

Strictly read, publishing the *solver code* only on GitHub (not on Kaggle) is a technical deviation.
ARC Prize simultaneously mandates open sourcing, and every 2024/2025 winner published off-Kaggle,
so this is standard practice rather than a real barrier — but the Writeup's mandatory
**Public Notebook / Project Links** field is a free way to satisfy 3.6.b at the same time.
Posting the *paper* on arXiv is not restricted by 3.6.b, which is about code.

**Bottom line: arXiv preprint during the competition is safe and is the winning precedent.**

---

## 7. Discrepancies and traps (all VERIFIED as existing conflicts)

| # | Conflict | Sources | Recommended action |
|---|---|---|---|
| 1 | **Paper deadline.** Kaggle `Timeline` + API say `2026-11-09T23:59Z`; arcprize.org Key Dates say "**November 8, 2026** - Papers due". Forum topic 735361 asks which is binding — unanswered. | `--page-name Timeline`; `https://arcprize.org/competitions/2026` | **Target Nov 8.** Submit with a full day of slack. |
| 2 | **Submission count.** Rules §2: "each Team may submit one (1) Submission only." CLI: "Remaining today: 5." Topic 735361 asserts multiple/day are allowed. | `--page-name rules` L33; `submission-limits` | Treat as **ONE**. Never test-submit. |
| 3 | **Tie-break.** Evaluation page: "the Paper that was entered first … will be the winner." General rules §3.7.b: "For Hackathon Competitions … there will be no tiebreakers." | `--page-name Evaluation`; rules L136 | Unknown. Mild argument for submitting a complete version early. |
| 4 | **Saved ≠ submitted.** "Any un-submitted or draft Writeups by the competition deadline will not be considered by the Judges." | Submission Requirements page | Click **Submit**, then verify with `kaggle competitions submissions`. |
| 5 | **Missing submission = zero.** We currently have 0 lifetime submissions on the paper track. | `submission-limits` | Must confirm a real, listed submission. |
| 6 | **`projects` URL is the only stated entry point.** We could not render it (JS-only). | Submission Requirements page | Human must click it in a logged-in browser. |

---

## 8. What a human must check by logging into Kaggle in a browser

These are the only remaining unknowns; all require an authenticated browser (Kaggle HTML is
JS-rendered and returns a 5,614-byte shell to any fetcher, verified).

1. Open <https://www.kaggle.com/competitions/arc-prize-2026-paper-track/projects> and confirm the
   **"New Writeup"** button exists and what fields it asks for.
2. Determine whether **"Public Project Link"** accepts an uploaded **PDF file** or only a URL.
3. Find **where the ARC-AGI-2 submission ID is entered** ("you will be asked to provide your
   submission ID").
4. Confirm the **Submit** button appears top-right after saving, and that pressing it does not
   immediately consume the single allowed submission in a way that blocks later edits.
5. On <https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2>, find the **Solution Writeup**
   route: is it a **Writeups** tab, or the **team page** "solution write-up URL" field?
6. Check whether the Writeup editor **enforces** the 1,500-word cap or merely warns.
7. Ask in the paper-track forum: does a bibliography count toward 1,500 words, and does the cap
   apply to a PDF in Project Links? (Both are open and unanswered since 2026-05-02.)

---

## 9. Reproduce this research

```powershell
kaggle competitions list --search "arc-prize"

kaggle competitions pages -c arc-prize-2026-paper-track
kaggle competitions pages -c arc-prize-2026-paper-track --content --page-name "Submission Requirements"
kaggle competitions pages -c arc-prize-2026-paper-track --content --page-name Timeline
kaggle competitions pages -c arc-prize-2026-paper-track --content --page-name Evaluation
kaggle competitions pages -c arc-prize-2026-paper-track --content --page-name rules

# historical mechanism (page name has a TRAILING SPACE — filter, don't --page-name)
kaggle competitions pages -c arc-prize-2025 --content        # -> page "Paper Award "
kaggle competitions pages -c arc-prize-2024 --content        # -> page "Paper Award "

# ARC-AGI-2 Grand/Innovation Prize writeup requirement
kaggle competitions pages -c arc-prize-2026-arc-agi-2 --content --page-name Prizes
kaggle competitions pages -c arc-prize-2026-arc-agi-2 --content --page-name rules

# forum evidence
kaggle competitions topics list -c arc-prize-2026-paper-track
kaggle competitions topics show arc-prize-2026-paper-track 694752   # unified paper allowed
kaggle competitions topics show arc-prize-2026-paper-track 696513   # 1500-word / bibliography (unanswered)
kaggle competitions topics show arc-prize-2026-paper-track 735361   # 1 sub/day? deadline? (unanswered)
kaggle forums topics show product-announcements 582328              # what a Hackathon Writeup is
kaggle forums topics show product-feedback 373153                   # what a Solution Write-Up is

# read-only state checks (SAFE — never run 'kaggle competitions submit' here)
kaggle competitions files             -c arc-prize-2026-paper-track -v
kaggle competitions submissions       -c arc-prize-2026-paper-track
kaggle competitions submission-limits -c arc-prize-2026-paper-track
```

Web sources fetched: <https://arcprize.org/competitions/2026>,
<https://arcprize.org/competitions/2026/paper>, <https://arcprize.org/competitions/2026/arc-agi-2>,
<https://arcprize.org/competitions/2026/arc-agi-3>, <https://arcprize.org/competitions/2025>,
<https://arcprize.org/competitions/2025/archive>, <https://arcprize.org/competitions/2024>,
<https://arcprize.org/blog/arc-prize-2025-results-analysis>, <https://arcprize.org/blog>,
<https://arxiv.org/abs/2510.04871>.

> Note: `r.jina.ai` was unreachable from this machine (connection refused) and the `web_search`
> backend returned HTTP 402 (insufficient balance) partway through, so all Kaggle content above was
> obtained through the **authenticated Kaggle CLI**, not a text proxy. Nothing in this document is
> sourced from a rendering proxy.

---

## 10. Ordered checklist to submit our paper

**Phase A — prerequisites (before writing)**

- [ ] **A1.** Confirm ARC-AGI-2 has at least one valid scored submission on the leaderboard
      (`kaggle competitions submissions -c arc-prize-2026-arc-agi-2`; we currently have
      `Lifetime submissions: 2`). The paper is ineligible without a corresponding entry, and that
      submission ID must be quoted in the Writeup.
- [ ] **A2.** Record the ARC-AGI-2 **submission ID** we will point the paper at.
- [ ] **A3.** Decide the target: ARC-AGI-2 (we are entered). A unified ARC-AGI-2 + ARC-AGI-3 paper is
      explicitly allowed, but ARC-AGI-3 is only for entrants of that track.

**Phase B — required artifacts**

- [ ] **B1.** Prepare a **public Kaggle Notebook** that reproduces the approach. Push it with
      `python tools/kpush.py` (or upload directly). It may be private; it auto-publishes after the
      deadline, but public is safer for the "must be public" spirit of the rules.
- [ ] **B2.** Produce a **cover image** (required) plus any result figures for the Media Gallery.
- [ ] **B3.** Export the paper as a **PDF** for the optional Public Project Link.
- [ ] **B4.** Write the paper body against the six required sections — Abstract, Intro, Prior work,
      Approach, Results, Conclusion — and against arcprize.org's explicit warning: *"Shorter and
      clearer is always better. No filler, no unnecessary equations."*
- [ ] **B5.** **Trim to ≤1,500 words** on the Writeup body (excluding title/subtitle), keeping a
      bibliography short and assuming it may count against the cap.

**Phase C — create the Writeup**

- [ ] **C1.** Open <https://www.kaggle.com/competitions/arc-prize-2026-paper-track/projects> while
      logged in as our account (which has `userHasEntered = True`).
- [ ] **C2.** Click **"New Writeup"**.
- [ ] **C3.** Set **title** and **subtitle**.
- [ ] **C4.** **Select the Track** — this is mandatory: choose **ARC-AGI-2**.
- [ ] **C5.** Paste the ≤1,500-word body into the Writeup editor.
- [ ] **C6.** Upload the **cover image** and any figures to the **Media Gallery**.
- [ ] **C7.** Add the **public notebook** to the **Project Links** field.
- [ ] **C8.** Add the **PDF** via the optional **Public Project Link**.
- [ ] **C9.** Enter the **ARC-AGI-2 submission ID** where requested.

**Phase D — submit and verify (the step that actually counts)**

- [ ] **D1.** Save the Writeup.
- [ ] **D2.** Click the **"Submit"** button in the top-right of the Writeup. Saving alone is **not**
      a submission.
- [ ] **D3.** Verify from the CLI:
      `kaggle competitions submissions -c arc-prize-2026-paper-track` must **no longer** say
      "No submissions found".
- [ ] **D4.** Verify `kaggle competitions submission-limits -c arc-prize-2026-paper-track` shows
      `Lifetime submissions: 1`.
- [ ] **D5.** Re-open the Writeup and confirm it shows as **submitted, not draft**.

**Phase E — deadline discipline**

- [ ] **E1.** Complete D1–D5 by **2026-11-08** (arcprize.org's stated date), not 2026-11-09,
      because the two sources disagree.
- [ ] **E2.** Only edit the Writeup after submitting if a human confirms editing preserves the
      submission (forum topic 735361, unanswered). Otherwise freeze it.
- [ ] **E3.** If pursuing the ARC-AGI-2 **Innovation/Grand Prize** as well, attach all artifacts to
      the ARC-AGI-2 **Solution Writeup** by **2026-11-09T23:59Z** (7 days after 2026-11-02).
- [ ] **E4.** Make sure the solver code is open-sourced under CC-BY-4.0-compatible terms (the
      winner license for both competitions) and reachable without login.
