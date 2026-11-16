# Emotion-S1 vs Claude vs GPT vs Jev — out-of-domain

Comparison run on 16 November 2026 (Africa/Lagos, UTC+1). This is the companion to the in-domain run: the same four systems, the same models and settings, on **kinds of chat Emotion-S1 never trained on**.

All four systems answered the **same 600 held-out questions** (300 choice, 300 yes/no; 100 of each type from each dataset), plus **100 second-pass repeats** (50 choice, 50 yes/no, stratified). Every one of the 2,800 calls succeeded and parsed (0 failures, 0 skipped). No training was done on this machine.

**Score questions are omitted.** None of the three datasets has an official emotion-intensity label, and no proxy was invented.

## Datasets, licences and mappings

Official **test splits only**. None of these six training sources were used: GoEmotions, DailyDialog, EmpatheticDialogues, ESConv, SemEval-2018 EI-reg, EmoBank.

| Dataset | Kind of text | Official test used | Licence | Source |
|---|---|---|---|---|
| **MELD** (text annotations) | Multi-party scripted TV dialogue from *Friends* | 2,610 utterances / 280 dialogues (`test_sent_emo.csv`) | Repository is **GPL-3.0**. The transcripts are from copyrighted *Friends* episodes; used here for research/evaluation only, text only (no audio or video). | [GitHub CSV](https://github.com/declare-lab/MELD/blob/master/data/MELD/test_sent_emo.csv), [paper](https://arxiv.org/pdf/1810.02508.pdf) |
| **EmoWOZ** | Task-oriented user–system dialogue (MultiWOZ + DialMAGE human–machine) | Official test IDs: 1,000 MultiWOZ + 100 DialMAGE dialogues, 8,620 labelled user turns after dropping empty text | **CC BY-NC 4.0** (non-commercial, attribution required). Authors also say research use only. | [Zenodo 6506504](https://zenodo.org/records/6506504), [paper](https://aclanthology.org/2022.lrec-1.436/) |
| **CARER** (`dair-ai/emotion`) | Single-turn English Twitter messages (not multi-turn chat) | Official `split` test set, 2,000 tweets | **Educational and research purposes only** (dataset card / GitHub). | [Hugging Face](https://huggingface.co/datasets/dair-ai/emotion), [paper](https://aclanthology.org/D18-1404/) |

### Label mappings

Choice gold is always the dataset's official label, **not** remapped onto Emotion-S1's training inventory. The choice head is option-conditioned, so each question passes that dataset's own option list.

| Dataset | Choice options (official) | Yes/No mappings |
|---|---|---|
| MELD | anger, disgust, fear, joy, neutral, sadness, surprise (MELD's *joy*, not DailyDialog's *happiness*) | **emotion:** yes iff the official Emotion equals the named option. **frustrated** (derived): anger or disgust. **support** (derived): sadness or fear. **positive:** joy. **neutral:** neutral. |
| EmoWOZ | abusive, apologetic, dissatisfied, excited, fearful, neutral, satisfied. Integers 0–6 follow the paper: 1 *fearful* also covers sad/disappointed; 2 *dissatisfied* also covers disliking; 5 *excited* also covers happy/anticipating; 6 *satisfied* also covers liking. | **emotion:** yes iff the official token equals the named option. **frustrated** (derived): dissatisfied or abusive (negative emotion elicited by the system). **support** (derived): fearful only (event-elicited sadness/fear). Dissatisfied is not treated as needing support. **positive:** satisfied or excited. **neutral:** neutral. |
| CARER | anger, fear, joy, love, sadness, surprise (no neutral class) | **emotion:** yes iff the official label equals the named option. **frustrated** (derived): anger. **support** (derived): sadness or fear. **positive:** joy or love. |

Derived yes/no families use the same idea as the in-domain "frustrated" and "needs support" labels, which were also derived. Yes/no items were sampled to be close to 50/50 (overall yes rate 48.3%), so a majority baseline is about 51–53%. EmoWOZ *fearful* and *abusive* are rare in the official test set (18 and 17 turns); the choice sample includes almost all of them, and the support yes/no slice for EmoWOZ has only four genuine positives.

Context: MELD uses up to four earlier utterances with speaker names; EmoWOZ uses up to four earlier User/Assistant turns; CARER tweets have no prior turns. The state format matches `emotion_s1_core.format_state`.

## Model IDs and settings

Same as the in-domain run: Emotion-S1 local fp32 CPU; `claude-sonnet-5-5` with `output_config.effort=low`; `gpt-6.1-sol` with `reasoning_effort=low`; `jev-latest` served as `jev-1.14.0`. Temperature omitted (both mid-tier LLMs reject it).

## Headline results (out-of-domain)

| System | Choice acc | Yes/No acc | Score MAE (within 1) | ECE (c / y-n / s) | Median ms | p95 ms | $ per 1,000 turns* | Self-disagreement | Parse-fail |
|---|---|---|---|---|---|---|---|---|---|
| Emotion-S1 (CPU) | 0.3433 | 0.75 | — | 0.3147 / 0.1131 / — | 95.6 | 464.1 | 0.0 | 0.0 | 0.0/0.0/— |
| anthropic:claude-sonnet-5-5 | 0.4733 | 0.7533 | — | 0.1668 / 0.0767 / — | 930.0 | 2079.8 | 1.8426 | 0.02 | 0.0/0.0/— |
| openai:gpt-6.1-sol | 0.4567 | 0.7933 | — | 0.4024 / 0.1526 / — | 2262.9 | 8579.6 | 1.1076 | 0.04 | 0.0/0.0/— |
| typesafe:jev-latest | 0.4467 | 0.76 | — | 0.3439 / 0.0744 / — | 114.8 | 160.7 | 0.0302 | 0.04 | 0.0/0.0/— |

\*A "turn" in the in-domain table was three questions (choice + yes/no + score). Here there is no score, so this column is the cost of one choice + one yes/no question, and is **not** directly comparable to the in-domain dollar figures. Emotion-S1 latency was measured with 4 torch threads (an environment leftover); the earlier exclusive 8-thread in-domain bench is the better CPU speed figure.

Claude is the best calibrated on this set. Emotion-S1's choice ECE rises to 0.31, which is expected: it is still confident when the label inventory or style no longer matches training.

### By dataset (choice / yes-no)

| Dataset | Random choice | Emotion-S1 | Claude | GPT | Jev |
|---|---|---|---|---|---|
| MELD (Friends TV) | 14.3% | 29% / 75% | 53% / 72% | **56%** / **84%** | 48% / 73% |
| EmoWOZ (task-oriented) | 14.3% | 29% / 74% | **45%** / **84%** | 40% / 83% | **45%** / **84%** |
| CARER (tweets) | 16.7% | **45%** / **76%** | 44% / 70% | 41% / 71% | 41% / 71% |

Emotion-S1 stays well above random on every choice slice. On CARER, it slightly leads all three hosted systems on both question types. On the two dialogue sets, Claude, GPT and Jev pull ahead on fine-grained choice.

## In-domain vs out-of-domain

In-domain numbers are from the earlier 600-question run on held-out messages from the six training datasets (those exact messages were not used in training either).

| System | Choice in-domain → OOD | Yes/No in-domain → OOD |
|---|---|---|
| Emotion-S1 | 70% → 34% (−36 pp) | 79.5% → 75% (−4.5 pp) |
| Claude Sonnet 5.5 | 62.5% → 47% (−15 pp) | 77% → 75% (−2 pp) |
| GPT-6.1 Sol | 64% → 46% (−18 pp) | 77.5% → 79% (+1.5 pp) |
| Jev 1.14.0 | 66.5% → 45% (−22 pp) | 76% → 76% (0 pp) |

The specialist's advantage is in-domain. On unseen kinds of chat, the hosted systems take the lead on "which emotion?", while yes/no questions an app would actually act on stay close for everyone, and Emotion-S1 even leads on tweets. That is the pair of results to put on the slides: a 26-minute specialist can replace an LLM judge on its own data, but it is not a general emotion model, and the yes/no heads transfer better than the fine-grained choice head.

## Prices

Same list prices as the in-domain run, fetched 16 November 2026.

| System | Input / 1M | Output / 1M | Source | Mean tokens / question (in, out) | $ / 1,000 two-question turns |
|---|---|---|---|---|---|
| Emotion-S1 | — | — | local | 0, 0 | 0.0000 |
| GPT `gpt-6.1-sol` | $2.00 | $10.00 | https://developers.openai.com/api/docs/pricing | choice 176.1 / 24.3; yes/no 151.2 / 21.0 | 1.1076 |
| Claude `claude-sonnet-5-5` | $2.00 | $10.00 | https://platform.claude.com/docs/en/about-claude/pricing | choice 267.5 / 49.1; yes/no 230.3 / 35.6 | 1.8426 |
| Jev `jev-latest` (`jev-1.14.0`) | $0.042 | $0 | https://docs.typesafe.ai/models | choice 385.1 / 69.8; yes/no 333.3 / 20.0 | 0.0302 |

Total paid usage (600 answers + 100 repeats each): GPT 114,206 input + 15,775 output tokens (~$0.39); Claude 173,687 input + 29,102 output tokens (~$0.64); Jev 251,032 input tokens (~$0.011). Under $1 per provider.

## Caveats

- Out-of-domain is about **genre and labelling scheme**, not about leaked test messages. All items are official test splits Emotion-S1 never trained on.
- CARER is Twitter, not multi-turn chat. It is included as a third unseen genre of user text. MELD and EmoWOZ are the dialogue sets.
- MELD labels come from annotators who also watched the clips; we only give the systems the transcript, so some errors are expected.
- EmoWOZ's inventory (apologetic, abusive, dissatisfied, …) is deliberately different from Ekman-style labels. That is part of the OOD test.
- Derived frustrated/support labels are noisy, as they were in-domain.
- LLM effort was `low`, same as the in-domain run.
- Emotion-S1 OOD latency used 4 CPU threads; do not treat 95.6 ms as a new official 8-core number. The exclusive in-domain bench (44.7 ms for one yes/no, 310.6 ms for four questions) remains the CPU speed figure for the slides.
- GPT's p95 of 8.6 s is from a long tail of slower calls, even at low reasoning effort.

## File paths

All under `/workspace/emotion-s1/compare/results_ood/`:

- `results.md` — this report
- `results.json` — full metrics
- `accuracy_by_type.png` — OOD choice and yes/no
- `latency.png` — median and p95
- `id_vs_ood.png` — combined in-domain vs out-of-domain chart
- `mappings.json` — licences and label mappings
- `questions.json` — the 600-question set (seed 7)
- `s1_raw.json`, `llm_openai_gpt-6.1-sol_raw.json`, `llm_anthropic_claude-sonnet-5-5_raw.json`, `jev_raw.json`
