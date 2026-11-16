# Emotion-S1

Emotion-S1 is an emotion-recognition decision model. The encoder is ModernBERT-base (about 150 million parameters, `answerdotai/ModernBERT-base`) with three heads:

- **choice** — which emotion, from the options passed in with the question
- **score** — a 1–5 intensity
- **yes/no** (called `noul` in the code) — a yes/no question such as "Is the user frustrated?" or "Does the user need support?"

It returns a calibrated probability for that question. The comparison in this repo runs the same questions through Emotion-S1, Claude (`claude-sonnet-5-5`), GPT (`gpt-6.1-sol`) and Jev (`jev-latest`, served as `jev-1.14.0`). Claude and GPT were called with reasoning effort `low`.

The model was trained in Colab. The weights are not in this repository. See [Weights](#weights).

## Repository layout

```
comparison/                  comparison scripts and saved runs
  run_emotion_s1.py          Emotion-S1 inference on CPU
  run_llm.py                 Claude and GPT (dry-run unless --live)
  run_jev.py                 Jev on the TypeSafe API (dry-run unless --live)
  score.py                   accuracy, calibration, latency and cost
  make_questions.py          600 held-out questions from the training datasets
  measure_cpu.py             exclusive CPU latency bench
  plot_results.py            accuracy and latency charts
  common.py                  shared prompts and parsing
  results/                   held-out test: splits of the training datasets
  results_ood/               unseen data: MELD, EmoWOZ and CARER
  ood/                       scripts and raw answers for the unseen-data run
    data/meld_test_sent_emo.csv
training/                    training code, config and dataset notes
  emotion_s1_core.py         encoder, three heads and ask()
  config_full.py             full-run settings
  datasets.md                sources and licences
  PLAN.md                    design notes
  build_notebook.py          rebuilds the Colab notebook
notebook/
  emotion_s1_t4.ipynb        Colab notebook, with saved outputs
```

`comparison/results/` is the held-out test (questions drawn from the training datasets' held-out splits). `comparison/results_ood/` is the unseen-data run (MELD, EmoWOZ, CARER). Both folders hold the written report, raw answers, logs and charts. Copies of the raw JSON also sit next to the scripts that wrote them (`comparison/*.json` for the held-out run, `comparison/ood/*.json` for the unseen run).

Charts:

- `comparison/results/accuracy_by_type.png`, `comparison/results/latency.png`
- `comparison/results_ood/accuracy_by_type.png`, `comparison/results_ood/latency.png`, `comparison/results_ood/id_vs_ood.png`

## Headline results

The tables in this section are copied from `comparison/results/results.md` and `comparison/results_ood/results.md`. The figures are the ones in those files.

### Held-out test (training datasets)

600 questions (200 choice, 200 yes/no, 200 score), stratified across the training datasets, plus 100 second-pass repeats. Run on 16 November 2026. All 2,800 calls parsed.

| System | Choice acc | Yes/No acc | Score MAE (within 1) | ECE (c / y-n / s) | Median ms | p95 ms | $ per 1,000 turns | Self-disagreement | Parse-fail |
|---|---|---|---|---|---|---|---|---|---|
| Emotion-S1 (CPU) | 0.7 | 0.795 | 0.4122 (0.89) | 0.078 / 0.0772 / 0.0563 | 40.5 | 184.0 | 0.0 | 0.0 | 0.0/0.0/0.0 |
| anthropic:claude-sonnet-5-5 | 0.625 | 0.77 | 1.0299 (0.65) | 0.0831 / 0.1274 / 0.3206 | 877.6 | 1536.0 | 2.1466 | 0.02 | 0.0/0.0/0.0 |
| openai:gpt-6.1-sol | 0.64 | 0.775 | 1.0374 (0.655) | 0.2136 / 0.1788 / 0.4869 | 2133.1 | 4163.9 | 1.5718 | 0.03 | 0.0/0.0/0.0 |
| typesafe:jev-latest | 0.665 | 0.76 | 0.9928 (0.59) | 0.1661 / 0.1117 / 0.3604 | 119.0 | 159.3 | 0.0455 | 0.29* | 0.0/0.0/0.0 |

\*Jev's 0.29 exact-match disagreement is almost all soft score wobble of 0.01–0.07 on the 1–5 scale (max 0.07; none above 0.10). Categorical flips were 2/34 choice and 1/33 yes/no. If scores within 0.10 count as the same answer, Jev's disagreement rate is **0.03**, in line with Claude and GPT. Claude flipped 1 choice and 1 score (by a whole point). GPT flipped 2 choice and 1 score (by a whole point). Emotion-S1 is deterministic.

Emotion-S1 leads this set on choice, yes/no and score, and it has the lowest calibration error. These questions come from the held-out splits of the datasets it was trained on.

### Unseen data (MELD, EmoWOZ, CARER)

The same four systems, on official test splits of datasets that were not in training. 600 questions (300 choice, 300 yes/no) plus 100 repeats. Score is omitted: none of the three has an official emotion-intensity label.

| System | Choice acc | Yes/No acc | Score MAE (within 1) | ECE (c / y-n / s) | Median ms | p95 ms | $ per 1,000 turns* | Self-disagreement | Parse-fail |
|---|---|---|---|---|---|---|---|---|---|
| Emotion-S1 (CPU) | 0.3433 | 0.75 | — | 0.3147 / 0.1131 / — | 95.6 | 464.1 | 0.0 | 0.0 | 0.0/0.0/— |
| anthropic:claude-sonnet-5-5 | 0.4733 | 0.7533 | — | 0.1668 / 0.0767 / — | 930.0 | 2079.8 | 1.8426 | 0.02 | 0.0/0.0/— |
| openai:gpt-6.1-sol | 0.4567 | 0.7933 | — | 0.4024 / 0.1526 / — | 2262.9 | 8579.6 | 1.1076 | 0.04 | 0.0/0.0/— |
| typesafe:jev-latest | 0.4467 | 0.76 | — | 0.3439 / 0.0744 / — | 114.8 | 160.7 | 0.0302 | 0.04 | 0.0/0.0/— |

\*A "turn" in the in-domain table was three questions (choice + yes/no + score). Here there is no score, so this column is the cost of one choice + one yes/no question, and is **not** directly comparable to the in-domain dollar figures. Emotion-S1 latency was measured with 4 torch threads (an environment leftover); the earlier exclusive 8-thread in-domain bench is the better CPU speed figure.

By dataset (choice / yes-no). Bold marks the best score for that dataset and question type, as in the source report.

| Dataset | Random choice | Emotion-S1 | Claude | GPT | Jev |
|---|---|---|---|---|---|
| MELD (Friends TV) | 14.3% | 29% / 75% | 53% / 72% | **56%** / **84%** | 48% / 73% |
| EmoWOZ (task-oriented) | 14.3% | 29% / 74% | **45%** / **84%** | 40% / 83% | **45%** / **84%** |
| CARER (tweets) | 16.7% | **45%** / **76%** | 44% / 70% | 41% / 71% | 41% / 71% |

In-domain versus out-of-domain:

| System | Choice in-domain → OOD | Yes/No in-domain → OOD |
|---|---|---|
| Emotion-S1 | 70% → 34% (−36 pp) | 79.5% → 75% (−4.5 pp) |
| Claude Sonnet 5.5 | 62.5% → 47% (−15 pp) | 77% → 75% (−2 pp) |
| GPT-6.1 Sol | 64% → 46% (−18 pp) | 77.5% → 79% (+1.5 pp) |
| Jev 1.14.0 | 66.5% → 45% (−22 pp) | 76% → 76% (0 pp) |

Emotion-S1 wins on held-out data from its training datasets. On unseen datasets it trails the LLMs on fine-grained choice (0.3433, against 0.4733 for Claude and 0.4567 for GPT; Jev is 0.4467) and it is about level on yes/no (0.75, against 0.7533, 0.7933 and 0.76). The choice head drops from 70% to 34%. Yes/no moves from 79.5% to 75%. On CARER tweets it still leads both columns (45% / 76%). The choice gap is on MELD and EmoWOZ.

## Latency and cost

On CPU, one question is about 45 ms, and a call costs $0.

The exclusive bench below is copied from `comparison/results/results.md` (8-core Intel Xeon, 8 torch threads, fp32, after warm-up):

| | Median | p95 | n |
|---|---|---|---|
| One yes/no question (fp32) | 44.7 ms | 57.1 ms | 40 |
| All four questions, one turn (fp32) | 310.6 ms | 387.3 ms | 40 |
| One yes/no question (int8) | 28.0 ms | 34.2 ms | 40 |
| All four questions, one turn (int8) | 163.8 ms | 211.2 ms | 40 |

Across the 600 mixed held-out questions the median was 40.5 ms and p95 was 184.0 ms. Choice questions are slower because each option is a separate encoder row. The held-out cost column for Emotion-S1 is $0 per 1,000 turns (three questions: choice, score and yes/no). The same column is $1.5718 for GPT, $2.1466 for Claude and $0.0455 for Jev.

The 95.6 ms median in the unseen-data table used 4 torch threads. The 8-thread bench above is the CPU figure for this model (44.7 ms median for one yes/no question).

## How to run

Keys are read from the environment at run time. Set them in your shell. Do not commit them, and do not paste them into the repo:

- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
- `TYPESAFE_API_KEY`

`run_llm.py` and `run_jev.py` make no paid calls unless you pass `--live`.

Held-out comparison, from `comparison/` (also see `comparison/README.md`). `W` is a local copy of the weights folder, which is not in this repo:

```bash
cd comparison
export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
export TYPESAFE_API_KEY=...

W=/path/to/emotion_s1_model
python make_questions.py "$W"
python run_emotion_s1.py "$W"
python measure_cpu.py "$W"

python run_llm.py --provider openai    --model gpt-6.1-sol
python run_llm.py --provider anthropic --model claude-sonnet-5-5
python run_llm.py --provider openai    --model gpt-6.1-sol --live --repeat 100
python run_llm.py --provider anthropic --model claude-sonnet-5-5 --live --repeat 100
python run_jev.py --live --repeat 100

python score.py \
  --price "llm_openai_gpt-6.1-sol_raw.json=2.00,10.00" \
  --price "llm_anthropic_claude-sonnet-5-5_raw.json=2.00,10.00" \
  --price "jev_raw.json=0.042,0" \
  --price-source "llm_openai_gpt-6.1-sol_raw.json=https://developers.openai.com/api/docs/pricing" \
  --price-source "llm_anthropic_claude-sonnet-5-5_raw.json=https://platform.claude.com/docs/en/about-claude/pricing" \
  --price-source "jev_raw.json=https://docs.typesafe.ai/models"
python plot_results.py
```

Those model ids and list prices are the ones used on 16 November 2026. `score.py` does not hard-code prices. Copy input and output USD per 1M tokens from each provider's pricing page on the day you run, and pass them with `--price`.

Unseen-data scripts live in `comparison/ood/`. They read `questions.json` from the working directory. MELD text is the CSV already in `comparison/ood/data/`. EmoWOZ defaults to `--emowoz-dir /tmp/ood-data`. CARER (`dair-ai/emotion`) is loaded from Hugging Face.

```bash
cd comparison/ood
python make_ood_questions.py
python run_emotion_s1.py "$W"
python run_llm.py --provider openai    --model gpt-6.1-sol --live --repeat 100
python run_llm.py --provider anthropic --model claude-sonnet-5-5 --live --repeat 100
python run_jev.py --live --repeat 100
python score.py --out-dir ../results_ood \
  --price "llm_openai_gpt-6.1-sol_raw.json=2.00,10.00" \
  --price "llm_anthropic_claude-sonnet-5-5_raw.json=2.00,10.00" \
  --price "jev_raw.json=0.042,0"
```

Training is the Colab notebook: [emotion_s1_t4.ipynb](https://colab.research.google.com/drive/1FjluwT6thpshMoTF-NON1fLEVa8UFCkx). A copy with saved outputs is at `notebook/emotion_s1_t4.ipynb`. In Colab, choose a T4 GPU and run all cells. `QUICK = True` is a short test; `QUICK = False` is the full run, stopped at 45 minutes (`training/config_full.py`). The notebook writes `/content/emotion_s1` and copies the folder to Google Drive at `My Drive/emotion_s1_model`.

`training/smoke_test.py` looks for `emotion_s1_t4.ipynb` beside itself. In this repo the notebook is under `notebook/`.

Once the weights folder is on disk, inference is the `ask()` function in `training/emotion_s1_core.py` (the same file is saved next to the weights):

```python
import sys
sys.path.insert(0, "emotion_s1_model")
from emotion_s1_core import load_emotion_s1, ask
m = load_emotion_s1("emotion_s1_model", device="cpu")
ask(m, "It's STILL broken after three resets.", {
  "emotion":    {"type": "choice", "question": "Which emotion is the user showing?", "options": ["joy", "anger", "sadness", "fear", "neutral"]},
  "intensity":  {"type": "score",  "question": "How intense is the user's emotion? 1 = calm, 5 = extremely intense.", "scale": [1, 5]},
  "frustrated": {"type": "noul",   "question": "Is the user frustrated?"}},
  context=[("Assistant", "Could you try restarting the router?")])
```

## Weights

The trained weights are not in this repository. The fp32 state dict is 596.1 MB (the optional int8 build in the held-out report is 265.3 MB). Training was done in Colab, and the notebook keeps the folder in Google Drive at `My Drive/emotion_s1_model`. The weights are available on request.

## Datasets and licences

From `training/datasets.md`. The notebook downloads the training sets at run time. This repo does not redistribute them.

| # | Dataset | Used for | Source loaded by the notebook | Size | Licence | Commercial use |
|---|---|---|---|---|---|---|
| 1 | **GoEmotions** (Google, Reddit) | choice, yes/no, neutral score anchors | HF `google-research-datasets/go_emotions`, config `simplified` | 43,410 / 5,426 / 5,427 comments; 27 emotions + neutral | **Apache-2.0** | ✅ Yes |
| 2 | **EmoBank** (JULIE Lab) | score (valence, arousal) | Official CSV: `github.com/JULIELab/EmoBank/corpus/emobank.csv` | 8,062 / 1,000 / 1,000 sentences; V/A/D on 1–5 | **CC BY-SA 4.0** | ✅ Yes, with attribution and share-alike on the data |
| 3 | **DailyDialog** | choice and yes/no per turn, with context | HF `li2017dailydialog/daily_dialog`, `revision="refs/convert/parquet"` | 11,118 / 1,000 / 1,000 dialogues → 23,773 / 1,805 / 2,177 turns after keeping about 1 in 8 "no emotion" turns | **CC BY-NC-SA 4.0** | ❌ Non-commercial |
| 4 | **EmpatheticDialogues** (Meta) | choice (32 emotions), yes/no, support | HF `facebook/empathetic_dialogues`, `revision="refs/convert/parquet"` | 76,673 / 12,030 / 10,943 utterances → 35,612 / 5,520 / 5,080 early speaker turns | **CC BY-NC 4.0** | ❌ Non-commercial |
| 5 | **ESConv** (emotional-support chats) | distress score 1–5, support, emotion type | HF `thu-coai/esconv` | 910 / 195 / 195 conversations → 4,550 / 975 / 975 seeker turns | **CC BY-NC 4.0** | ❌ Non-commercial |
| 6 | **SemEval-2018 Task 1, EI-reg** (tweets) | per-emotion intensity score (anger, fear, joy, sadness) | Official zip from saifmohammad.com (`SemEval2018-Task1-all-data.zip`); falls back to the unofficial HF mirror `Holmeister/ei_reg_*` | 7,102 / 1,464 / 4,068 tweets | **Research only; do not redistribute** (NRC terms of use) | ❌ Research only |

**Slide-ready takeaways**
* **2 of the 6 datasets are commercially usable:** GoEmotions and EmoBank.
* **3 are non-commercial (NC):** DailyDialog, EmpatheticDialogues and ESConv.
* **1 is research-only:** SemEval-2018 EI-reg.
* A model trained on NC or research-only data inherits those restrictions, so this full model is fine for a **tutorial or demo, not a product**.
* `COMMERCIAL_SAFE = True` in the notebook trains on GoEmotions and EmoBank only. That's the starting point for a client. Even better, fine-tune on the client's own labelled chat logs.
* We don't redistribute any data. The notebook downloads everything from the original sources at run time.

**Rejected:**
* `dair-ai/emotion` (educational and research use only, and it adds little).
* `stepp1/tweet_emotion_intensity` (unclear provenance).
* MELD (GPL-3.0, raw tar archives only).
* `SemEvalWorkshop/sem_eval_2018_task_1` (its parquet copy has only the E-c subtask).
* `reallycarlaost/emobank*` (unofficial copies with changed labels).

The unseen-data sets are separate from those six. Their licence notes, copied from `comparison/results_ood/results.md`:

| Dataset | Kind of text | Official test used | Licence | Source |
|---|---|---|---|---|
| **MELD** (text annotations) | Multi-party scripted TV dialogue from *Friends* | 2,610 utterances / 280 dialogues (`test_sent_emo.csv`) | Repository is **GPL-3.0**. The transcripts are from copyrighted *Friends* episodes; used here for research/evaluation only, text only (no audio or video). | [GitHub CSV](https://github.com/declare-lab/MELD/blob/master/data/MELD/test_sent_emo.csv), [paper](https://arxiv.org/pdf/1810.02508.pdf) |
| **EmoWOZ** | Task-oriented user–system dialogue (MultiWOZ + DialMAGE human–machine) | Official test IDs: 1,000 MultiWOZ + 100 DialMAGE dialogues, 8,620 labelled user turns after dropping empty text | **CC BY-NC 4.0** (non-commercial, attribution required). Authors also say research use only. | [Zenodo 6506504](https://zenodo.org/records/6506504), [paper](https://aclanthology.org/2022.lrec-1.436/) |
| **CARER** (`dair-ai/emotion`) | Single-turn English Twitter messages (not multi-turn chat) | Official `split` test set, 2,000 tweets | **Educational and research purposes only** (dataset card / GitHub). | [Hugging Face](https://huggingface.co/datasets/dair-ai/emotion), [paper](https://aclanthology.org/D18-1404/) |

`comparison/ood/data/meld_test_sent_emo.csv` is the MELD test transcript used for that run. The report treats it as research and evaluation only.

## Colab

https://colab.research.google.com/drive/1FjluwT6thpshMoTF-NON1fLEVa8UFCkx
