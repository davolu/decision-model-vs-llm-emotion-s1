# Emotion-S1: a small emotion decision model (plan)

> **Decisions (16 Nov 2026):**
> * This is a tutorial only, so by default we train on all six datasets. The `COMMERCIAL_SAFE` switch (GoEmotions + EmoBank only) is documented in the notebook.
> * The model is named **Emotion-S1**.
> * We compare against Claude, GPT and the real Jev.
> * The licence table is in `datasets.md`.
> * SemEval EI-reg now loads from the **official** zip, with the HF mirror only as a fallback.
> * The notebook is built: `emotion_s1_t4.ipynb`.

**Goal:** a model trained from scratch in the mini-Jev style that answers typed questions about one chat turn, plus 0 to 4 earlier turns as context. It returns calibrated probabilities, never free text, and runs on a CPU in about 40 ms per question.

| Type | Example question | Output |
|---|---|---|
| choice | "Which emotion is the user showing?" with options given per call | a probability for each option |
| score | "How intense is the user's emotion, 1 to 5?" | a distribution over 1..5, plus the expected value |
| noul | "Is the user frustrated?" / "Does the user need support?" / "Is the user showing fear?" | P(yes) |

**Pitch:** calling an LLM on every turn for this costs 0.5–2 s and some money per turn, and it can give different answers to the same input. Emotion-S1 costs about 40 ms on a CPU and nothing per call, and it gives the same answer every time with an honest confidence.

## 1. Datasets (all checked on the box with `datasets` 5.2.0, 16 Nov 2026; see `dataset_check.json`)

| Use | HF id / how to load (no remote code) | Size | Labels | Licence / commercial use |
|---|---|---|---|---|
| choice + noul (core) | `google-research-datasets/go_emotions`, config `simplified` | 43,410 / 5,426 / 5,427 | 27 emotions + neutral, multi-label; official Ekman map → 6 + neutral | **Apache-2.0: commercial OK** |
| turn-level choice in dialogue (with context) | `li2017dailydialog/daily_dialog`, `revision="refs/convert/parquet"` (the main branch is a script and fails) | 11,118 / 1,000 / 1,000 dialogues, about 103k turns | no emotion, anger, disgust, fear, happiness, sadness, surprise. **83% "no emotion"**; fear (146) and disgust (303) are tiny | CC BY-NC-SA 4.0: **non-commercial** |
| fine-grained choice, support | `facebook/empathetic_dialogues`, `revision="refs/convert/parquet"` (alternative: `Estwld/empathetic_dialogues_llm`, parquet, conversation format) | 76,673 / 12,030 / 10,943 utterances, about 25k conversations | 32 emotions (per conversation; label the speaker's first turns) | CC BY-NC 4.0: **non-commercial** |
| intensity + support + emotion type | `thu-coai/esconv` (loads; one JSON string per row in `text`, parse with `json.loads`) | 910 / 195 / 195 conversations, about 13.9k seeker turns in train | `initial_emotion_intensity` and `final_emotion_intensity` 1–5 per conversation; `emotion_type` (anxiety, depression, sadness, anger, fear...); supporter strategy | CC BY-NC 4.0: **non-commercial** |
| intensity (per emotion) | SemEval-2018 Task 1 **EI-reg** English through the unofficial mirrors `Holmeister/ei_reg_{anger,fear,joy,sadness}` (parquet) | about 7.1k train / 1.5k val / 4.1k test tweets | intensity 0–1 for a named emotion | Official terms: **research only, do not redistribute.** The mirror has no licence tag. OK for the tutorial; for a client, download from the official page |
| intensity (arousal) | EmoBank CSV: `load_dataset("csv", data_files="https://raw.githubusercontent.com/JULIELab/EmoBank/master/corpus/emobank.csv")` (on the box it needed a local download because aiohttp is missing; Colab has it). There is no good HF copy (`reallycarlaost/*` are unofficial and relabelled) | 10,062 sentences (8,062 / 1,000 / 1,000 via a `split` column) | V, A, D on a 1–5 SAM scale (observed A 1.8–4.4, V 1.2–4.6) | CC BY-SA 4.0: **commercial OK** (share-alike on the data) |

**Rejected:**
- `dair-ai/emotion` (6 labels, Twitter; licence says "educational and research purposes only", and it adds little beyond GoEmotions).
- `stepp1/tweet_emotion_intensity` (unclear provenance; only low, medium or high).
- `declare-lab/MELD` (GPL-3.0, tar archives only, doesn't load).
- `SemEvalWorkshop/sem_eval_2018_task_1` (its parquet branch has only subtask 5, E-c; EI-reg isn't there).
- `isa-ras/frustration_dataset` (Russian).
- There is **no clean English "frustration" or "needs support" dataset**, so both are derived (§2).

**Commercial note:** a model trained on the full mix inherits the NC and research-only terms. Build a **`COMMERCIAL_SAFE=True` switch** that trains on GoEmotions + EmoBank only. Say this on camera, and tell clients to retrain on their own labelled logs.

## 2. Label → question mapping

The input pair follows mini-Jev. Part A is `[type] question (+ option)`. Part B is the state: `Context:\nUser: …\nAssistant: …\nTarget turn:\nUser: <text>`. Up to 4 earlier turns are taken from DailyDialog, ED and ESConv. GoEmotions, EI-reg and EmoBank rows have no context. 30% of the dialogue rows get their context dropped so the model works with and without it.

**Choice ("Which emotion…?")**
- Question templates: about 8 paraphrases ("Which emotion is the user showing?", "What is the speaker mainly feeling?", …).
- Options: the gold label plus **2–6 random distractors**, shuffled every time. Label sets are mixed: Ekman 6 + neutral, GoEmotions-27, ED-32 and ESConv types. Surface forms vary ("joy" / "happiness" / "happy"). 15% of rows use the full label set.
- Multi-label GoEmotions: use only single-label rows, or never use another gold label as a distractor. Map near-synonyms through a small table (annoyance ≈ anger) so they never appear as rival options.
- DailyDialog "no emotion" is downsampled to about 35% of its rows. "neutral" is included as an option on most rows so the model learns to say "no emotion".

**Noul (Yes/No)**
- *"Is the user showing {emotion}?"*: from any label set. Positives are gold labels. Negatives come 50% from random labels and 50% from **hard negatives** (same polarity, e.g. fear vs sadness). Target balance is 50/50.
- *"Is the user frustrated / annoyed / angry with the situation?"*: Yes for GoEmotions anger, annoyance or disappointment; DailyDialog anger or disgust; ED angry, annoyed, furious or disappointed. No otherwise (including neutral and joy).
- *"Does the user need emotional support?"*: Yes for ESConv seeker turns in conversations with initial intensity ≥ 3; ED negative-emotion first turns (sad, lonely, devastated, afraid, anxious, ashamed…); GoEmotions grief, sadness, fear, nervousness or remorse. No for DailyDialog and GoEmotions turns that are neutral or positive, and for ED positive emotions.
- A few templates for each, plus 10% questions with negation ("Is the user calm?") so the model reads the question instead of memorising "negative = yes".

**Score (1–5)**

| Source | Normalisation to 1–5 | Rubric given |
|---|---|---|
| EI-reg | `1 + 4·x` (x in 0–1) | "How intense is the user's {anger/fear/joy/sadness}? 1 = none, 5 = extreme" |
| EmoBank A | min-max stretch of the observed range 1.8–4.4 → 1–5 (the raw A is compressed around 3) | "How emotionally aroused/intense is the user?" |
| EmoBank V | the same stretch, used as a second rubric | "How positive is the user's mood? 1 = very negative, 5 = very positive" |
| ESConv | `initial_emotion_intensity` for the first 3 seeker turns; `final_emotion_intensity` for the last 2 (110 conversations have no final value, so skip them) | "How intense is the user's distress, 1 to 5?" |
| Neutral anchors | GoEmotions/DailyDialog neutral turns → 1 (a small share, ≤10% of score rows) | general intensity rubric |

Targets use the mini-Jev 5-level soft distribution (a Gaussian around the target), so fractional scores train cleanly. ESConv is skewed (initial intensity 4–5 in 77% of conversations), so it is capped per level.

## 3. Model and training (reuse mini-Jev code nearly unchanged)

- `answerdotai/ModernBERT-base` (149M) with the same three heads, the same pair encoding, and per-head temperature calibration. Fallback: `microsoft/deberta-v3-small`.
- Turns are short, so `MAX_LEN = 192` (QUICK 128). Choice batch = 8 questions; noul and score batches = 32. LR 3e-5, head LR 1e-3. These are the same as mini-Jev.
- **T4 budget:** the mini-Jev full run reached about 110k encoder rows in 45 min at MAX_LEN 320 (3,211 steps, stopped early by the time guard). At about 192 tokens, expect roughly 1.5–2× that rate.
  - **Full:** about 120k encoder rows, with `MAX_TRAIN_MINUTES = 45`:
    - noul 36k: GoEmotions 14k, DailyDialog 8k, ED 8k, ESConv 4k, plus frustration and support templates mixed in;
    - choice 12k questions (~4.5 options, so about 54k rows): GoEmotions 5k, DailyDialog 3k, ED 3k, ESConv 1k;
    - score 30k: EI-reg 7k (all), EmoBank 16k (8k sentences × the V and A rubrics), ESConv 5k, neutral anchors 2k.
  - **QUICK:** about 25k rows, 10 min.
  - The notebook prints the measured rows/s and projected time. Nothing above is measured yet.
- **Splits:** use each dataset's official train/val/test. Split by **conversation id** (DailyDialog, ED, ESConv), so no conversation appears in more than one split. Half of each validation set is used for temperature fitting and half for early sanity checks. Test sets are untouched until the final eval.
- **Never train on the box.** On the box: data inspection, the `check_notebook.py`-style dry run, and CPU latency/inference only.

## 4. Evaluation

- **choice:** accuracy and macro-F1 per source (DailyDialog macro-F1 excluding "no emotion", the standard ERC metric). Baselines are random over the offered options and majority class.
- **noul:** accuracy, F1 and AUROC per question family (emotion-X, frustrated, needs-support), against a majority baseline.
- **score:** MAE and Pearson per source. The EI-reg Pearson can be compared with the SemEval 2018 leaderboard.
- **Calibration:** NLL and **ECE** (15 bins) before and after temperature scaling, per head, plus one reliability diagram for the video.
- **CPU latency:** the same harness as `mini-jev/compare/minijev_latency.json`: median and p90 per question, and a 3-question batch, on the box CPU.
- **LLM-judge baseline:** a fixed sample of **600 held-out test turns** (200 per type, stratified by source), asked as the same questions with the same options and rubric. JSON output, temperature 0.
  - Report accuracy, MAE, ECE (from the verbalised confidence), median and p90 latency, failure/parse rate, and **cost per 1,000 turns** = measured tokens × the provider's published price on the run date.
  - **Keys available in the box env (names only, values not read):** `OPENAI_API_KEY` and `ANTHROPIC_API_KEY`. Use one cheap and one mid-tier model per provider, picked at run time.
  - `TYPESAFE_API_KEY` is also present, so the real Jev can be added as a third column by reusing `mini-jev/compare/run_jev.py`.
  - Optional "consistency" check: ask each LLM the same 100 questions twice and report the disagreement rate. Emotion-S1's disagreement is 0 by construction.
- **Domain sanity set:** about 150 synthetic "user talking to an AI assistant" turns, LLM-drafted and **checked by hand**. This shows the model works on real agent chats and not only on Reddit, tweets and scripted dialogue. Report it separately; it is not used for training.

## 5. Risks and open questions

1. **Licences:** most of the dialogue data is NC or research-only (§1). The tutorial is fine, but clients need the `COMMERCIAL_SAFE` path or their own data. EI-reg comes from an unofficial mirror of data that must not be redistributed; decide whether to download it from the official page in the notebook instead.
2. **Derived labels:** "frustrated" and "needs support" are heuristics built from emotion labels, so there is real label noise. The hand-checked domain set is the honest test, and the video should say so.
3. **Domain gap:** the data is Reddit, tweets, scripted ESL dialogue and crowd role-play, not users talking to agents. Expect lower accuracy on real logs; the fix (fine-tuning on a few thousand corrected production labels) is a good closing message for the video.
4. **Imbalance:** DailyDialog is 83% "no emotion", and fear and disgust are rare. ESConv intensity is skewed high. Needs downsampling, capping, and reporting macro-F1.
5. **Intensity scales disagree:** EI-reg, EmoBank arousal and ESConv distress measure different things, which is why each gets its own rubric wording. The score head has to follow the rubric, so check this per source.
6. **Context:** the ED and ESConv labels are per conversation, not per turn, so apply them only to early or late seeker turns as described in §2.
7. **Open questions for the user:** keep `COMMERCIAL_SAFE` on by default? Model name (placeholder "Emotion-S1")? Which LLMs to show in the comparison?
