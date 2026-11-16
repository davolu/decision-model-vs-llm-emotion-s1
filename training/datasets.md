# Emotion-S1 datasets and licences

I checked that all six load in the notebook's code (with `datasets` 5.x, and no loading scripts), on 16 Nov 2026.
Sizes are the rows the notebook actually builds items from.

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
