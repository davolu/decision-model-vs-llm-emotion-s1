import os
QUICK           = False     # True: ~10 min of training (test run). False: full run, time-guarded at 45 min.
COMMERCIAL_SAFE = False    # True: train only on commercially usable data (GoEmotions + EmoBank)
TINY            = os.environ.get("EMOTION_S1_TINY") == "1"   # local CPU smoke test only: a handful of rows, a few steps

BACKBONE  = "answerdotai/ModernBERT-base"   # fallback if ModernBERT gives you trouble: "microsoft/deberta-v3-small"
OUT_DIR   = "/content/emotion_s1" if not TINY else "/tmp/emotion_s1_smoke"
DRIVE_DIR = "/content/drive/MyDrive/emotion_s1_model"
SEED      = 42
CONTEXT_DROP = 0.3         # 30% of dialogue rows lose their earlier turns, so the model works with and without context

FULL_TRAIN = {             # (question type) -> (dataset) -> number of training questions
    "noul":   dict(go=14000, dd=8000, ed=9000, es=5000),
    "choice": dict(go=5000,  dd=3000, ed=3000, es=1000),
    "score":  dict(ei=7100,  eb=16000, es=5000, anchor=2000),
}
if TINY:
    scale, N_CALIB, N_TEST = None, 6, 6
    MAX_LEN, EPOCHS, MAX_TRAIN_MINUTES, MAX_STEPS = 96, 1, 3, 3
    BATCH = dict(noul=4, choice=2, score=4)
elif QUICK:
    scale, N_CALIB, N_TEST = 0.2, 150, 300
    MAX_LEN, EPOCHS, MAX_TRAIN_MINUTES, MAX_STEPS = 128, 1, 10, None
    BATCH = dict(noul=32, choice=8, score=32)
else:
    scale, N_CALIB, N_TEST = 1.0, 300, 600
    MAX_LEN, EPOCHS, MAX_TRAIN_MINUTES, MAX_STEPS = 192, 1, 45, None
    BATCH = dict(noul=32, choice=8, score=32)    # choice batch = questions (each expands to one row per option)
N_TRAIN = {t: {s: (8 if TINY else int(n * scale)) for s, n in d.items()} for t, d in FULL_TRAIN.items()}
LR, HEAD_LR, WARMUP_FRAC, WEIGHT_DECAY = 3e-5, 1e-3, 0.06, 0.01
print("TINY" if TINY else ("QUICK" if QUICK else "FULL"), "| commercial-safe:", COMMERCIAL_SAFE, "| max_len", MAX_LEN)