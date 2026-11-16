"""Builds emotion_s1_t4.ipynb. Edit this file (and emotion_s1_core.py), then re-run: python build_notebook.py"""
import nbformat as nbf, pathlib
HERE = pathlib.Path(__file__).parent
cells = []
def md(s): cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))
def code(s, tag):
    c = nbf.v4.new_code_cell(s.strip("\n")); c.metadata["tags"] = [tag]; cells.append(c)

# ---------------------------------------------------------------------------------------------------------------
md(r"""
# Emotion-S1: a small emotion decision model you train yourself on a free Colab T4

A lot of agent and chatbot products call an LLM **on every user turn** just to ask "how is the user feeling?".
That costs money on every turn, adds 0.5–2 seconds of latency, and can give different answers to the same message.

**Emotion-S1** is a small model built **from scratch, only for emotion**. You give it a chat turn (plus a few earlier turns if
you have them) and typed questions. It answers every question in **one forward pass**, with **calibrated probabilities**,
and it runs on a normal CPU in tens of milliseconds. It never writes free text.

| Type | Example question | What you get back |
|---|---|---|
| `choice` | "Which emotion is the user showing?" with **your own options** | a probability for every option, plus the top pick |
| `score` | "How intense is the user's emotion, 1 to 5?" | a distribution over 1–5, plus the expected score |
| `noul` (yes/no) | "Is the user frustrated?" / "Does the user need support?" | the probability of *yes* |

It's the same recipe as our earlier **mini-Jev** (one shared ModernBERT encoder and three tiny heads), but trained only on emotion data.

**How to run:** go to `Runtime → Change runtime type → T4 GPU`, then `Runtime → Run all`.
* `QUICK = True` is a short test run of about 10 minutes of training.
* `QUICK = False` is the full run. A time guard stops training at 45 minutes.

Every number this notebook prints is measured live by the notebook itself.
""")

code(r"""
# 1) Install. Colab already has PyTorch; ModernBERT needs transformers >= 4.48.
import os
if os.environ.get("EMOTION_S1_TINY") != "1":          # (the local smoke test skips installs)
    !pip -q install -U "transformers>=4.48" datasets accelerate
""", "install")

md(r"""
## 2) Configuration
* `QUICK`: a short test run, or the full run.
* `COMMERCIAL_SAFE`: this is a **tutorial**, so by default we train on all six datasets, and four of them are
  **non-commercial or research-only**. If you set it to `True`, it trains only on the two datasets that allow commercial use:
  GoEmotions (Apache-2.0) and EmoBank (CC BY-SA 4.0). See the licence table in the next section.
* `N_TRAIN`: how many training questions to build for each (question type, dataset). Each choice question becomes
  one encoder row per option.
* `MAX_TRAIN_MINUTES`: the time guard. Training stops when it reaches this, whatever happens.
""")
code(r"""
import os
QUICK           = True     # True: ~10 min of training (test run). False: full run, time-guarded at 45 min.
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
""", "config")

code(r"""
import time, math, json, random, io, zipfile, urllib.request, collections
import numpy as np, pandas as pd, torch, torch.nn.functional as F
from datasets import load_dataset
from transformers import get_linear_schedule_with_warmup

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP = DEVICE == "cuda"
print("device:", DEVICE, torch.cuda.get_device_name(0) if DEVICE == "cuda" else "(no GPU: fine for the smoke test, far too slow for real training)")
""", "imports")

# ---------------------------------------------------------------------------------------------------------------
md(r"""
## 3) The model: one encoder, three typed heads
The encoder reads a **text pair**:
* **A** is the question type, plus the question or rubric, plus the candidate option for a `choice` question;
* **B** is the *state*: up to 4 earlier chat turns, then the user's latest turn.

The averaged encoder output feeds three tiny heads:
* `noul` gives 1 logit, and a sigmoid turns it into P(yes);
* `choice` gives 1 logit **per option row**, softmaxed across that question's options, so any number of options works;
* `score` gives 5 logits, a distribution over levels 1 to 5, and its average is the expected score.

The next cell writes the model code to `emotion_s1_core.py`. The **same file** is saved next to the weights, so inference
later uses exactly the same input format as training.
""")
code("%%writefile emotion_s1_core.py\n" + (HERE / "emotion_s1_core.py").read_text(), "core")
code(r"""
import importlib, sys
sys.path.insert(0, os.getcwd())
import emotion_s1_core; importlib.reload(emotion_s1_core)
from emotion_s1_core import EmotionS1, load_emotion_s1, format_state, pair_noul, pair_choice, pair_score, ask, ask_state, N_BINS
model = EmotionS1(BACKBONE, max_len=MAX_LEN, device=DEVICE, use_amp=USE_AMP).to(DEVICE)
print(f"parameters: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
print(format_state("It's still broken.", [("User", "My app crashes."), ("Assistant", "Try reinstalling it.")]))
""", "model")

# ---------------------------------------------------------------------------------------------------------------
md(r"""
## 4) The data: six free emotion datasets

| Short name | Dataset | What it gives us | Licence | Commercial use? |
|---|---|---|---|---|
| `go` | **GoEmotions** (Google, Reddit comments) | 27 emotions + neutral, per message | Apache-2.0 | ✅ yes |
| `eb` | **EmoBank** (JULIE Lab) | valence and arousal ratings, 1–5 | CC BY-SA 4.0 | ✅ yes (share-alike) |
| `dd` | **DailyDialog** | an emotion label on **every turn** of a two-person chat | CC BY-NC-SA 4.0 | ❌ non-commercial |
| `ed` | **EmpatheticDialogues** (Meta) | 32 fine-grained emotions per conversation | CC BY-NC 4.0 | ❌ non-commercial |
| `es` | **ESConv** (emotional-support chats) | the seeker's emotion type, plus distress 1–5 at the start and end | CC BY-NC 4.0 | ❌ non-commercial |
| `ei` | **SemEval-2018 Task 1, EI-reg** (tweets) | anger, fear, joy and sadness intensity, 0–1 | research only, no redistribution | ❌ research only |

All of them load directly. We use the official parquet copies on Hugging Face, the official EmoBank CSV on GitHub, and the official
SemEval zip from the task organiser's site. If that site is down, we fall back to an unofficial Hugging Face mirror.
We use each dataset's own **train / validation / test** splits. DailyDialog, EmpatheticDialogues and ESConv are split by **conversation**,
so no chat appears in both training and test. Validation fits the calibration, and test produces every reported number.
""")
code(r"""
def clean(s): return " ".join(str(s).replace("_comma_", ",").split())

def split_ctx(turns, i, target_role_of):
    # earlier turns (oldest first) as (role, text); the target speaker is always "User"
    return [(target_role_of(j), clean(turns[j])) for j in range(max(0, i - 4), i)]

RAW = {}   # src -> split -> list of base items
def item(src, turn, context=(), **kw):
    return dict(src=src, turn=clean(turn), context=list(context), **kw)

# --- GoEmotions (Apache-2.0) -----------------------------------------------------------------------------
go = load_dataset("google-research-datasets/go_emotions", "simplified")
GO_NAMES = go["train"].features["labels"].feature.names
RAW["go"] = {sp: [item("go", r["text"], labels=[GO_NAMES[i] for i in r["labels"]]) for r in go[sp]]
             for sp in ["train", "validation", "test"]}

# --- EmoBank (CC BY-SA 4.0): official CSV, V/A/D on a 1-5 scale ------------------------------------------
eb = pd.read_csv("https://raw.githubusercontent.com/JULIELab/EmoBank/master/corpus/emobank.csv")
RAW["eb"] = {sp: [item("eb", r.text, V=float(r.V), A=float(r.A)) for r in eb[eb.split == s].itertuples()]
             for sp, s in [("train", "train"), ("validation", "dev"), ("test", "test")]}

if not COMMERCIAL_SAFE:
    # --- DailyDialog (CC BY-NC-SA 4.0): parquet revision (the main branch is an old loading script) -------
    DD_NAMES = ["neutral", "anger", "disgust", "fear", "happiness", "sadness", "surprise"]   # 0 = "no emotion"
    dd = load_dataset("li2017dailydialog/daily_dialog", revision="refs/convert/parquet")
    RAW["dd"] = {}
    for sp in ["train", "validation", "test"]:
        rows, rng = [], random.Random(SEED)
        for d in dd[sp]:
            for i, (t, e) in enumerate(zip(d["dialog"], d["emotion"])):
                if e == 0 and rng.random() > 0.12: continue     # 83% of turns are "no emotion": keep ~1 in 8
                ctx = split_ctx(d["dialog"], i, lambda j, i=i: "User" if (i - j) % 2 == 0 else "Assistant")
                rows.append(item("dd", t, ctx, labels=[DD_NAMES[e]]))
        RAW["dd"][sp] = rows

    # --- EmpatheticDialogues (CC BY-NC 4.0): label = the emotion of the speaker who opens the conversation --
    ed = load_dataset("facebook/empathetic_dialogues", revision="refs/convert/parquet")
    RAW["ed"] = {}
    for sp in ["train", "validation", "test"]:
        convs = collections.defaultdict(list)
        for r in ed[sp]: convs[r["conv_id"]].append(r)
        rows = []
        for cid, us in convs.items():
            us.sort(key=lambda r: r["utterance_idx"]); texts = [u["utterance"] for u in us]
            spk = us[0]["speaker_idx"]
            for i, u in enumerate(us[:4]):                       # the emoter's EARLY turns carry the label best
                if u["speaker_idx"] != spk: continue
                ctx = split_ctx(texts, i, lambda j: "User" if us[j]["speaker_idx"] == spk else "Assistant")
                rows.append(item("ed", u["utterance"], ctx, labels=[us[0]["context"]]))
        RAW["ed"][sp] = rows

    # --- ESConv (CC BY-NC 4.0): one JSON conversation per row ------------------------------------------
    es = load_dataset("thu-coai/esconv")
    RAW["es"] = {}
    for sp in ["train", "validation", "test"]:
        rows = []
        for r in es[sp]:
            j = json.loads(r["text"]); dlg = j["dialog"]; s = j["survey_score"]["seeker"]
            texts = [u["text"] for u in dlg]
            role = lambda k: "User" if dlg[k]["speaker"] == "usr" else "Assistant"
            usr = [k for k, u in enumerate(dlg) if u["speaker"] == "usr"]
            init, fin = s.get("initial_emotion_intensity"), s.get("final_emotion_intensity")
            for n, k in enumerate(usr):
                early, late = n < 3, n >= len(usr) - 2 and n >= 3
                if not (early or late): continue
                score = float(init) if early and init else (float(fin) if late and fin else None)
                rows.append(item("es", texts[k], split_ctx(texts, k, role),
                                 labels=[j["emotion_type"]] if early else None,
                                 support=(float(init) >= 3) if early and init else None, distress=score))
        RAW["es"][sp] = rows

    # --- SemEval-2018 EI-reg (research only): official zip, fallback to an unofficial HF mirror ----------
    def load_eireg():
        url = "https://saifmohammad.com/WebDocs/AIT-2018/AIT2018-DATA/SemEval2018-Task1-all-data.zip"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (emotion-s1 tutorial)", "Accept": "*/*"})
            z = zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(req, timeout=60).read()))
            out = {"train": [], "validation": [], "test": []}
            for n in z.namelist():
                if "__MACOSX" in n or "/English/EI-reg/" not in n or not n.endswith(".txt"): continue
                sp = "train" if "/training/" in n else "validation" if "/development/" in n else "test"
                df = pd.read_csv(io.BytesIO(z.read(n)), sep="\t", quoting=3)
                out[sp] += [item("ei", r[1], emo=r[2], value=float(r[3])) for r in df.itertuples(index=False)]
            print("EI-reg: official SemEval zip")
            return out
        except Exception as e:
            print("EI-reg: official download failed (", repr(e)[:80], ") -> using the unofficial HF mirror Holmeister/ei_reg_*")
            out = {"train": [], "validation": [], "test": []}
            for emo in ["anger", "fear", "joy", "sadness"]:
                d = load_dataset(f"Holmeister/ei_reg_{emo}")
                for sp in out: out[sp] += [item("ei", r["input"], emo=emo, value=float(r["output"])) for r in d[sp]]
            return out
    RAW["ei"] = load_eireg()

for s, d in RAW.items():
    print(f"{s:3s}", {sp: len(v) for sp, v in d.items()})
""", "data_load")

md(r"""
## 5) Turning labels into typed questions
The datasets only contain labels. We turn every labelled turn into **questions**, using several phrasings so the model learns
to *read the question* instead of memorising one template.

* **Choice:** "Which emotion is the user showing?" The options are the correct label plus **2 to 6 random wrong ones**,
  shuffled every time and drawn from different label sets (6 basic emotions, GoEmotions-27, EmpatheticDialogues-32, ESConv).
  Near-synonyms are **never** offered as rival options (for example, fear and nervousness are in the same *cluster*).
* **Yes/No:** "Is the user expressing {emotion}?" uses real negatives, half of them **hard negatives** of the same polarity
  (fear vs sadness). "Is the user frustrated?" and "Does the user need support?" are **derived** from the emotion labels,
  for example GoEmotions anger, annoyance or disappointment counts as frustrated. That makes them a little noisy.
  There are also "Is the user calm / neutral?" and "Is the user in a positive mood?" questions, so "negative" doesn't always mean "yes".
* **Score (1–5):** SemEval intensity `1 + 4x`; EmoBank arousal and valence stretched from their observed range to 1–5;
  ESConv distress 1–5 as given; and neutral turns as "intensity 1" anchors. Each source keeps its **own rubric wording**,
  and some rows use a general "how intense is the emotion?" rubric.
""")
code(r"""
EKMAN = {"anger": ["anger", "annoyance", "disapproval"], "disgust": ["disgust"], "fear": ["fear", "nervousness"],
         "joy": ["joy", "amusement", "approval", "excitement", "gratitude", "love", "optimism", "relief", "pride",
                 "admiration", "desire", "caring"],
         "sadness": ["sadness", "disappointment", "embarrassment", "grief", "remorse"],
         "surprise": ["surprise", "realization", "confusion", "curiosity"]}        # official GoEmotions Ekman mapping
CLUSTER = {"neutral": "neutral", "happiness": "joy", "no emotion": "neutral"}
for ek, labs in EKMAN.items():
    CLUSTER[ek] = ek
    for l in labs: CLUSTER[l] = ek
CLUSTER.update({  # EmpatheticDialogues 32 + ESConv types
    "afraid": "fear", "terrified": "fear", "anxious": "fear", "apprehensive": "fear", "anxiety": "fear",
    "angry": "anger", "annoyed": "anger", "furious": "anger", "jealous": "anger", "jealousy": "anger",
    "sad": "sadness", "lonely": "sadness", "devastated": "sadness", "disappointed": "sadness", "ashamed": "sadness",
    "guilty": "sadness", "embarrassed": "sadness", "depression": "sadness", "shame": "sadness", "guilt": "sadness", "pain": "sadness",
    "disgusted": "disgust", "surprised": "surprise", "impressed": "surprise",
    "joyful": "joy", "excited": "joy", "content": "joy", "proud": "joy", "grateful": "joy", "hopeful": "joy",
    "confident": "joy", "caring": "joy", "trusting": "joy", "faithful": "joy", "prepared": "joy", "anticipating": "joy",
    "nostalgic": "nostalgia", "sentimental": "nostalgia"})
POLARITY = {"anger": "neg", "disgust": "neg", "fear": "neg", "sadness": "neg", "joy": "pos",
            "surprise": "mixed", "nostalgia": "mixed", "neutral": "none"}

LABELSETS = {"ekman": ["anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral"],
             "go27": GO_NAMES,
             "dd": ["neutral", "anger", "disgust", "fear", "happiness", "sadness", "surprise"],
             "ed": sorted({k for k, v in CLUSTER.items() if k in {
                 "afraid", "terrified", "anxious", "apprehensive", "angry", "annoyed", "furious", "jealous", "sad", "lonely",
                 "devastated", "disappointed", "ashamed", "guilty", "embarrassed", "disgusted", "surprised", "impressed",
                 "joyful", "excited", "content", "proud", "grateful", "hopeful", "confident", "caring", "trusting",
                 "faithful", "prepared", "anticipating", "nostalgic", "sentimental"}}),
             "es": ["anxiety", "depression", "sadness", "anger", "fear", "disgust", "shame"]}
RENDER = {"joy": ["joy", "happiness"], "happiness": ["happiness", "joy"], "neutral": ["neutral", "no particular emotion"],
          "anger": ["anger", "anger / irritation"], "sadness": ["sadness"], "fear": ["fear", "fear / worry"]}
def render(l, rng): return rng.choice(RENDER.get(l, [l]))

FRUSTRATED = {"anger", "annoyance", "disappointment", "angry", "annoyed", "furious", "disappointed"}
NEEDS_SUPPORT = {"grief", "sadness", "fear", "nervousness", "remorse", "embarrassment", "sad", "lonely", "devastated",
                 "afraid", "anxious", "ashamed", "terrified", "guilty", "apprehensive", "disappointed", "embarrassed"}

Q_CHOICE = ["Which emotion is the user showing?", "What is the user mainly feeling in their latest message?",
            "Which of these best describes the user's emotional state?", "Pick the emotion that best matches the user's latest turn.",
            "How is the user feeling right now?", "What emotion does the user express?"]
Q_EMO = ["Is the user expressing {x}?", "Does the user's latest message show {x}?",
         "Would you describe the user's emotion as {x}?", "Is {x} the main emotion in the user's message?"]
Q_FRUS = ["Is the user frustrated?", "Is the user annoyed or frustrated?", "Does the user sound irritated or fed up?",
          "Is the user getting frustrated?"]
Q_SUPP = ["Does the user need emotional support?", "Should the assistant respond with empathy and support here?",
          "Is the user in distress and in need of support?", "Does the user seem to be struggling emotionally?"]
Q_NEUT = ["Is the user calm and emotionally neutral?", "Is the user's message emotionally neutral?"]
Q_POS  = ["Is the user in a positive mood?", "Does the user sound happy or positive?"]
R_GEN  = ["How intense is the emotion the user is expressing? 1 = calm / no emotion, 5 = extremely intense.",
          "Rate the strength of the user's feelings in their latest turn. 1 = not emotional at all, 5 = overwhelming.",
          "How emotionally intense is the user right now, from 1 (calm) to 5 (very intense)?"]
R_EMO  = ["How intense is the user's {e}? 1 = no {e} at all, 5 = extreme {e}.",
          "Rate how strongly the user expresses {e}, from 1 (none) to 5 (very strongly)."]
R_DIST = ["How distressed is the user? 1 = not at all, 5 = extremely distressed.",
          "Rate the intensity of the user's negative feelings, from 1 (mild) to 5 (severe)."]
R_VAL  = ["How positive or negative is the user's mood? 1 = very negative, 5 = very positive.",
          "Rate the user's mood from 1 (very unhappy) to 5 (very happy)."]
R_ARO  = ["How emotionally aroused or activated is the user? 1 = very calm, 5 = very excited or agitated."]

def stretch(v, lo, hi): return float(min(5.0, max(1.0, 1 + 4 * (v - lo) / (hi - lo))))
EB_A = (1.8, 4.4); EB_V = (1.2, 4.6)          # observed EmoBank ranges (train), stretched to 1-5

def views(it):
    # (labelset name, gold labels) pairs available for this item
    if not it.get("labels"): return []
    if it["src"] == "go":
        ek = sorted({CLUSTER[l] for l in it["labels"]})
        return [("go27", it["labels"]), ("ekman", ek)]
    return [({"dd": "dd", "ed": "ed", "es": "es"}[it["src"]], it["labels"])]

def state_of(it, rng, train):
    ctx = it["context"] if not (train and rng.random() < CONTEXT_DROP) else []
    return format_state(it["turn"], ctx)

def make_choice(it, rng, train):
    vs = views(it)
    if not vs: return None
    if it["src"] == "go" and not train: vs = [v for v in vs if v[0] == "ekman"]          # eval: fixed 7-way Ekman set
    name, gold = rng.choice(vs)
    if it["src"] == "es" and gold[0] not in LABELSETS["es"]: return None
    ans = rng.choice(gold); gc = {CLUSTER[g] for g in gold}
    pool = [l for l in LABELSETS[name] if CLUSTER[l] not in gc]
    if not pool: return None
    full = name in ("ekman", "dd", "es") and (not train or rng.random() < 0.15)
    if full:  # the whole label set, minus same-cluster rivals of the answer
        opts = [ans] + [l for l in LABELSETS[name] if CLUSTER[l] not in gc]
    else:
        k = 5 if not train else rng.randint(2, 6)
        opts = [ans] + rng.sample(pool, min(k, len(pool)))
    rng.shuffle(opts)
    shown = [render(o, rng) if train else o for o in opts]
    return dict(task="choice", src=it["src"], fam=name, state=state_of(it, rng, train), question=rng.choice(Q_CHOICE),
                options=shown, label=opts.index(ans), gold=shown[opts.index(ans)])

def make_noul(it, rng, train):
    labs = it.get("labels")
    fams = []
    if labs: fams += ["emotion"] * 5
    if labs and it["src"] != "es": fams += ["frustrated"] * 2
    sup = it.get("support")
    if sup is None and labs and it["src"] != "es":
        if any(l in NEEDS_SUPPORT for l in labs): sup = True
        elif all(POLARITY[CLUSTER[l]] in ("pos", "none") for l in labs): sup = False
    if sup is not None: fams += ["support"] * 2
    if labs and it["src"] in ("go", "dd"): fams += ["neutral"]
    if labs and it["src"] != "es": fams += ["positive"]
    if not fams: return None
    fam = rng.choice(fams); st = state_of(it, rng, train)
    if fam == "emotion":
        name, gold = rng.choice(views(it)); gc = {CLUSTER[g] for g in gold}
        if rng.random() < 0.5:
            x, y = rng.choice(gold), 1
        else:
            pol = {POLARITY[c] for c in gc}
            hard = [l for l in LABELSETS[name] if CLUSTER[l] not in gc and POLARITY[CLUSTER[l]] in pol]
            easy = [l for l in LABELSETS[name] if CLUSTER[l] not in gc]
            cand = hard if hard and rng.random() < 0.5 else easy
            if not cand: return None
            x, y = rng.choice(cand), 0
        q = rng.choice(Q_EMO).format(x=render(x, rng) if train else x)
    elif fam == "frustrated":
        q, y = rng.choice(Q_FRUS), int(any(l in FRUSTRATED or (it["src"] == "dd" and l in ("anger", "disgust")) for l in labs))
    elif fam == "support":
        q, y = rng.choice(Q_SUPP), int(sup)
    elif fam == "neutral":
        q, y = rng.choice(Q_NEUT), int(all(CLUSTER[l] == "neutral" for l in labs))
    else:
        q, y = rng.choice(Q_POS), int(all(POLARITY[CLUSTER[l]] == "pos" for l in labs))
    return dict(task="noul", src=it["src"], fam=fam, state=st, question=q, label=y)

def make_score(it, rng, train):
    s = it["src"]
    if s == "ei":
        e = it["emo"]; v = 1 + 4 * it["value"]
        r, fam = (rng.choice(R_GEN), "general") if rng.random() < 0.3 else (rng.choice(R_EMO).format(e=e), f"intensity_{e}")
    elif s == "eb":
        if rng.random() < 0.5: v, r, fam = stretch(it["V"], *EB_V), rng.choice(R_VAL), "valence"
        else:
            v = stretch(it["A"], *EB_A)
            r, fam = (rng.choice(R_GEN), "general") if rng.random() < 0.3 else (rng.choice(R_ARO), "arousal")
    elif s == "es":
        if it.get("distress") is None: return None
        v = it["distress"]; r, fam = (rng.choice(R_GEN), "general") if rng.random() < 0.5 else (rng.choice(R_DIST), "distress")
    elif s in ("go", "dd") and it.get("labels") and all(CLUSTER[l] == "neutral" for l in it["labels"]):
        v, r, fam, s = 1.0, rng.choice(R_GEN), "general", "anchor"
    else:
        return None
    return dict(task="score", src=s, fam=fam, state=state_of(it, rng, train), rubric=r, scale=(1, 5), value=float(v))

MAKERS = {"noul": make_noul, "choice": make_choice, "score": make_score}

def build(split, budget, seed):
    rng = random.Random(seed); out = []
    for task, per_src in budget.items():
        for src, n in per_src.items():
            pool_src = "go" if src == "anchor" else src
            items = RAW.get(pool_src, {}).get(split, [])
            if src == "anchor": items = items + RAW.get("dd", {}).get(split, [])
            if not items or n <= 0: continue
            got, tries = [], 0
            order = list(range(len(items)))
            while len(got) < n and tries < 3:          # up to 3 passes, each with fresh random questions
                rng.shuffle(order)
                for i in order:
                    ex = MAKERS[task](items[i], rng, split == "train")
                    if ex and ex["src"] == src:
                        got.append(ex)
                        if len(got) >= n: break
                tries += 1
            out += got
    for i, ex in enumerate(out): ex["id"] = f"{split}-{i}"
    return out

from_budget = lambda n: {t: {s: n for s in d} for t, d in FULL_TRAIN.items()}
train_ex = build("train", N_TRAIN, SEED)
calib_ex = build("validation", from_budget(N_CALIB), SEED + 1)    # fits the calibration temperatures
test_ex  = build("test", from_budget(N_TEST), SEED + 2)           # every reported number comes from here
for name, exs in [("train", train_ex), ("calib", calib_ex), ("test", test_ex)]:
    c = collections.Counter((e["task"], e["src"]) for e in exs)
    rows = sum(len(e["options"]) if e["task"] == "choice" else 1 for e in exs)
    print(f"{name:5s} {len(exs):6d} questions, {rows:6d} encoder rows ", dict(sorted(c.items())))
for t in ["choice", "noul", "score"]:
    ex = next(e for e in train_ex if e["task"] == t)
    print(f"\n--- example {t} ---\n{ex['state']}\nQ: {ex.get('question') or ex.get('rubric')}",
          f"\noptions: {ex['options']}  answer: {ex['gold']}" if t == "choice" else f"\nanswer: {ex.get('label', ex.get('value'))}")
""", "questions")

# ---------------------------------------------------------------------------------------------------------------
md(r"""
## 6) Train (multi-task)
Each batch contains a single question type (all yes/no, all choice or all score), and the batches are shuffled together, so every step
updates the shared encoder. After 20 steps the cell prints the **measured** speed and a projected training time. Training
stops early if it would go past `MAX_TRAIN_MINUTES`.
""")
code(r"""
def collate(task, exs):
    if task == "noul":
        return dict(enc=model.encode_pairs([pair_noul(e["state"], e["question"]) for e in exs]),
                    y=torch.tensor([e["label"] for e in exs], dtype=torch.float32))
    if task == "choice":
        pairs, rows, cols = [], [], []
        for i, e in enumerate(exs):
            for j, o in enumerate(e["options"]):
                pairs.append(pair_choice(e["state"], e["question"], o)); rows.append(i); cols.append(j)
        return dict(enc=model.encode_pairs(pairs), rows=torch.tensor(rows), cols=torch.tensor(cols),
                    n=len(exs), width=max(len(e["options"]) for e in exs), y=torch.tensor([e["label"] for e in exs]))
    return dict(enc=model.encode_pairs([pair_score(e["state"], e["rubric"], e["scale"]) for e in exs]),
                y=torch.tensor(np.stack([soft_target(e["value"], e["scale"]) for e in exs])))

def soft_target(value, scale, k=N_BINS):
    # a fractional score such as 3.4 becomes 60% on level 3 and 40% on level 4
    lo, hi = scale; pos = (min(max(value, lo), hi) - lo) / (hi - lo) * (k - 1)
    t = np.zeros(k, dtype=np.float32); i = int(math.floor(pos)); f = pos - i
    t[i] += 1 - f
    if f > 0: t[i + 1] += f
    return t

def task_logits(task, b):
    ln, lc, ls = model.heads(model.pool(**b["enc"]))
    if task == "noul": return ln
    if task == "score": return ls
    out = torch.full((b["n"], b["width"]), -1e4, device=lc.device)
    out[b["rows"].to(lc.device), b["cols"].to(lc.device)] = lc
    return out

def task_loss(task, logits, y):
    y = y.to(logits.device)
    if task == "noul":   return F.binary_cross_entropy_with_logits(logits, y)
    if task == "choice": return F.cross_entropy(logits, y)
    return -(y * F.log_softmax(logits, -1)).sum(-1).mean()

def make_batches(examples):
    by = {"noul": [], "choice": [], "score": []}
    for e in examples: by[e["task"]].append(e)
    batches = []
    for t, exs in by.items():
        random.shuffle(exs); batches += [(t, exs[i:i + BATCH[t]]) for i in range(0, len(exs), BATCH[t])]
    random.shuffle(batches); return batches
def rows_in(t, exs): return sum(len(e["options"]) for e in exs) if t == "choice" else len(exs)

head_params = [p for n, p in model.named_parameters() if not n.startswith("enc.")]
opt = torch.optim.AdamW([{"params": model.enc.parameters(), "lr": LR}, {"params": head_params, "lr": HEAD_LR}],
                        weight_decay=WEIGHT_DECAY)
plan = [make_batches(train_ex) for _ in range(EPOCHS)]
total_steps = sum(len(p) for p in plan) if not MAX_STEPS else min(MAX_STEPS, sum(len(p) for p in plan))
sched = get_linear_schedule_with_warmup(opt, int(WARMUP_FRAC * total_steps), total_steps)
scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)

model.train(); t0 = time.time(); step = 0; seen_rows = 0; stopped_early = False
running = {"noul": [], "choice": [], "score": []}
for ep, batches in enumerate(plan):
    for task, exs in batches:
        b = collate(task, exs)
        loss = task_loss(task, task_logits(task, b), b["y"])
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update(); sched.step()
        step += 1; seen_rows += rows_in(task, exs); running[task].append(loss.item())
        el = time.time() - t0
        if step == 20:
            print(f"[speed] {seen_rows/el:.1f} encoder rows/s -> projected {el/step*total_steps/60:.1f} min for {total_steps} steps")
        if step % 100 == 0 or step == total_steps:
            avg = {k: round(float(np.mean(v[-50:])), 3) for k, v in running.items() if v}
            print(f"epoch {ep+1} step {step}/{total_steps}  {el/60:.1f} min  loss {avg}")
        if el > MAX_TRAIN_MINUTES * 60:
            print(f"Time guard hit at step {step}/{total_steps}; stopping training."); stopped_early = True; break
        if MAX_STEPS and step >= MAX_STEPS: break
    if stopped_early or (MAX_STEPS and step >= MAX_STEPS): break
TRAIN_MINUTES = (time.time() - t0) / 60
print(f"training done: {step} steps, {seen_rows} encoder rows, {TRAIN_MINUTES:.1f} min")
model.eval();
""", "train")

# ---------------------------------------------------------------------------------------------------------------
md(r"""
## 7) Calibrate and evaluate on held-out test turns
**Temperature scaling:** we fit one number per head on the *validation* questions by minimising log-loss. It makes the confidences
honest, so when the model says it's 80% sure it should be right about 80% of the time.
**ECE** (expected calibration error, 15 bins) measures that gap. Lower is better, and 0 is perfect. We report it before and after.

Metrics are computed on the **test** questions only:
* **choice:** accuracy (vs. picking randomly) and macro-F1. GoEmotions is asked as a fixed 7-way Ekman choice, DailyDialog over
  its 7 labels, ESConv over its 7 emotion types, and EmpatheticDialogues over 6 options. DailyDialog's macro-F1 excludes "neutral",
  as in the research literature. Note that we downsampled its neutral turns.
* **yes/no:** accuracy, F1 and AUROC per question family (emotion-X, frustrated, needs-support, neutral, positive), vs. always
  giving the majority answer.
* **score:** MAE and Pearson correlation per source, vs. always predicting the training mean.
""")
code(r"""
@torch.no_grad()
def collect(examples, task):
    exs = [e for e in examples if e["task"] == task]
    if not exs: return exs, None, None
    W = max(len(e["options"]) for e in exs) if task == "choice" else 0
    L, Y, bs = [], [], BATCH[task] * 2
    for i in range(0, len(exs), bs):
        b = collate(task, exs[i:i + bs]); lg = task_logits(task, b).float().cpu()
        if task == "choice" and lg.shape[1] < W: lg = F.pad(lg, (0, W - lg.shape[1]), value=-1e4)
        L.append(lg); Y.append(b["y"])
    return exs, torch.cat(L), torch.cat(Y)

def tempered(task, logits, T):
    z = logits / T
    return z.masked_fill(logits <= -1e3, -1e9) if task == "choice" else z

def nll(task, logits, y, T):
    z = tempered(task, logits, T)
    if task == "noul":   return F.binary_cross_entropy_with_logits(z, y)
    if task == "choice": return F.cross_entropy(z, y)
    return -(y * F.log_softmax(z, -1)).sum(-1).mean()

def fit_temperature(task, logits, y):
    logT = torch.zeros(1, requires_grad=True); o = torch.optim.LBFGS([logT], lr=0.1, max_iter=200)
    def closure():
        o.zero_grad(); l = nll(task, logits, y, logT.exp()); l.backward(); return l
    o.step(closure); return float(logT.exp().clamp(0.05, 20.0).item())

def ece(conf, correct, n_bins=15):
    conf, correct = np.asarray(conf), np.asarray(correct, dtype=float); e = 0.0
    edges = np.linspace(0, 1, n_bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any(): e += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return float(e)

def conf_correct(task, logits, y, T):
    if task == "noul":
        p = torch.sigmoid(logits / T).numpy(); return np.maximum(p, 1 - p), (p >= 0.5) == y.numpy().astype(bool)
    P = torch.softmax(tempered(task, logits, T), -1).numpy()
    if task == "choice": return P.max(1), P.argmax(1) == y.numpy()
    return P.max(1), P.argmax(1) == y.numpy().argmax(1)       # score: top level vs the level nearest the true value

def auroc(p, y):
    y = np.asarray(y).astype(bool); pos, neg = y.sum(), (~y).sum()
    if pos == 0 or neg == 0: return None
    r = pd.Series(p).rank().values; return float((r[y].sum() - pos * (pos + 1) / 2) / (pos * neg))

def f1(pred, y, cls=True):
    tp = np.sum((pred == cls) & (y == cls)); fp = np.sum((pred == cls) & (y != cls)); fn = np.sum((pred != cls) & (y == cls))
    return float(2 * tp / (2 * tp + fp + fn)) if tp + fp + fn else 0.0

R = lambda x: None if x is None else round(float(x), 4)
TEMPS, RESULTS = {}, {}
for i, task in enumerate(["noul", "choice", "score"]):
    _, Lc, Yc = collect(calib_ex, task)
    T = fit_temperature(task, Lc, Yc); TEMPS[task] = T; model.temps[i] = T
    exs, Lt, Yt = collect(test_ex, task)
    c0, k0 = conf_correct(task, Lt, Yt, 1.0); c1, k1 = conf_correct(task, Lt, Yt, T)
    res = dict(temperature=R(T), n_test=len(exs), nll_before=R(nll(task, Lt, Yt, 1.0)), nll_after=R(nll(task, Lt, Yt, T)),
               ece_before=R(ece(c0, k0)), ece_after=R(ece(c1, k1)), groups={})
    if task == "noul":
        p = torch.sigmoid(Lt / T).numpy(); yy = Yt.numpy().astype(bool); pred = p >= 0.5
        keys = np.array([e["fam"] for e in exs])
        for k in sorted(set(keys)):
            m = keys == k; maj = np.mean([e["label"] for e in train_ex if e["task"] == "noul" and e["fam"] == k]) >= 0.5
            res["groups"][k] = dict(acc=R((pred[m] == yy[m]).mean()), f1=R(f1(pred[m], yy[m])), auroc=R(auroc(p[m], yy[m])),
                                    majority_baseline=R((yy[m] == maj).mean()), positive_rate=R(yy[m].mean()), n=int(m.sum()))
    elif task == "choice":
        P = torch.softmax(tempered(task, Lt, T), -1).numpy(); pred = P.argmax(1); yy = Yt.numpy()
        srcs = np.array([e["src"] for e in exs])
        for s in sorted(set(srcs)):
            idx = np.where(srcs == s)[0]
            g = np.array([exs[j]["options"][yy[j]] for j in idx]); pr = np.array([exs[j]["options"][pred[j]] for j in idx])
            classes = sorted(set(g) - ({"neutral"} if s == "dd" else set()))
            res["groups"][s] = dict(acc=R((pred[idx] == yy[idx]).mean()),
                                    random_baseline=R(np.mean([1 / len(exs[j]["options"]) for j in idx])),
                                    macro_f1=R(np.mean([f1(pr, g, c) for c in classes])) if classes else None, n=int(len(idx)))
    else:
        P = torch.softmax(Lt / T, -1).numpy(); ev = P @ np.linspace(1, 5, N_BINS)
        true = np.array([e["value"] for e in exs]); srcs = np.array([e["src"] for e in exs])
        for s in sorted(set(srcs)):
            m = srcs == s; tr_mean = np.mean([e["value"] for e in train_ex if e["task"] == "score" and e["src"] == s])
            res["groups"][s] = dict(mae=R(np.abs(ev[m] - true[m]).mean()), within_1=R((np.abs(ev[m] - true[m]) <= 1).mean()),
                                    pearson=R(np.corrcoef(ev[m], true[m])[0, 1]) if m.sum() > 2 and true[m].std() > 0 else None,
                                    mae_baseline_train_mean=R(np.abs(tr_mean - true[m]).mean()), n=int(m.sum()))
    RESULTS[task] = res
print(json.dumps(RESULTS, indent=2))
""", "eval")

# ---------------------------------------------------------------------------------------------------------------
md(r"""
## 8) Demo: ask questions about real-looking chat turns
`ask(model, turn, questions, context=...)` sends **all questions in one forward pass**. The options and rubrics are yours to choose.
These are made-up support-bot and coaching chats the model never saw during training.
""")
code(r"""
EMOTIONS = ["joy", "sadness", "anger", "fear", "surprise", "neutral"]
QS = {
    "emotion":   {"type": "choice", "question": "Which emotion is the user showing?", "options": EMOTIONS},
    "intensity": {"type": "score", "question": "How intense is the user's emotion? 1 = calm, 5 = extremely intense.", "scale": [1, 5]},
    "frustrated": {"type": "noul", "question": "Is the user frustrated?"},
    "support":   {"type": "noul", "question": "Does the user need emotional support?"},
}
DEMOS = [
    ("Support bot, third failed fix",
     "This is the THIRD time I've reset the router and it's still dropping every ten minutes. I'm done.",
     [("User", "My internet keeps cutting out."), ("Assistant", "Please try restarting your router."),
      ("User", "Did that already."), ("Assistant", "Could you try a factory reset?")]),
    ("Happy customer", "Oh wow, it works now! Thank you so much, you saved my evening.", 
     [("Assistant", "I've re-sent the activation link, can you try again?")]),
    ("Coaching app, anxious user", "I have my job interview tomorrow and I can't stop shaking. What if I freeze?", []),
    ("Neutral question", "What time does the store open on Saturdays?", []),
]
for title, turn, ctx in DEMOS:
    a = ask(model, turn, QS, context=ctx)
    print(f"=== {title} ===\nUser: {turn}")
    print(f"  emotion   -> {a['emotion']['answer']} ({a['emotion']['confidence']:.0%})")
    print(f"  intensity -> {a['intensity']['expected']} / 5")
    print(f"  frustrated-> P(yes) = {a['frustrated']['probability']:.2f}")
    print(f"  support   -> P(yes) = {a['support']['probability']:.2f}\n")
print(json.dumps(ask(model, DEMOS[0][1], QS, context=DEMOS[0][2]), indent=1)[:1200])
""", "demo")

# ---------------------------------------------------------------------------------------------------------------
md(r"""
## 9) Latency: GPU, and the number that matters, CPU
In production you'd run this on a plain CPU, so we time it on Colab's CPU as well as the GPU. Colab's free CPU has only 2 vCPUs,
so a normal server CPU will be faster. One "call" is one forward pass for all the questions you ask.
""")
code(r"""
def bench(device, n=20):
    model.to_device(device)
    one = {"frustrated": QS["frustrated"]}; turn, ctx = DEMOS[0][1], DEMOS[0][2]
    for _ in range(3): ask(model, turn, QS, context=ctx)                     # warm-up
    res = {}
    for name, qs in [("one_yes_no_question", one), ("all_4_questions", QS)]:
        ts = []
        for _ in range(n):
            if device == "cuda": torch.cuda.synchronize()
            t = time.perf_counter(); ask(model, turn, qs, context=ctx)
            if device == "cuda": torch.cuda.synchronize()
            ts.append((time.perf_counter() - t) * 1000)
        rows = sum(len(q["options"]) if q["type"] == "choice" else 1 for q in qs.values())
        res[name] = dict(encoder_rows=rows, median_ms=round(float(np.median(ts)), 1), p90_ms=round(float(np.percentile(ts, 90)), 1))
    return res
LATENCY = {}
if DEVICE == "cuda": LATENCY["gpu"] = bench("cuda")
torch.set_num_threads(os.cpu_count() or 2)
LATENCY["cpu"] = dict(threads=torch.get_num_threads(), **bench("cpu", n=5 if TINY else 20))
model.to_device(DEVICE)
print(json.dumps(LATENCY, indent=1))
""", "latency")

# ---------------------------------------------------------------------------------------------------------------
md(r"""
## 10) Save the model (and export the test questions for the LLM comparison)
`/content/emotion_s1` contains:
* `backbone/` and `heads.pt`: the model;
* `emotion_s1_core.py`: the code you need to load and run it anywhere;
* `metrics.json`: everything measured above;
* `test_questions.jsonl`: the held-out test questions. Later we ask Claude, GPT and Jev these exact questions.
""")
code(r"""
import shutil
model.save(OUT_DIR, extra={"backbone_name": BACKBONE})
shutil.copy("emotion_s1_core.py", f"{OUT_DIR}/emotion_s1_core.py")
with open(f"{OUT_DIR}/test_questions.jsonl", "w") as f:
    for e in test_ex: f.write(json.dumps(e) + "\n")
summary = dict(model="Emotion-S1", backbone=BACKBONE, mode="tiny" if TINY else ("quick" if QUICK else "full"),
               commercial_safe=COMMERCIAL_SAFE, sources=sorted(RAW), max_len=MAX_LEN, n_train=N_TRAIN,
               n_train_questions=len(train_ex), steps=step, stopped_early=stopped_early, train_minutes=round(TRAIN_MINUTES, 1),
               temperatures=TEMPS, results=RESULTS, latency=LATENCY,
               gpu=torch.cuda.get_device_name(0) if DEVICE == "cuda" else "cpu")
json.dump(summary, open(f"{OUT_DIR}/metrics.json", "w"), indent=2)
print("saved to", OUT_DIR, sorted(os.listdir(OUT_DIR)))

reloaded = load_emotion_s1(OUT_DIR, device=DEVICE)
print("reload check:", ask(reloaded, DEMOS[0][1], {"frustrated": QS["frustrated"]}, context=DEMOS[0][2]))
del reloaded
""", "save")

md(r"""
## 11) Optional: copy the model to Google Drive
This copies everything to **My Drive / emotion_s1_model**, so it survives after the Colab session ends. Colab will ask you to allow Drive access.
""")
code(r"""
SAVE_TO_DRIVE = True
try:
    from google.colab import drive
    IN_COLAB = True
except ImportError:
    IN_COLAB = False
if SAVE_TO_DRIVE and IN_COLAB and not TINY:
    drive.mount("/content/drive")
    shutil.copytree(OUT_DIR, DRIVE_DIR, dirs_exist_ok=True)
    print("copied to", DRIVE_DIR, sorted(os.listdir(DRIVE_DIR)))
else:
    print("skipped Drive copy (not in Colab, or SAVE_TO_DRIVE is False)")
""", "drive")

md(r"""
## 12) Next: Emotion-S1 vs an LLM judge vs Jev
The comparison runs **outside** this notebook, on a normal CPU machine, with the saved model folder and `test_questions.jsonl`
(see `compare/README.md` in the project). It asks Claude, GPT and TypeSafe's Jev the **same** 600 held-out questions and
reports accuracy, calibration, latency and cost per 1,000 turns side by side. API keys live in environment variables, **never in
this notebook**.
""")

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata = {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
               "kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"}}
nbf.write(nb, str(HERE / "emotion_s1_t4.ipynb"))
print("wrote", len(cells), "cells:", sum(c.cell_type == "code" for c in cells), "code,", sum(c.cell_type == "markdown" for c in cells), "markdown")
