"""Build the out-of-domain Emotion-S1 comparison set from official TEST splits only.

Datasets (none of these were in the Emotion-S1 training six):
  * MELD text annotations (Friends TV dialogue) — official test CSV from declare-lab/MELD
  * EmoWOZ (task-oriented user–system dialogue) — official test IDs from Zenodo 6506504
  * dair-ai/emotion / CARER (Twitter messages) — official Hugging Face test split

None of the three has a labelled emotion-intensity score, so this set has choice and
yes/no questions only (300 of each). Score is omitted rather than invented.

Usage: python make_ood_questions.py  -> questions.json and mappings.json
"""
import csv, json, random, collections, argparse
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=7)
ap.add_argument("--per-type-per-src", type=int, default=100)
ap.add_argument("--emowoz-dir", default="/tmp/ood-data")
ap.add_argument("--meld-csv", default="data/meld_test_sent_emo.csv")
ap.add_argument("--out", default="questions.json")
ap.add_argument("--mappings", default="mappings.json")
a = ap.parse_args()

rng = random.Random(a.seed)
MAX_CTX = 4

CHOICE_PROMPTS = [
    "Which emotion is the user showing?",
    "What emotion does the user express?",
    "Pick the emotion that best matches the user's latest turn.",
    "How is the user feeling right now?",
    "What is the user mainly feeling in their latest message?",
    "Which of these best describes the user's emotional state?",
]

# ---- state formatting matches emotion_s1_core.format_state ----
def format_state(turn, context):
    lines = []
    ctx = list(context or [])[-MAX_CTX:]
    if ctx:
        lines.append("Earlier turns:")
        lines += [f"{role}: {text}" for role, text in ctx]
    lines += ["Latest user turn:", f"User: {turn}"]
    return "\n".join(lines)

def clean(s):
    return " ".join((s or "").replace("\r", " ").replace("\n", " ").split())

# ======================== loaders (TEST only) ========================
def load_meld(path):
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    by = collections.defaultdict(list)
    for r in rows:
        by[r["Dialogue_ID"]].append(r)
    for did in by:
        by[did].sort(key=lambda r: int(r["Utterance_ID"]))
    items = []
    for did, utts in by.items():
        for i, r in enumerate(utts):
            text = clean(r["Utterance"])
            if len(text) < 2:
                continue
            ctx = [(u["Speaker"], clean(u["Utterance"])) for u in utts[max(0, i - MAX_CTX):i]]
            items.append(dict(
                src="meld", uid=f"meld-{did}-{r['Utterance_ID']}",
                state=format_state(text, ctx),
                label=r["Emotion"].strip().lower(),
                sentiment=r["Sentiment"].strip().lower(),
            ))
    return items

EMOWOZ_LABEL = {
    0: "neutral", 1: "fearful", 2: "dissatisfied", 3: "apologetic",
    4: "abusive", 5: "excited", 6: "satisfied",
}

def load_emowoz(d):
    d = Path(d)
    split = json.load(open(d / "data-split.json"))
    mw = json.load(open(d / "emowoz-multiwoz.json"))
    dm = json.load(open(d / "emowoz-dialmage.json"))
    dialogues = {**mw, **dm}
    ids = list(set(split["test"]["multiwoz"] + split["test"]["dialmage"]))
    items = []
    for did in ids:
        log = dialogues[did]["log"]
        # User / system alternate; user turns have a non-empty emotion array.
        seq = []  # (role, text, label_or_none)
        for turn in log:
            text = clean(turn["text"])
            emo = turn.get("emotion") or []
            if emo and isinstance(emo[-1], dict) and "emotion" in emo[-1]:
                seq.append(("User", text, EMOWOZ_LABEL[int(emo[-1]["emotion"])]))
            else:
                seq.append(("Assistant", text, None))
        for i, (role, text, lab) in enumerate(seq):
            if lab is None or len(text) < 2:
                continue
            ctx = [(r, t) for r, t, _ in seq[max(0, i - MAX_CTX):i]]
            items.append(dict(
                src="ew", uid=f"ew-{did}-t{i}",
                state=format_state(text, ctx), label=lab,
            ))
    return items

def load_carer():
    from datasets import load_dataset
    ds = load_dataset("dair-ai/emotion", "split")["test"]
    names = ds.features["label"].names
    items = []
    for i, row in enumerate(ds):
        text = clean(row["text"])
        if len(text) < 2:
            continue
        items.append(dict(
            src="carer", uid=f"carer-test-{i}",
            state=format_state(text, []),
            label=names[int(row["label"])],
        ))
    return items

# ======================== mappings ========================
# Official choice inventories (dataset names, not remapped onto Ekman).
OPTIONS = {
    "meld":  ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"],
    "ew":    ["abusive", "apologetic", "dissatisfied", "excited", "fearful", "neutral", "satisfied"],
    "carer": ["anger", "fear", "joy", "love", "sadness", "surprise"],
}

# Derived yes/no families. These are documented mappings, not original labels.
# frustrated / support follow the same idea as the in-domain derived labels.
YESNO = {
    "meld": {
        "frustrated": dict(positive={"anger", "disgust"},
                           questions=["Is the user annoyed or frustrated?",
                                      "Is the user frustrated?",
                                      "Does the user sound irritated or fed up?",
                                      "Is the user getting frustrated?"]),
        "support":    dict(positive={"sadness", "fear"},
                           questions=["Does the user need emotional support?",
                                      "Should the assistant respond with empathy and support here?",
                                      "Is the user in distress and in need of support?",
                                      "Does the user seem to be struggling emotionally?"]),
        "positive":   dict(positive={"joy"},
                           questions=["Is the user in a positive mood?",
                                      "Does the user sound happy or positive?"]),
        "neutral":    dict(positive={"neutral"},
                           questions=["Is the user calm and emotionally neutral?",
                                      "Is the user's message emotionally neutral?"]),
    },
    "ew": {
        # Dissatisfied = negative emotion elicited by the system; abusive = the impolite form of that.
        "frustrated": dict(positive={"dissatisfied", "abusive"},
                           questions=["Is the user annoyed or frustrated?",
                                      "Is the user frustrated?",
                                      "Does the user sound irritated or fed up?",
                                      "Is the user getting frustrated?"]),
        # Fearful in EmoWOZ also covers sad/disappointed, elicited by events outside the system's control.
        "support":    dict(positive={"fearful"},
                           questions=["Does the user need emotional support?",
                                      "Should the assistant respond with empathy and support here?",
                                      "Is the user in distress and in need of support?",
                                      "Does the user seem to be struggling emotionally?"]),
        "positive":   dict(positive={"satisfied", "excited"},
                           questions=["Is the user in a positive mood?",
                                      "Does the user sound happy or positive?"]),
        "neutral":    dict(positive={"neutral"},
                           questions=["Is the user calm and emotionally neutral?",
                                      "Is the user's message emotionally neutral?"]),
    },
    "carer": {
        "frustrated": dict(positive={"anger"},
                           questions=["Is the user annoyed or frustrated?",
                                      "Is the user frustrated?",
                                      "Does the user sound irritated or fed up?",
                                      "Is the user getting frustrated?"]),
        "support":    dict(positive={"sadness", "fear"},
                           questions=["Does the user need emotional support?",
                                      "Should the assistant respond with empathy and support here?",
                                      "Is the user in distress and in need of support?",
                                      "Does the user seem to be struggling emotionally?"]),
        "positive":   dict(positive={"joy", "love"},
                           questions=["Is the user in a positive mood?",
                                      "Does the user sound happy or positive?"]),
    },
}

EMOTION_Q = {
    # "{label}" is filled in
    "show": "Does the user's latest message show {lab}?",
    "main": "Is {lab} the main emotion in the user's message?",
    "expressing": "Is the user expressing {lab}?",
    "describe": "Would you describe the user's emotion as {lab}?",
}

def pick_stratified(items, n, rng):
    """Take up to n items, spreading across labels; leftover goes to remaining items."""
    by = collections.defaultdict(list)
    for it in items:
        by[it["label"]].append(it)
    for lab in by:
        rng.shuffle(by[lab])
    labels = sorted(by)
    out, used = [], set()
    # First pass: round-robin so rare classes appear
    i = 0
    while len(out) < n and any(by[lab] for lab in labels):
        lab = labels[i % len(labels)]
        i += 1
        if not by[lab]:
            continue
        it = by[lab].pop()
        out.append(it); used.add(it["uid"])
    return out

def make_choice(items, src, n, rng, start_id):
    chosen = pick_stratified(items, n, rng)
    qs = []
    for i, it in enumerate(chosen):
        qs.append(dict(
            id=f"ood-{src}-choice-{start_id + i:04d}",
            type="choice", src=src, fam="emotion",
            state=it["state"], uid=it["uid"],
            question=rng.choice(CHOICE_PROMPTS),
            options=list(OPTIONS[src]),
            gold=it["label"],
        ))
    return qs, {it["uid"] for it in chosen}

def make_noul(items, src, n, rng, start_id, avoid_uids=None):
    """Mix of derived families + 'is this emotion X?' questions, aiming for a balanced yes/no split."""
    pool = [it for it in items if not avoid_uids or it["uid"] not in avoid_uids]
    if len(pool) < n:
        pool = list(items)  # allow overlap with choice turns if needed
    rng.shuffle(pool)
    # Target mix: half derived families, half emotion-X; ~50% yes overall
    n_derived = n // 2
    n_emotion = n - n_derived
    families = list(YESNO[src].keys())
    per_fam = collections.Counter()
    base, rem = divmod(n_derived, len(families))
    for i, f in enumerate(families):
        per_fam[f] = base + (1 if i < rem else 0)

    qs, used = [], set()
    def add(it, question, gold, fam):
        qs.append(dict(
            id=f"ood-{src}-noul-{start_id + len(qs):04d}",
            type="noul", src=src, fam=fam,
            state=it["state"], uid=it["uid"],
            question=question, gold=bool(gold),
        ))
        used.add(it["uid"])

    # Derived families, balanced yes/no within each
    for fam, k in per_fam.items():
        spec = YESNO[src][fam]
        pos = [it for it in pool if it["label"] in spec["positive"] and it["uid"] not in used]
        neg = [it for it in pool if it["label"] not in spec["positive"] and it["uid"] not in used]
        rng.shuffle(pos); rng.shuffle(neg)
        n_yes = k // 2
        n_no = k - n_yes
        # If not enough positives (e.g. EmoWOZ fearful only 18), take what we have and fill with no
        take_yes = pos[:n_yes]
        take_no = neg[:n_no + (n_yes - len(take_yes))]
        for it in take_yes:
            add(it, rng.choice(spec["questions"]), True, fam)
        for it in take_no:
            add(it, rng.choice(spec["questions"]), False, fam)

    # Emotion-X yes/no, half true-label (yes) and half a different official option (no)
    remain = [it for it in pool if it["uid"] not in used]
    rng.shuffle(remain)
    n_yes = n_emotion // 2
    templates = list(EMOTION_Q.values())
    yes_items = remain[:n_yes]
    no_items = remain[n_yes:n_yes + (n_emotion - n_yes)]
    for it in yes_items:
        lab = it["label"]
        add(it, rng.choice(templates).format(lab=lab), True, "emotion")
    opts = OPTIONS[src]
    for it in no_items:
        others = [o for o in opts if o != it["label"]]
        lab = rng.choice(others)
        add(it, rng.choice(templates).format(lab=lab), False, "emotion")

    # If we still need more (rare), fill from leftover with emotion-X
    remain = [it for it in pool if it["uid"] not in used]
    rng.shuffle(remain)
    while len(qs) < n and remain:
        it = remain.pop()
        if rng.random() < 0.5:
            add(it, rng.choice(templates).format(lab=it["label"]), True, "emotion")
        else:
            others = [o for o in OPTIONS[src] if o != it["label"]]
            add(it, rng.choice(templates).format(lab=rng.choice(others)), False, "emotion")
    return qs[:n]

def main():
    print("loading MELD test CSV...", flush=True)
    meld = load_meld(a.meld_csv)
    print("  ", len(meld), collections.Counter(x["label"] for x in meld))
    print("loading EmoWOZ official test...", flush=True)
    ew = load_emowoz(a.emowoz_dir)
    print("  ", len(ew), collections.Counter(x["label"] for x in ew))
    print("loading CARER / dair-ai/emotion test...", flush=True)
    carer = load_carer()
    print("  ", len(carer), collections.Counter(x["label"] for x in carer))

    sources = [("meld", meld), ("ew", ew), ("carer", carer)]
    qs = []
    for src, items in sources:
        ch, used = make_choice(items, src, a.per_type_per_src, rng, 1)
        nu = make_noul(items, src, a.per_type_per_src, rng, 1, avoid_uids=used)
        print(src, "choice", len(ch), collections.Counter(q["gold"] for q in ch))
        print(src, "noul", len(nu),
              "yes", sum(q["gold"] for q in nu), "no", sum(not q["gold"] for q in nu),
              collections.Counter(q["fam"] for q in nu))
        qs.extend(ch); qs.extend(nu)

    # Stable order: type then src, like the in-domain set (choice then noul). score omitted.
    order = {"choice": 0, "noul": 1}
    qs.sort(key=lambda q: (order[q["type"]], q["src"], q["id"]))
    json.dump(qs, open(a.out, "w"), indent=1)
    mappings = {
        "note": "Official TEST splits only. No intensity/score labels exist on these datasets, so score questions are omitted rather than proxied.",
        "datasets": {
            "meld": {
                "name": "MELD (text annotations)",
                "domain": "Multi-party scripted TV dialogue from Friends",
                "source": "https://github.com/declare-lab/MELD/blob/master/data/MELD/test_sent_emo.csv",
                "paper": "https://arxiv.org/pdf/1810.02508.pdf",
                "split": "official test (2,610 utterances, 280 dialogues)",
                "licence": "Repository is GPL-3.0. The transcripts are from the copyrighted TV series Friends; used here for research/evaluation only, text annotations only (no audio or video).",
                "choice_options": OPTIONS["meld"],
                "choice_mapping": "Gold is the official Emotion field, lowercased. Labels are not renamed (MELD uses joy, not happiness).",
                "noul_mapping": {
                    "emotion": "Yes iff the official Emotion equals the named option.",
                    "frustrated": "Derived: yes iff Emotion is anger or disgust.",
                    "support": "Derived: yes iff Emotion is sadness or fear.",
                    "positive": "Yes iff Emotion is joy.",
                    "neutral": "Yes iff Emotion is neutral.",
                },
            },
            "ew": {
                "name": "EmoWOZ",
                "domain": "Task-oriented user–system dialogue (MultiWOZ + DialMAGE human–machine)",
                "source": "https://zenodo.org/records/6506504",
                "paper": "https://aclanthology.org/2022.lrec-1.436/",
                "split": "official test (1,000 MultiWOZ + 100 DialMAGE dialogues; 8,634 labelled user turns)",
                "licence": "CC BY-NC 4.0 (non-commercial, attribution required). Authors also say research use only.",
                "choice_options": OPTIONS["ew"],
                "choice_mapping": "Gold is the official majority-vote emotion integer mapped to the paper's tokens: 0 neutral, 1 fearful (also sad/disappointed), 2 dissatisfied (also disliking), 3 apologetic, 4 abusive, 5 excited (also happy/anticipating), 6 satisfied (also liking). Options are these tokens, not remapped onto Ekman.",
                "noul_mapping": {
                    "emotion": "Yes iff the official token equals the named option.",
                    "frustrated": "Derived: yes iff the token is dissatisfied or abusive (negative emotion elicited by the system, polite or impolite).",
                    "support": "Derived: yes iff the token is fearful (sad/disappointed/fearful elicited by events outside the system's control). Dissatisfied is not treated as needing support.",
                    "positive": "Yes iff the token is satisfied or excited.",
                    "neutral": "Yes iff the token is neutral.",
                },
            },
            "carer": {
                "name": "CARER / dair-ai/emotion",
                "domain": "Single-turn English Twitter messages (not multi-turn chat)",
                "source": "https://huggingface.co/datasets/dair-ai/emotion",
                "homepage": "https://github.com/dair-ai/emotion_dataset",
                "paper": "https://aclanthology.org/D18-1404/",
                "split": "official test split of the 'split' config (2,000 tweets)",
                "licence": "Educational and research purposes only (dataset card / GitHub).",
                "choice_options": OPTIONS["carer"],
                "choice_mapping": "Gold is the official six-way label: sadness, joy, love, anger, fear, surprise. There is no neutral class.",
                "noul_mapping": {
                    "emotion": "Yes iff the official label equals the named option.",
                    "frustrated": "Derived: yes iff the label is anger.",
                    "support": "Derived: yes iff the label is sadness or fear.",
                    "positive": "Yes iff the label is joy or love.",
                },
            },
        },
        "score": "Skipped. None of MELD, EmoWOZ or CARER has an official emotion-intensity score. No proxy was invented.",
        "seed": a.seed,
        "counts": {f"{k[0]}/{k[1]}": v for k,v in collections.Counter((q["type"], q["src"]) for q in qs).items()},
    }
    json.dump(mappings, open(a.mappings, "w"), indent=2)
    print("wrote", a.out, len(qs), "questions")
    print("types", collections.Counter(q["type"] for q in qs))
    print("gold yes rate noul", sum(q["gold"] for q in qs if q["type"]=="noul") / sum(1 for q in qs if q["type"]=="noul"))

if __name__ == "__main__":
    main()
