"""Shared helpers for the Emotion-S1 vs LLM-judge vs Jev comparison. No API keys are stored anywhere in this folder:
scripts read OPENAI_API_KEY / ANTHROPIC_API_KEY / TYPESAFE_API_KEY from the environment at run time."""
import json, re, random, collections, os, tempfile

SYSTEM = ("You are an emotion-analysis judge inside a chat application. You read a chat (optional earlier turns, then the "
          "user's latest turn) and answer exactly one question about the user's latest turn. Reply with JSON only, no prose.")

def question_text(q):
    if q["type"] == "noul":
        return (f"Question: {q['question']}\n"
                'Reply as {"answer": "yes" or "no", "confidence": <0-1, the probability your answer is correct>}')
    if q["type"] == "choice":
        return (f"Question: {q['question']}\nOptions: {json.dumps(q['options'])}\n"
                'Reply as {"answer": "<exactly one option, copied as written>", "confidence": <0-1>}')
    lo, hi = q["scale"]
    return (f"Scoring task: {q['rubric']}\nUse a whole number from {lo} to {hi}.\n"
            'Reply as {"score": <number>, "confidence": <0-1, the probability your score is exactly right>}')

def user_prompt(q):
    return f"Chat:\n{q['state']}\n\n{question_text(q)}"

def parse(q, text):
    """Returns a normalised answer dict, or None if the reply could not be parsed."""
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m: return None
    try: d = json.loads(m.group(0))
    except Exception: return None
    conf = d.get("confidence")
    try: conf = min(1.0, max(0.0, float(conf)))
    except Exception: conf = None
    if q["type"] == "noul":
        a = str(d.get("answer", "")).strip().lower()
        if a not in ("yes", "no"): return None
        yes = a == "yes"; c = conf if conf is not None else 1.0
        return dict(answer=yes, p_yes=c if yes else 1 - c, confidence=c)
    if q["type"] == "choice":
        a = str(d.get("answer", "")).strip()
        hit = [o for o in q["options"] if o.lower() == a.lower()]
        if not hit: return None
        return dict(answer=hit[0], confidence=conf if conf is not None else 1.0)
    try: s = float(d.get("score"))
    except Exception: return None
    return dict(score=s, confidence=conf if conf is not None else 1.0)

def load_questions(path="questions.json"):
    return json.load(open(path))

def pick_repeats(questions, n, seed=7):
    """Stratified sample across question types. Taking Q[:n] would only cover the first type,
    because make_questions.py writes choice, then yes/no, then score."""
    if not n:
        return []
    by = collections.defaultdict(list)
    for q in questions:
        by[q["type"]].append(q)
    rng = random.Random(seed)
    types = [t for t in ("choice", "noul", "score") if by[t]]
    if not types:
        return []
    for t in types:
        rng.shuffle(by[t])
    # even split, leftover goes to earlier types
    base, rem = divmod(n, len(types))
    alloc = {t: min(len(by[t]), base + (1 if i < rem else 0)) for i, t in enumerate(types)}
    leftover = n - sum(alloc.values())
    for t in types:
        if leftover <= 0:
            break
        extra = min(leftover, len(by[t]) - alloc[t])
        alloc[t] += extra
        leftover -= extra
    out = []
    for t in types:
        out.extend(by[t][:alloc[t]])
    return out

def atomic_write_json(path, obj, indent=1):
    """Write JSON atomically so a crash mid-save cannot truncate a paid-run file."""
    directory = os.path.dirname(path) or "."
    fd, tmp = tempfile.mkstemp(prefix=".tmp-", suffix=".json", dir=directory)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(obj, f, indent=indent)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
