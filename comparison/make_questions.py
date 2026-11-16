"""Pick the comparison set from the notebook's held-out test questions.
Usage: python make_questions.py /path/to/emotion_s1_model [--per-type 200] -> questions.json
Stratified: --per-type questions of each type (choice / noul / score), split evenly across the datasets of that type."""
import json, sys, random, argparse, collections
ap = argparse.ArgumentParser(); ap.add_argument("weights"); ap.add_argument("--per-type", type=int, default=200)
ap.add_argument("--seed", type=int, default=7); a = ap.parse_args()
rows = [json.loads(l) for l in open(f"{a.weights}/test_questions.jsonl")]
rng = random.Random(a.seed); out = []
for t in ["choice", "noul", "score"]:
    by = collections.defaultdict(list)
    for r in rows:
        if r["task"] == t: by[r["src"]].append(r)
    srcs = sorted(by); per = a.per_type // max(1, len(srcs))
    for s in srcs:
        rng.shuffle(by[s]); out += by[s][:per]
qs = []
for r in out:
    q = dict(id=r["id"], type=r["task"], src=r["src"], fam=r.get("fam"), state=r["state"])
    if r["task"] == "noul":   q.update(question=r["question"], gold=bool(r["label"]))
    if r["task"] == "choice": q.update(question=r["question"], options=r["options"], gold=r["options"][r["label"]])
    if r["task"] == "score":  q.update(rubric=r["rubric"], scale=list(r["scale"]), gold=r["value"])
    qs.append(q)
json.dump(qs, open("questions.json", "w"), indent=1)
print(len(qs), "questions", dict(collections.Counter((q["type"], q["src"]) for q in qs)))
