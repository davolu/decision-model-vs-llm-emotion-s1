"""Score every system on the same questions and write results.json + results.md.
Usage: python score.py [--price "llm_openai_<model>_raw.json=IN,OUT" ...] [--price-source "file=URL"] [--out-dir results]
IN/OUT = USD per 1M input/output tokens, copied from the provider's pricing page ON THE RUN DATE (prices change; none are hard-coded).
A "turn" = 3 questions about one user message (one choice, one score, one yes/no), each asked as a separate call."""
import json, glob, argparse, os, collections, numpy as np
from common import load_questions
ap = argparse.ArgumentParser()
ap.add_argument("--price", action="append", default=[])
ap.add_argument("--price-source", action="append", default=[], help='e.g. --price-source "llm_openai_gpt-6.1-sol_raw.json=https://..."')
ap.add_argument("--out-dir", default="results")
a = ap.parse_args()
PRICES = {k: tuple(map(float, v.split(","))) for k, v in (p.split("=", 1) for p in a.price)}
SOURCES = {k: v for k, v in (p.split("=", 1) for p in a.price_source)}
Q = load_questions(); QD = {q["id"]: q for q in Q}
os.makedirs(a.out_dir, exist_ok=True)

def ece(conf, correct, n_bins=15):
    conf, correct = np.asarray(conf, float), np.asarray(correct, float); e = 0.0
    for lo, hi in zip(np.linspace(0, 1, n_bins + 1)[:-1], np.linspace(0, 1, n_bins + 1)[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any(): e += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return round(float(e), 4)

def same_answer(a1, a2, qtype):
    if a1 is None or a2 is None:
        return False
    if qtype == "score":
        try:
            return abs(float(a1.get("score")) - float(a2.get("score"))) < 1e-6
        except Exception:
            return False
    return a1.get("answer") == a2.get("answer")

def evaluate(name, answers, repeats=None, price=None, price_source=None, extra=None):
    r = {"system": name}; lat = []; toks = {"choice": [], "noul": [], "score": []}
    statuses = collections.Counter()
    for rec in answers.values():
        statuses[str(rec.get("status", "missing"))] += 1
    r["n_answers"] = len(answers)
    r["status_counts"] = dict(statuses)
    served = sorted({rec.get("served_model") for rec in answers.values() if rec.get("served_model")})
    if served:
        r["served_model"] = served
    for t in ["choice", "noul", "score"]:
        qs = [q for q in Q if q["type"] == t]; ok = [q for q in qs if answers.get(q["id"], {}).get("ok")]
        lat += [answers[q["id"]]["latency_ms"] for q in qs if q["id"] in answers]
        toks[t] = [(answers[q["id"]].get("input_tokens", 0), answers[q["id"]].get("output_tokens", 0)) for q in qs if q["id"] in answers]
        d = dict(n=len(qs), n_ok=len(ok), parse_fail_rate=round(1 - len(ok) / max(1, len(qs)), 4))
        if t == "score":
            ev = np.array([answers[q["id"]]["answer"]["score"] for q in ok]); g = np.array([q["gold"] for q in ok])
            # top_level may be a discrete bin (Emotion-S1) or a soft expected value (Jev); round to a 1-5 level for ECE.
            top = np.array([
                round(answers[q["id"]]["answer"].get("top_level", answers[q["id"]]["answer"]["score"]))
                for q in ok
            ])
            conf = [answers[q["id"]]["answer"]["confidence"] for q in ok]
            d.update(mae=round(float(np.abs(ev - g).mean()), 4) if len(ok) else None,
                     within_1=round(float((np.abs(ev - g) <= 1).mean()), 4) if len(ok) else None,
                     pearson=round(float(np.corrcoef(ev, g)[0, 1]), 4) if len(ok) > 2 and ev.std() > 0 and g.std() > 0 else None,
                     ece=ece(conf, top == np.clip(np.round(g), 1, 5)) if len(ok) else None)
        else:
            corr = [answers[q["id"]]["answer"]["answer"] == q["gold"] for q in ok]
            conf = [answers[q["id"]]["answer"]["confidence"] for q in ok]
            d.update(accuracy=round(float(np.mean(corr)), 4) if ok else None, ece=ece(conf, corr) if ok else None)
            d["accuracy_counting_failures_wrong"] = round(sum(corr) / max(1, len(qs)), 4)
        r[t] = d
    if lat:
        r["latency_ms"] = dict(median=round(float(np.median(lat)), 1),
                               p90=round(float(np.percentile(lat, 90)), 1),
                               p95=round(float(np.percentile(lat, 95)), 1),
                               n=len(lat))
    else:
        r["latency_ms"] = None
    mean_tok = {t: tuple(np.mean(v, axis=0).round(1)) if v else (0, 0) for t, v in toks.items()}
    r["mean_tokens_per_question"] = {t: dict(input=float(v[0]), output=float(v[1])) for t, v in mean_tok.items()}
    if price:
        per_turn = sum(v[0] * price[0] / 1e6 + v[1] * price[1] / 1e6 for v in mean_tok.values())
        r["usd_per_1000_turns"] = round(per_turn * 1000, 4)
        r["price_usd_per_1m_tokens"] = dict(input=price[0], output=price[1])
    if price_source:
        r["price_source"] = price_source
    if repeats:
        same = []
        by_type = collections.defaultdict(list)
        for qid, rep in repeats.items():
            a1, a2 = answers.get(qid, {}).get("answer"), rep.get("answer")
            q = QD.get(qid)
            if a1 and a2 and q:
                hit = same_answer(a1, a2, q["type"])
                same.append(hit)
                by_type[q["type"]].append(hit)
        r["self_disagreement_rate"] = round(1 - float(np.mean(same)), 4) if same else None
        r["self_disagreement_n"] = len(same)
        r["self_disagreement_by_type"] = {t: round(1 - float(np.mean(v)), 4) for t, v in by_type.items() if v}
    if extra:
        r.update(extra)
    return r

results = []
try:
    s1 = json.load(open("s1_raw.json")); res = evaluate("Emotion-S1 (CPU)", s1)
    res["usd_per_1000_turns"] = 0.0; res["self_disagreement_rate"] = 0.0; res["note"] = "runs locally; deterministic"
    results.append(res)
except FileNotFoundError:
    print("no s1_raw.json yet")
for fn in sorted(glob.glob("llm_*_raw.json")) + sorted(glob.glob("jev_raw.json")):
    d = json.load(open(fn))
    results.append(evaluate(f"{d['provider']}:{d['model']}", d["answers"], d.get("repeats"),
                            PRICES.get(fn), SOURCES.get(fn)))

out_json = os.path.join(a.out_dir, "results.json")
out_md = os.path.join(a.out_dir, "results.md")
json.dump(results, open(out_json, "w"), indent=1)

def fmt(x, nd=4):
    if x is None: return "—"
    return x if isinstance(x, str) else x

hdr = ("| System | Choice acc | Yes/No acc | Score MAE (within 1) | ECE (c / y-n / s) | "
       "Median ms | p95 ms | $ per 1,000 turns | Self-disagreement | Parse-fail |\n"
       "|---|---|---|---|---|---|---|---|---|---|\n")
rows = ""
for r in results:
    pf = "/".join(str(r[t]["parse_fail_rate"]) for t in ("choice", "noul", "score"))
    score_cell = "—" if r["score"].get("mae") is None else f"{r['score']['mae']} ({r['score'].get('within_1')})"
    cost = r.get("usd_per_1000_turns", "set --price")
    rows += (f"| {r['system']} | {r['choice'].get('accuracy')} | {r['noul'].get('accuracy')} | {score_cell} | "
             f"{r['choice'].get('ece')} / {r['noul'].get('ece')} / {r['score'].get('ece')} | "
             f"{(r['latency_ms'] or {}).get('median')} | {(r['latency_ms'] or {}).get('p95')} | "
             f"{cost} | {r.get('self_disagreement_rate')} | {pf} |\n")
notes = ["", "## Price sources and model IDs", ""]
for r in results:
    notes.append(f"- **{r['system']}**")
    if r.get("served_model"):
        notes.append(f"  - served model: `{', '.join(r['served_model'])}`")
    if r.get("price_usd_per_1m_tokens"):
        p = r["price_usd_per_1m_tokens"]
        notes.append(f"  - list price used: ${p['input']}/1M input, ${p['output']}/1M output")
    if r.get("price_source"):
        notes.append(f"  - source: {r['price_source']}")
    if r.get("mean_tokens_per_question"):
        notes.append(f"  - mean tokens per question: {json.dumps(r['mean_tokens_per_question'])}")
    if r.get("note"):
        notes.append(f"  - {r['note']}")
    notes.append("")
open(out_md, "w").write(hdr + rows + "\n".join(notes))
print(hdr + rows)
print("wrote", out_json, "and", out_md)
