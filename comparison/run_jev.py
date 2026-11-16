"""TypeSafe Jev on the same questions (same request pattern as mini-jev/compare/run_jev.py). DRY RUN by default.
Usage: python run_jev.py [--live] [--limit N] [--repeat 100]   (reads TYPESAFE_API_KEY from the environment) -> jev_raw.json"""
import json, os, sys, time, argparse, threading, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from common import load_questions, pick_repeats, atomic_write_json

URL, MODEL = "https://api.typesafe.ai/v1/systemone", "jev-latest"
ap = argparse.ArgumentParser(); ap.add_argument("--live", action="store_true"); ap.add_argument("--limit", type=int)
ap.add_argument("--repeat", type=int, default=0)
ap.add_argument("--sleep", type=float, default=0.0)
ap.add_argument("--workers", type=int, default=6)
ap.add_argument("--resume", action="store_true", default=True)
ap.add_argument("--no-resume", dest="resume", action="store_false")
a = ap.parse_args()
Q = load_questions()[: a.limit]
repeats_Q = pick_repeats(Q, a.repeat)
LEVELS = ["1 (lowest end of the rubric)", "2", "3", "4", "5 (highest end of the rubric)"]

def jev_q(q):
    if q["type"] == "noul":   return {"type": "noul", "instructions": q["question"]}
    if q["type"] == "choice": return {"type": "choice", "instructions": q["question"], "criteria": {o: None for o in q["options"]}}
    return {"type": "score", "instructions": q["rubric"], "criteria": LEVELS}

def call(body):
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), method="POST",
                                 headers={"Authorization": f"Bearer {os.environ['TYPESAFE_API_KEY']}", "Content-Type": "application/json"})
    t = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, json.loads(r.read()), (time.perf_counter() - t) * 1000
    except urllib.error.HTTPError as e:
        raw = e.read().decode()[:800]
        try:
            parsed = json.loads(raw)
        except Exception:
            parsed = raw
        return e.code, parsed, (time.perf_counter() - t) * 1000
    except Exception as e:
        return 0, f"{type(e).__name__}: {e}"[:300], (time.perf_counter() - t) * 1000

def norm(q, r):
    try: x = r["answers"]["q"]
    except Exception: return None
    if q["type"] == "noul":
        p = float(x["noul"]); return dict(answer=p >= 0.5, p_yes=p, confidence=max(p, 1 - p))
    if q["type"] == "choice":
        return dict(answer=x["choice"], confidence=float(x.get("confidence", 1.0)), probabilities=x.get("probabilities"))
    # Jev score levels are 0-based; shift +1 to the 1-5 rubric, same as mini-jev/compare.
    P = {int(k): float(v) for k, v in x.get("probabilities", {}).items()}
    ev = sum((k + 1) * v for k, v in P.items()) if P else float(x["score"]) + 1
    return dict(score=ev, top_level=float(x["score"]) + 1, confidence=float(x.get("confidence", 1.0)))

def run_one(q):
    state = q["state"] if q.get("state") not in (None, "") else "(no context)"
    body = {"model": MODEL, "state": state, "questions": {"q": jev_q(q)}}
    last = None
    for attempt in range(6):
        st, r, ms = call(body)
        last = (st, r, ms)
        if st == 429 or st >= 500 or st == 0:
            time.sleep(2 ** attempt * 2)
            continue
        break
    st, r, ms = last
    ans = norm(q, r) if st == 200 else None
    u = r.get("usage", {}) if isinstance(r, dict) else {}
    served = r.get("model") if isinstance(r, dict) else None
    raw = None if st == 200 else (r if isinstance(r, str) else json.dumps(r)[:400])
    return dict(status=st, latency_ms=round(ms, 1), ok=ans is not None, answer=ans, raw=raw,
                input_tokens=u.get("input_tokens", 0), output_tokens=u.get("output_tokens", 0), served_model=served)

if not a.live:
    print(f"DRY RUN (Jev): {len(Q)} questions would be sent (+{len(repeats_Q)} repeats). No calls made. Example body:")
    print(json.dumps({"model": MODEL, "state": Q[0]["state"], "questions": {"q": jev_q(Q[0])}}, indent=1)); sys.exit(0)
if not os.environ.get("TYPESAFE_API_KEY"):
    sys.exit("TYPESAFE_API_KEY is not set")

fn = "jev_raw.json"
out = {"provider": "typesafe", "model": MODEL, "answers": {}, "repeats": {}}
if a.resume and os.path.exists(fn):
    try:
        prev = json.load(open(fn))
        if prev.get("provider") == "typesafe":
            out["answers"] = prev.get("answers") or {}
            out["repeats"] = prev.get("repeats") or {}
            print(f"resuming {fn}: {len(out['answers'])} answers, {len(out['repeats'])} repeats")
    except Exception as e:
        print("could not resume, starting fresh:", type(e).__name__)

lock = threading.Lock()
done = [0]
print_lock = threading.Lock()

def work(kind, q):
    rec = run_one(q)
    if a.sleep:
        time.sleep(a.sleep)
    with lock:
        out[kind][q["id"]] = rec
        done[0] += 1
        n = done[0]
        if n % 10 == 0 or rec["status"] != 200 or not rec["ok"]:
            atomic_write_json(fn, out, indent=2)
    with print_lock:
        print(n, kind, q["id"], rec["status"], rec["latency_ms"], rec["answer"], flush=True)
    return rec

todo_answers = [q for q in Q if q["id"] not in out["answers"] or not out["answers"][q["id"]].get("ok")]
todo_repeats = [q for q in repeats_Q if q["id"] not in out["repeats"] or not out["repeats"][q["id"]].get("ok")]
print(f"live Jev {MODEL}: {len(todo_answers)} answers + {len(todo_repeats)} repeats, workers={a.workers}")

with ThreadPoolExecutor(max_workers=max(1, a.workers)) as ex:
    futs = [ex.submit(work, "answers", q) for q in todo_answers]
    futs += [ex.submit(work, "repeats", q) for q in todo_repeats]
    for f in as_completed(futs):
        f.result()

atomic_write_json(fn, out, indent=2)
print("wrote", fn)
