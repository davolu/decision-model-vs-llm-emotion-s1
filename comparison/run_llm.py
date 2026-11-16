"""LLM-as-judge baseline on the same questions. DRY RUN by default (prints the prompt, makes no paid calls).
Usage:
  python run_llm.py --provider openai    --model <model-name> [--live] [--limit N] [--repeat 100]
  python run_llm.py --provider anthropic --model <model-name> [--live] [--limit N] [--repeat 100]
Keys come from OPENAI_API_KEY / ANTHROPIC_API_KEY in the environment; nothing is written to disk except answers.
--repeat N asks N questions a second time (stratified by type) to measure how often the LLM changes its own answer.
Output: llm_<provider>_<model>_raw.json"""
import json, os, sys, time, argparse, threading, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from common import SYSTEM, user_prompt, parse, load_questions, pick_repeats, atomic_write_json

ap = argparse.ArgumentParser()
ap.add_argument("--provider", choices=["openai", "anthropic"], required=True); ap.add_argument("--model", required=True)
ap.add_argument("--live", action="store_true"); ap.add_argument("--limit", type=int); ap.add_argument("--repeat", type=int, default=0)
ap.add_argument("--sleep", type=float, default=0.0)
ap.add_argument("--workers", type=int, default=8)
ap.add_argument("--max-tokens", type=int, default=4096)
ap.add_argument("--effort", default="low", help="Anthropic effort / OpenAI reasoning effort. Empty string to omit.")
ap.add_argument("--resume", action="store_true", default=True)
ap.add_argument("--no-resume", dest="resume", action="store_false")
a = ap.parse_args()
Q = load_questions()[: a.limit]
repeats_Q = pick_repeats(Q, a.repeat)

def post(url, headers, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json", **headers})
    t = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
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

def err_text(r):
    return r if isinstance(r, str) else json.dumps(r)

def call(q, use_temp=True, use_effort=True, use_json=True):
    if a.provider == "openai":
        body = {"model": a.model,
                "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_prompt(q)}],
                "max_completion_tokens": a.max_tokens}
        if use_json:
            body["response_format"] = {"type": "json_object"}
        if use_temp:
            body["temperature"] = 0
        if use_effort and a.effort:
            body["reasoning_effort"] = a.effort
        st, r, ms = post("https://api.openai.com/v1/chat/completions",
                         {"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"}, body)
        if st != 200:
            return st, r, ms, None, 0, 0, None
        u = r.get("usage", {})
        return st, r, ms, r["choices"][0]["message"].get("content"), u.get("prompt_tokens", 0), u.get("completion_tokens", 0), r.get("model")
    body = {"model": a.model, "system": SYSTEM, "max_tokens": a.max_tokens,
            "messages": [{"role": "user", "content": user_prompt(q)}]}
    if use_temp:
        body["temperature"] = 0
    if use_effort and a.effort:
        # Sonnet 5.5+ takes effort under output_config, not as a top-level field.
        body["output_config"] = {"effort": a.effort}
    st, r, ms = post("https://api.anthropic.com/v1/messages",
                     {"x-api-key": os.environ["ANTHROPIC_API_KEY"], "anthropic-version": "2023-06-01"}, body)
    if st != 200:
        return st, r, ms, None, 0, 0, None
    u = r.get("usage", {})
    text = "".join(b.get("text", "") for b in r.get("content", []) if b.get("type") == "text")
    return st, r, ms, text, u.get("input_tokens", 0), u.get("output_tokens", 0), r.get("model")

# Flags that some models reject; once rejected, stay off for the rest of the run.
flags_lock = threading.Lock()
FLAGS = dict(use_temp=False, use_effort=True, use_json=True)  # gpt-6.1-sol and claude-sonnet-5-5 reject temperature

def call_adaptive(q):
    with flags_lock:
        use_temp, use_effort, use_json = FLAGS["use_temp"], FLAGS["use_effort"], FLAGS["use_json"]
    last = None
    for attempt in range(6):
        st, r, ms, text, tin, tout, served = call(q, use_temp, use_effort, use_json)
        last = (st, r, ms, text, tin, tout, served)
        msg = err_text(r).lower()
        if st == 400:
            changed = False
            with flags_lock:
                if use_temp and "temperature" in msg:
                    FLAGS["use_temp"] = False; use_temp = False; changed = True
                elif use_effort and ("reasoning_effort" in msg or ("effort" in msg and "reasoning" in msg) or "effort" in msg):
                    FLAGS["use_effort"] = False; use_effort = False; changed = True
                elif use_json and ("response_format" in msg or "json_object" in msg):
                    FLAGS["use_json"] = False; use_json = False; changed = True
            if changed:
                continue
        if st == 429 or st >= 500 or st == 0:
            # honour retry-after when present
            wait = 2 ** attempt * 2
            if isinstance(r, dict):
                # some APIs put retry hints in the body; default backoff is enough
                pass
            time.sleep(wait)
            continue
        break
    return last

def run_one(q):
    st, r, ms, text, tin, tout, served = call_adaptive(q)
    ans = parse(q, text) if st == 200 else None
    return dict(status=st, latency_ms=round(ms, 1), ok=ans is not None, answer=ans,
                raw=text if st == 200 else err_text(r)[:400],
                input_tokens=tin, output_tokens=tout, served_model=served)

if not a.live:
    print(f"DRY RUN ({a.provider} {a.model}): {len(Q)} questions would be sent (+{len(repeats_Q)} repeats). No calls made.")
    print("---- system ----\n" + SYSTEM + "\n---- example user prompt ----\n" + user_prompt(Q[0]))
    approx = sum(len(SYSTEM) + len(user_prompt(q)) for q in Q) / 4
    print(f"approx input tokens for the run: {approx:,.0f} (chars/4 estimate). Re-run with --live to call the API.")
    sys.exit(0)
key = "OPENAI_API_KEY" if a.provider == "openai" else "ANTHROPIC_API_KEY"
if not os.environ.get(key):
    sys.exit(f"{key} is not set")

fn = f"llm_{a.provider}_{a.model.replace('/', '_')}_raw.json"
out = {"provider": a.provider, "model": a.model, "answers": {}, "repeats": {}}
if a.resume and os.path.exists(fn):
    try:
        prev = json.load(open(fn))
        if prev.get("provider") == a.provider and prev.get("model") == a.model:
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
            atomic_write_json(fn, out)
    with print_lock:
        print(n, kind, q["id"], rec["status"], rec["latency_ms"], rec["answer"], flush=True)
    return rec

todo_answers = [q for q in Q if q["id"] not in out["answers"] or not out["answers"][q["id"]].get("ok")]
todo_repeats = [q for q in repeats_Q if q["id"] not in out["repeats"] or not out["repeats"][q["id"]].get("ok")]
print(f"live {a.provider} {a.model}: {len(todo_answers)} answers + {len(todo_repeats)} repeats, workers={a.workers}")

with ThreadPoolExecutor(max_workers=max(1, a.workers)) as ex:
    futs = [ex.submit(work, "answers", q) for q in todo_answers]
    futs += [ex.submit(work, "repeats", q) for q in todo_repeats]
    for f in as_completed(futs):
        f.result()

atomic_write_json(fn, out)
print("wrote", fn)
