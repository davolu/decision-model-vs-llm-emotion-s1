"""Emotion-S1 INFERENCE ONLY (no training). Usage: python run_emotion_s1.py /path/to/emotion_s1_model [--device cpu]
-> s1_raw.json (answers) and s1_latency.json (CPU latency). Uses the emotion_s1_core.py saved next to the weights."""
import json, sys, time, os, platform, argparse, numpy as np, torch
ap = argparse.ArgumentParser(); ap.add_argument("weights"); ap.add_argument("--device", default="cpu"); a = ap.parse_args()
sys.path.insert(0, a.weights)
from emotion_s1_core import load_emotion_s1, ask_state
torch.set_grad_enabled(False)
model = load_emotion_s1(a.weights, device=a.device)
Q = json.load(open("questions.json"))
def to_s1(q):
    if q["type"] == "noul":   return {"type": "noul", "question": q["question"]}
    if q["type"] == "choice": return {"type": "choice", "question": q["question"], "options": q["options"]}
    return {"type": "score", "question": q["rubric"], "scale": q["scale"]}
for q in Q[:5]: ask_state(model, q["state"], {"q": to_s1(q)})          # warm-up
out = {}
for q in Q:
    t = time.perf_counter(); r = ask_state(model, q["state"], {"q": to_s1(q)})["q"]; ms = (time.perf_counter() - t) * 1000
    if q["type"] == "noul":   ans = dict(answer=r["answer"], p_yes=r["probability"], confidence=r["confidence"])
    elif q["type"] == "choice": ans = dict(answer=r["answer"], confidence=r["confidence"], probabilities=r["probabilities"])
    else:
        top = max(r["distribution"].items(), key=lambda kv: kv[1])
        ans = dict(score=r["expected"], top_level=float(top[0]), confidence=r["confidence"])
    out[q["id"]] = dict(latency_ms=round(ms, 2), ok=True, answer=ans, input_tokens=0, output_tokens=0)
json.dump(out, open("s1_raw.json", "w"), indent=1)
lat = [v["latency_ms"] for v in out.values()]
cpu = next((l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")), platform.processor()) \
    if os.path.exists("/proc/cpuinfo") else platform.processor()
info = dict(device=a.device, cpu=cpu, logical_cpus=os.cpu_count(), torch_threads=torch.get_num_threads(), torch=torch.__version__,
            n=len(lat), median_ms=round(float(np.median(lat)), 1), p90_ms=round(float(np.percentile(lat, 90)), 1))
json.dump(info, open("s1_latency.json", "w"), indent=1); print(info)
