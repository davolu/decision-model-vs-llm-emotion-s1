"""CPU latency for Emotion-S1 on this box. Inference only, no training.
Usage: python measure_cpu.py /path/to/emotion_s1_model [--n 40] -> results/cpu_latency.json
Times a single yes/no question and the same four-question turn used in the notebook, after warm-up."""
import json, os, sys, time, platform, argparse, numpy as np, torch

ap = argparse.ArgumentParser()
ap.add_argument("weights")
ap.add_argument("--n", type=int, default=40)
ap.add_argument("--warmup", type=int, default=5)
ap.add_argument("--out", default="results/cpu_latency.json")
a = ap.parse_args()

sys.path.insert(0, a.weights)
from emotion_s1_core import load_emotion_s1, ask

torch.set_grad_enabled(False)
torch.set_num_threads(os.cpu_count() or 8)

EMOTIONS = ["joy", "sadness", "anger", "fear", "surprise", "neutral"]
QS = {
    "emotion":    {"type": "choice", "question": "Which emotion is the user showing?", "options": EMOTIONS},
    "intensity":  {"type": "score", "question": "How intense is the user's emotion? 1 = calm, 5 = extremely intense.", "scale": [1, 5]},
    "frustrated": {"type": "noul", "question": "Is the user frustrated?"},
    "support":    {"type": "noul", "question": "Does the user need emotional support?"},
}
TURN = "This is the THIRD time I've reset the router and it's still dropping every ten minutes. I'm done."
CTX = [("User", "My internet keeps cutting out."), ("Assistant", "Please try restarting your router."),
       ("User", "Did that already."), ("Assistant", "Could you try a factory reset?")]

model = load_emotion_s1(a.weights, device="cpu")
for _ in range(a.warmup):
    ask(model, TURN, QS, context=CTX)

def bench(qs, n):
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        ask(model, TURN, qs, context=CTX)
        ts.append((time.perf_counter() - t) * 1000)
    rows = sum(len(q["options"]) if q["type"] == "choice" else 1 for q in qs.values())
    return dict(encoder_rows=rows, n=n,
                median_ms=round(float(np.median(ts)), 1),
                p90_ms=round(float(np.percentile(ts, 90)), 1),
                p95_ms=round(float(np.percentile(ts, 95)), 1),
                mean_ms=round(float(np.mean(ts)), 1),
                min_ms=round(float(np.min(ts)), 1),
                max_ms=round(float(np.max(ts)), 1))

cpu = next((l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")), platform.processor())
info = dict(
    device="cpu",
    cpu=cpu,
    logical_cpus=os.cpu_count(),
    torch_threads=torch.get_num_threads(),
    torch=torch.__version__,
    warmup=a.warmup,
    one_yes_no_question=bench({"frustrated": QS["frustrated"]}, a.n),
    all_4_questions=bench(QS, a.n),
)
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
json.dump(info, open(a.out, "w"), indent=1)
print(json.dumps(info, indent=1))
