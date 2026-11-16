"""Bar charts for the comparison: accuracy by question type, and latency."""
import json, os, argparse
import numpy as np
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--results", default="results/results.json")
ap.add_argument("--out-dir", default="results")
a = ap.parse_args()
rows = json.load(open(a.results))
os.makedirs(a.out_dir, exist_ok=True)

# Short display names
def name(r):
    s = r["system"]
    s = s.replace("Emotion-S1 (CPU)", "Emotion-S1")
    s = s.replace("openai:", "GPT ")
    s = s.replace("anthropic:", "Claude ")
    s = s.replace("typesafe:", "Jev ")
    return s

labels = [name(r) for r in rows]
choice = [r["choice"].get("accuracy") if r["choice"].get("accuracy") is not None else 0 for r in rows]
noul = [r["noul"].get("accuracy") if r["noul"].get("accuracy") is not None else 0 for r in rows]
# For score, plot "within 1" as a comparable 0-1 accuracy-like metric, and we'll annotate MAE on the latency-free chart
within = [r["score"].get("within_1") if r["score"].get("within_1") is not None else 0 for r in rows]
mae = [r["score"].get("mae") for r in rows]
lat = [(r["latency_ms"] or {}).get("median") or 0 for r in rows]
p95 = [(r["latency_ms"] or {}).get("p95") or 0 for r in rows]

plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 11,
    "figure.facecolor": "white", "axes.facecolor": "white",
})
colors = ["#2F6FED", "#1A9B6C", "#D97706"]  # choice, yes/no, score-within-1

x = np.arange(len(labels))
w = 0.25
fig, ax = plt.subplots(figsize=(10, 5.2))
b1 = ax.bar(x - w, choice, w, label="Choice accuracy", color=colors[0])
b2 = ax.bar(x, noul, w, label="Yes/No accuracy", color=colors[1])
b3 = ax.bar(x + w, within, w, label="Score within 1 point", color=colors[2])
ax.set_ylim(0, 1.08)
ax.set_ylabel("Accuracy")
ax.set_title("Emotion-S1 vs Claude vs GPT vs Jev — accuracy by question type")
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=15, ha="right")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
ax.legend(frameon=False, loc="upper right")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
for bars in (b1, b2, b3):
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.0%}", xy=(bar.get_x() + bar.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)
# MAE footnotes under the score bars
for i, m in enumerate(mae):
    if m is not None:
        ax.annotate(f"MAE {m:.2f}", xy=(x[i] + w, 0.02), ha="center", va="bottom", fontsize=7, color="#555")
fig.tight_layout()
p1 = os.path.join(a.out_dir, "accuracy_by_type.png")
fig.savefig(p1, dpi=160)
plt.close()

fig, ax = plt.subplots(figsize=(10, 5.2))
b1 = ax.bar(x - 0.15, lat, 0.3, label="Median", color="#2F6FED")
b2 = ax.bar(x + 0.15, p95, 0.3, label="p95", color="#6B8FCE")
ax.set_ylabel("Latency (ms, log scale)")
ax.set_title("Latency per question (HTTP round-trip for APIs; local forward pass for Emotion-S1)")
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=15, ha="right")
ax.set_yscale("log")
ax.legend(frameon=False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
for bar in list(b1) + list(b2):
    h = bar.get_height()
    if h > 0:
        ax.annotate(f"{h:.0f}", xy=(bar.get_x() + bar.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)
fig.tight_layout()
p2 = os.path.join(a.out_dir, "latency.png")
fig.savefig(p2, dpi=160)
plt.close()
print("wrote", p1, "and", p2)
