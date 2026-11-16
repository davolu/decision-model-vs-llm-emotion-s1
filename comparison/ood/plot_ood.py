"""OOD charts plus a combined in-domain vs out-of-domain accuracy figure."""
import json, os
import numpy as np
import matplotlib.pyplot as plt

OUT = "/workspace/emotion-s1/compare/results_ood"
ood = json.load(open(os.path.join(OUT, "results.json")))
idd = json.load(open("/workspace/emotion-s1/compare/results/results.json"))

def name(r):
    s = r["system"]
    s = s.replace("Emotion-S1 (CPU)", "Emotion-S1")
    s = s.replace("openai:", "GPT ")
    s = s.replace("anthropic:", "Claude ")
    s = s.replace("typesafe:", "Jev ")
    return s

def key(r):
    s = r["system"]
    if "Emotion-S1" in s: return "s1"
    if "anthropic" in s: return "claude"
    if "openai" in s: return "gpt"
    if "typesafe" or "jev" in s.lower(): return "jev"
    return s

plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 11,
    "figure.facecolor": "white", "axes.facecolor": "white",
})

# --- OOD accuracy (choice + yes/no only; no intensity labels) ---
labels = [name(r) for r in ood]
choice = [r["choice"].get("accuracy") or 0 for r in ood]
noul = [r["noul"].get("accuracy") or 0 for r in ood]
x = np.arange(len(labels)); w = 0.35
fig, ax = plt.subplots(figsize=(10, 5.2))
b1 = ax.bar(x - w/2, choice, w, label="Choice accuracy", color="#2F6FED")
b2 = ax.bar(x + w/2, noul, w, label="Yes/No accuracy", color="#1A9B6C")
ax.set_ylim(0, 1.08); ax.set_ylabel("Accuracy")
ax.set_title("Out-of-domain: Emotion-S1 vs Claude vs GPT vs Jev")
ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15, ha="right")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
ax.legend(frameon=False, loc="upper right")
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
# random-guess band for 6–7 way choice (~15%)
ax.axhline(1/7, color="#888", ls="--", lw=0.8, label="_rand")
ax.text(len(labels)-0.02, 1/7 + 0.02, "random choice (~14–17%)", ha="right", va="bottom", fontsize=8, color="#555")
for bars in (b1, b2):
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.0%}", xy=(bar.get_x()+bar.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)
fig.tight_layout()
p1 = os.path.join(OUT, "accuracy_by_type.png")
fig.savefig(p1, dpi=160); plt.close()

# --- latency ---
lat = [(r["latency_ms"] or {}).get("median") or 0 for r in ood]
p95 = [(r["latency_ms"] or {}).get("p95") or 0 for r in ood]
fig, ax = plt.subplots(figsize=(10, 5.2))
b1 = ax.bar(x - 0.15, lat, 0.3, label="Median", color="#2F6FED")
b2 = ax.bar(x + 0.15, p95, 0.3, label="p95", color="#6B8FCE")
ax.set_ylabel("Latency (ms, log scale)")
ax.set_title("Out-of-domain latency per question")
ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15, ha="right")
ax.set_yscale("log")
ax.legend(frameon=False)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
for bar in list(b1)+list(b2):
    h = bar.get_height()
    if h > 0:
        ax.annotate(f"{h:.0f}", xy=(bar.get_x()+bar.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)
fig.tight_layout()
p2 = os.path.join(OUT, "latency.png")
fig.savefig(p2, dpi=160); plt.close()

# --- combined in-domain vs OOD ---
id_map = {key(r): r for r in idd}
ood_map = {key(r): r for r in ood}
order = ["s1", "claude", "gpt", "jev"]
pretty = {"s1": "Emotion-S1", "claude": "Claude Sonnet 5.5", "gpt": "GPT-6.1 Sol", "jev": "Jev 1.14.0"}
xx = np.arange(len(order)); ww = 0.2
fig, ax = plt.subplots(figsize=(11, 5.4))
series = [
    ("Choice, in-domain",  [id_map[k]["choice"]["accuracy"] for k in order], "#2F6FED"),
    ("Choice, out-of-domain", [ood_map[k]["choice"]["accuracy"] for k in order], "#9BB8F5"),
    ("Yes/No, in-domain", [id_map[k]["noul"]["accuracy"] for k in order], "#1A9B6C"),
    ("Yes/No, out-of-domain", [ood_map[k]["noul"]["accuracy"] for k in order], "#8DCEB0"),
]
offsets = [-1.5*ww, -0.5*ww, 0.5*ww, 1.5*ww]
for (lab, vals, col), off in zip(series, offsets):
    bars = ax.bar(xx + off, vals, ww, label=lab, color=col)
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.0%}", xy=(bar.get_x()+bar.get_width()/2, h),
                    xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", fontsize=7)
ax.set_ylim(0, 1.08); ax.set_ylabel("Accuracy")
ax.set_title("In-domain vs out-of-domain accuracy")
ax.set_xticks(xx); ax.set_xticklabels([pretty[k] for k in order])
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
ax.legend(frameon=False, ncol=2, loc="upper right")
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
fig.tight_layout()
p3 = os.path.join(OUT, "id_vs_ood.png")
fig.savefig(p3, dpi=160); plt.close()
print("wrote", p1)
print("wrote", p2)
print("wrote", p3)
