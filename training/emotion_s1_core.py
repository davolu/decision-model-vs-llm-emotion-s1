"""Emotion-S1 core: input formatting, the model (one encoder + three typed heads) and ask().
The Colab notebook writes this exact file (%%writefile) and the local compare scripts import it, so the
training-time and inference-time input format can never drift apart."""
import inspect, os, numpy as np, torch, torch.nn as nn
from transformers import AutoTokenizer, AutoModel

N_BINS = 5          # score head: 5 evenly spaced levels across whatever scale you ask for (1..5 -> exactly 1,2,3,4,5)
MAX_CONTEXT_TURNS = 4


def format_state(turn, context=None, max_turns=MAX_CONTEXT_TURNS):
    """The 'state' the model reads: up to `max_turns` earlier turns, then the user's latest turn.
    context = list of (role, text) tuples, oldest first; role is usually "User" or "Assistant"."""
    lines = []
    ctx = list(context or [])[-max_turns:] if max_turns else []
    if ctx:
        lines.append("Earlier turns:")
        lines += [f"{role}: {text}" for role, text in ctx]
    lines += ["Latest user turn:", f"User: {turn}"]
    return "\n".join(lines)


def pair_noul(state, question):        return (f"Yes/no question: {question}", state)
def pair_choice(state, question, opt): return (f"Multiple choice question: {question} Candidate answer: {opt}", state)
def pair_score(state, rubric, scale):  return (f"Scoring task, scale {scale[0]} to {scale[1]}. Rubric: {rubric}", state)


class EmotionS1(nn.Module):
    def __init__(self, backbone, max_len=192, n_bins=N_BINS, device="cpu", use_amp=False):
        super().__init__()
        self.tok = AutoTokenizer.from_pretrained(backbone)
        self.tok.truncation_side = "left"            # if a chat is too long, drop the OLDEST context, never the latest turn
        self.enc = AutoModel.from_pretrained(backbone)
        h = self.enc.config.hidden_size
        self.drop = nn.Dropout(0.1)
        self.noul, self.choice, self.score = nn.Linear(h, 1), nn.Linear(h, 1), nn.Linear(h, n_bins)
        self.register_buffer("temps", torch.ones(3))   # calibration temperatures [noul, choice, score]
        self.accepts_tt = "token_type_ids" in inspect.signature(self.enc.forward).parameters
        self.max_len, self.n_bins, self.device_name, self.use_amp = max_len, n_bins, device, use_amp

    def to_device(self, device):
        self.device_name = device; self.use_amp = self.use_amp and device == "cuda"
        return self.to(device)

    def encode_pairs(self, pairs):
        enc = self.tok([a for a, _ in pairs], [b for _, b in pairs], truncation="only_second",
                       max_length=self.max_len, padding=True, return_tensors="pt")
        return {k: v.to(self.device_name) for k, v in enc.items()}

    def pool(self, input_ids, attention_mask, token_type_ids=None):
        kw = dict(input_ids=input_ids, attention_mask=attention_mask)
        if self.accepts_tt and token_type_ids is not None: kw["token_type_ids"] = token_type_ids
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.use_amp and self.device_name == "cuda"):
            hs = self.enc(**kw).last_hidden_state
        m = attention_mask.unsqueeze(-1).to(hs.dtype)
        return self.drop((hs * m).sum(1) / m.sum(1).clamp(min=1)).float()

    def heads(self, pooled):
        return self.noul(pooled).squeeze(-1), self.choice(pooled).squeeze(-1), self.score(pooled)

    def save(self, path, extra=None):
        os.makedirs(path, exist_ok=True)
        self.enc.save_pretrained(f"{path}/backbone"); self.tok.save_pretrained(f"{path}/backbone")
        torch.save({"noul": self.noul.state_dict(), "choice": self.choice.state_dict(), "score": self.score.state_dict(),
                    "temps": self.temps.cpu(), "n_bins": self.n_bins, "max_len": self.max_len, **(extra or {})},
                   f"{path}/heads.pt")


def load_emotion_s1(path, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(f"{path}/heads.pt", map_location="cpu")
    m = EmotionS1(f"{path}/backbone", max_len=ck["max_len"], n_bins=ck["n_bins"], device=device, use_amp=device == "cuda")
    for k in ("noul", "choice", "score"): getattr(m, k).load_state_dict(ck[k])
    m.temps.copy_(ck["temps"])
    return m.to(device).eval()


@torch.no_grad()
def ask_state(model, state, questions):
    """questions = {name: {"type": "noul"|"choice"|"score", "question": str, "options": [...], "scale": [lo, hi]}}.
    Every question (and every choice option) is one encoder row; all rows go through ONE batched forward pass."""
    pairs, slots = [], []
    for name, q in questions.items():
        t = q["type"]; start = len(pairs)
        if t == "noul":     pairs.append(pair_noul(state, q["question"]))
        elif t == "choice": pairs += [pair_choice(state, q["question"], o) for o in q["options"]]
        elif t == "score":  pairs.append(pair_score(state, q.get("instructions") or q["question"], q.get("scale", [1, 5])))
        else: raise ValueError(f"unknown question type {t!r}")
        slots.append((name, t, start, len(pairs)))
    ln, lc, ls = (x.float().cpu() for x in model.heads(model.pool(**model.encode_pairs(pairs))))
    Tn, Tc, Ts = model.temps.tolist(); out = {}
    for name, t, a, b in slots:
        q = questions[name]
        if t == "noul":
            p = float(torch.sigmoid(ln[a] / Tn))
            out[name] = dict(type="noul", probability=round(p, 4), answer=p >= 0.5, confidence=round(max(p, 1 - p), 4))
        elif t == "choice":
            P = torch.softmax(lc[a:b] / Tc, -1).tolist(); best = int(np.argmax(P))
            out[name] = dict(type="choice", answer=q["options"][best], confidence=round(P[best], 4),
                             probabilities={o: round(p, 4) for o, p in zip(q["options"], P)})
        else:
            lo, hi = q.get("scale", [1, 5]); P = torch.softmax(ls[a] / Ts, -1).numpy()
            levels = np.linspace(lo, hi, model.n_bins)
            out[name] = dict(type="score", expected=round(float(P @ levels), 3), confidence=round(float(P.max()), 4),
                             distribution={round(float(l), 3): round(float(p), 4) for l, p in zip(levels, P)})
    return out


def ask(model, turn, questions, context=None):
    """ask(model, "I've asked three times and it's STILL broken.", {...}, context=[("Assistant", "Did that fix it?")])"""
    return ask_state(model, format_state(turn, context), questions)
