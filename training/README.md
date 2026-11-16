# Emotion-S1: a small emotion decision model (tutorial project)

Emotion-S1 is a ModernBERT-base model with three heads: yes/no ("noul"), choice and score. It's trained from scratch on six emotion datasets and answers typed questions about a chat turn with calibrated confidence. It's the same recipe as `../mini-jev`.

## Files
* `emotion_s1_t4.ipynb`: the Colab notebook (T4). Set `QUICK = True` for a ~10-minute test, or `QUICK = False` for the full run, which has a 45-minute time guard.
* `build_notebook.py`: generates the notebook. Edit this file and `emotion_s1_core.py`, then re-run it.
* `emotion_s1_core.py`: the model, input format and `ask()`. The notebook writes this file with `%%writefile` and saves it next to the weights.
* `smoke_test.py` / `smoke_test.log`: a CPU smoke test that runs every cell with `EMOTION_S1_TINY=1` (a handful of rows, 3 training steps). It only proves the code runs end to end; the metrics in the log mean nothing.
* `datasets.md`: datasets, sources and the commercial vs non-commercial licence table.
* `PLAN.md`: the design plan.
* `compare/`: Emotion-S1 vs Claude, GPT and Jev on the same held-out questions. It runs on a CPU after training, and the paid calls only run with `--live` (see `compare/README.md`).

## Run
1. Upload `emotion_s1_t4.ipynb` to Colab, then choose Runtime → Change runtime type → T4 GPU → Run all.
2. The outputs are saved in `/content/emotion_s1`. The Drive cell copies them to `My Drive/emotion_s1_model`.
3. Download that folder to the box, then follow `compare/README.md`.

## Inference anywhere (CPU)
```python
import sys; sys.path.insert(0, "emotion_s1_model")
from emotion_s1_core import load_emotion_s1, ask
m = load_emotion_s1("emotion_s1_model", device="cpu")
ask(m, "It's STILL broken after three resets.", {
  "emotion":    {"type": "choice", "question": "Which emotion is the user showing?", "options": ["joy", "anger", "sadness", "fear", "neutral"]},
  "intensity":  {"type": "score",  "question": "How intense is the user's emotion? 1 = calm, 5 = extremely intense.", "scale": [1, 5]},
  "frustrated": {"type": "noul",   "question": "Is the user frustrated?"}},
  context=[("Assistant", "Could you try restarting the router?")])
```
