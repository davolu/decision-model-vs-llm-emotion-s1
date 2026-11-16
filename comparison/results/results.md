# Emotion-S1 vs Claude vs GPT vs Jev

Comparison run on 16 November 2026 (Africa/Lagos, UTC+1), on the box's 8-core Intel Xeon CPU. All four systems answered the **same 600 held-out questions** (200 choice, 200 yes/no, 200 score), stratified across datasets, plus **100 second-pass repeats** (34 choice, 33 yes/no, 33 score) for consistency. Every one of the 2,800 paid or local calls succeeded and parsed (0 failures, 0 skipped).

No training was done on this machine. Emotion-S1 ran locally (fp32, then an optional int8 check). Claude, GPT and Jev were called live with `--live`.

## Model IDs

| System | Requested ID | Served ID | Settings |
|---|---|---|---|
| Emotion-S1 | local ModernBERT-base + 3 heads (`emotion_s1_model`) | — | fp32, 8 torch threads, CPU |
| Claude | `claude-sonnet-5-5` | `claude-sonnet-5-5` | `output_config.effort=low`, no temperature (sampling params are rejected) |
| GPT | `gpt-6.1-sol` | `gpt-6.1-sol` | `reasoning_effort=low`, no temperature (only the default 1 is accepted) |
| Jev | `jev-latest` | `jev-1.14.0` | TypeSafe `POST https://api.typesafe.ai/v1/systemone` |

The earlier mini-Jev comparison only ran against Jev (`jev-latest` → `jev-1.14.0`), not against Claude or GPT. These are each provider's current mainstream mid-tier workhorse: [Claude Sonnet 5.5](https://platform.claude.com/docs/en/models/overview) ("best combination of speed and intelligence") and [GPT-6.1 Sol](https://developers.openai.com/api/docs/models) ("balance intelligence and cost"), both $2 / $10 per 1M tokens. Effort was set to `low` so the judge task would not spend reasoning/thinking tokens; smoke tests showed 0 reasoning tokens (GPT) and 0 thinking tokens (Claude) at this setting.

## Headline results

| System | Choice acc | Yes/No acc | Score MAE (within 1) | ECE (c / y-n / s) | Median ms | p95 ms | $ per 1,000 turns | Self-disagreement | Parse-fail |
|---|---|---|---|---|---|---|---|---|---|
| Emotion-S1 (CPU) | 0.7 | 0.795 | 0.4122 (0.89) | 0.078 / 0.0772 / 0.0563 | 40.5 | 184.0 | 0.0 | 0.0 | 0.0/0.0/0.0 |
| anthropic:claude-sonnet-5-5 | 0.625 | 0.77 | 1.0299 (0.65) | 0.0831 / 0.1274 / 0.3206 | 877.6 | 1536.0 | 2.1466 | 0.02 | 0.0/0.0/0.0 |
| openai:gpt-6.1-sol | 0.64 | 0.775 | 1.0374 (0.655) | 0.2136 / 0.1788 / 0.4869 | 2133.1 | 4163.9 | 1.5718 | 0.03 | 0.0/0.0/0.0 |
| typesafe:jev-latest | 0.665 | 0.76 | 0.9928 (0.59) | 0.1661 / 0.1117 / 0.3604 | 119.0 | 159.3 | 0.0455 | 0.29* | 0.0/0.0/0.0 |

\*Jev's 0.29 exact-match disagreement is almost all soft score wobble of 0.01–0.07 on the 1–5 scale (max 0.07; none above 0.10). Categorical flips were 2/34 choice and 1/33 yes/no. If scores within 0.10 count as the same answer, Jev's disagreement rate is **0.03**, in line with Claude and GPT. Claude flipped 1 choice and 1 score (by a whole point). GPT flipped 2 choice and 1 score (by a whole point). Emotion-S1 is deterministic.

Accuracy is among successfully parsed answers (here, all of them). Score "within 1" is the share of expected scores within 1 point of the gold value. ECE is expected calibration error (15 bins). Latency in this table is per question on the 600-question mix: a local forward pass for Emotion-S1, and the HTTPS round-trip for the APIs, measured from the same box. A "turn" for the cost column is three separate questions (choice + score + yes/no) about one message.

On this set, Emotion-S1 is ahead on every accuracy metric, especially intensity (MAE 0.41 vs about 1.0 for the others), and it is the best calibrated. The gap is largest on GoEmotions Reddit comments (choice 0.68 vs 0.48–0.50 for the LLMs and Jev). DailyDialog, EmpatheticDialogues and ESConv are closer; Jev and the LLMs sometimes match or slightly beat Emotion-S1 on those yes/no slices.

## CPU latency (exclusive, 8 cores)

Measured after warm-up on this box, with no other heavy jobs running. Device: Intel Xeon, 8 logical CPUs, `torch 2.15.0+cpu`, 8 threads. The four-question turn is the same demo used in the notebook (choice over 6 emotions + intensity + "frustrated?" + "needs support?" = 9 encoder rows).

| | Median | p95 | n |
|---|---|---|---|
| One yes/no question (fp32) | 44.7 ms | 57.1 ms | 40 |
| All four questions, one turn (fp32) | 310.6 ms | 387.3 ms | 40 |
| One yes/no question (int8) | 28.0 ms | 34.2 ms | 40 |
| All four questions, one turn (int8) | 163.8 ms | 211.2 ms | 40 |

Across the 600 mixed comparison questions (fp32), median was 40.5 ms and p95 was 184.0 ms. Choice questions are slower because each option is an encoder row.

Colab's 2-core CPU timing (324 ms / 2646 ms) was not a useful guide. On this 8-core box the model is roughly an order of magnitude faster.

## int8 dynamic quantization (optional)

`torch.ao.quantization.quantize_dynamic` on all `nn.Linear` layers, `dtype=torch.qint8`. (The API is deprecated in favour of torchao, but it still runs on this PyTorch 2.15 CPU build.)

| | fp32 | int8 | Delta |
|---|---|---|---|
| State-dict size | 596.1 MB | 265.3 MB | −55.5% |
| Choice accuracy | 0.70 | 0.68 | −2.0 pp |
| Yes/No accuracy | 0.795 | 0.79 | −0.5 pp |
| Score MAE | 0.4122 | 0.5117 | +0.10 |
| Score within 1 | 0.89 | 0.865 | −2.5 pp |
| One yes/no, median | 44.7 ms | 28.0 ms | −37% |
| All four questions, median | 310.6 ms | 163.8 ms | −47% |

Agreement with fp32 answers: 82% of choice, 91.5% of yes/no, and a mean absolute score drift of 0.25. Useful if you want a smaller/faster CPU build; the fp32 model is the one compared to Claude, GPT and Jev above.

## Prices and token use

List prices fetched on 16 November 2026. Cost uses observed mean tokens × these rates, with no cache discount.

| System | Input / 1M | Output / 1M | Source | Mean tokens / question (in, out) | $ / 1,000 turns |
|---|---|---|---|---|---|
| Emotion-S1 | — | — | local | 0, 0 | 0.0000 |
| GPT `gpt-6.1-sol` | $2.00 | $10.00 | https://developers.openai.com/api/docs/pricing (standard short-context) | choice 163.0 / 22.4; yes/no 151.4 / 20.2; score 154.9 / 20.8 | 1.5718 |
| Claude `claude-sonnet-5-5` | $2.00 | $10.00 | https://platform.claude.com/docs/en/about-claude/pricing | choice 246.2 / 27.8; yes/no 226.5 / 22.3; score 237.7 / 22.6 | 2.1466 |
| Jev `jev-latest` (`jev-1.14.0`) | $0.042 | $0 (free) | https://docs.typesafe.ai/models | choice 364.6 / 64.3; yes/no 351.8 / 21.2; score 366.3 / 20.7 | 0.0455 |

Claude's tokenizer produces more input tokens than GPT's on the same prompt, which is why Claude costs more at the same $2/$10 list price. Jev bills input only.

Total paid usage for this run (600 answers + 100 repeats each):

- GPT: 110,331 input + 14,765 output tokens
- Claude: 166,913 input + 16,914 output tokens
- Jev: 253,449 input tokens (output free)

Approximate bill at the list prices above: about $0.37 for GPT, $0.50 for Claude, and $0.011 for Jev.

## Caveats

- **Jev does support score questions.** All 200 score items and 33 score repeats returned HTTP 200 and parsed. Jev's score head is 0-based; answers were shifted +1 onto the 1–5 rubric, same as the mini-Jev comparison.
- Labels for some yes/no families (especially "frustrated" and "needs support") were derived from other annotations and are noisy. Fine-grained emotion labels also overlap (fear vs surprise, anger vs disgust). That caps everyone's accuracy, not just Emotion-S1's.
- Emotion-S1 was trained on the training split of these datasets, so it is in-domain for this test set. Claude, GPT and Jev were not trained on this labelled set in the same way. The comparison answers "can a 26-minute specialist model replace an LLM judge on this task?", not "which system is a better general reasoner?"
- Effort was set to `low` for Claude and GPT. A higher effort setting might improve their scores, at higher cost and latency.
- Latency for the APIs includes network time from this box. Emotion-S1 latency is a local forward pass.
- Comparison-set ECE for Emotion-S1 (0.06–0.08 on 200 items per type) is a bit higher than the full 2,400-item test-set ECE in `metrics.json` (0.02–0.04), which is expected from the smaller sample.

## File paths

All under `/workspace/emotion-s1/compare/`:

- `results/results.md` — this report
- `results/results.json` — full metrics
- `results/accuracy_by_type.png` — grouped bar chart of accuracy / score-within-1
- `results/latency.png` — median and p95 latency (log scale)
- `results/cpu_latency.json` — exclusive fp32 CPU bench
- `results/int8.json` — int8 size, accuracy and exclusive latency
- `results/questions.json` — the 600-question set (seed 7)
- `results/s1_raw.json`, `s1_latency.json`, `s1_int8_raw.json`
- `results/llm_openai_gpt-6.1-sol_raw.json`
- `results/llm_anthropic_claude-sonnet-5-5_raw.json`
- `results/jev_raw.json`
- Weights used: `/workspace/emotion-s1/weights/emotion_s1_model/`
