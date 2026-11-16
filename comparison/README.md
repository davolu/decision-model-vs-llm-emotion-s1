# Emotion-S1 vs LLM judge vs Jev

This runs on the box (CPU) **after** the Colab notebook has produced weights. No API keys are stored in these files.
The scripts read `OPENAI_API_KEY`, `ANTHROPIC_API_KEY` and `TYPESAFE_API_KEY` from the environment.
The LLM and Jev scripts are **dry-run by default**, and they only make paid calls with `--live`.

```bash
cd /workspace/emotion-s1/compare
W=/path/to/emotion_s1_model        # e.g. copied down from My Drive/emotion_s1_model
python make_questions.py $W        # 600 held-out test questions (200 per type, stratified by dataset) -> questions.json
python run_emotion_s1.py $W        # Emotion-S1 on CPU -> s1_raw.json, s1_latency.json
python run_llm.py --provider openai    --model <cheap GPT model>    # dry run: shows the prompt + token estimate
python run_llm.py --provider anthropic --model <cheap Claude model>
python run_llm.py --provider openai    --model <model> --live --repeat 100   # paid: 600 calls + 100 repeats
python run_llm.py --provider anthropic --model <model> --live --repeat 100
python run_jev.py --live --repeat 100
python score.py --price "llm_openai_<model>_raw.json=IN,OUT" --price "llm_anthropic_<model>_raw.json=IN,OUT"
```

* Choose the model names on the run date, ideally one cheap and one mid-tier model for each provider.
* `--price` is USD per 1M input and output tokens, from each provider's pricing page on the run date.
* A "turn" is 3 questions about one message (choice, score and yes/no), each asked as a separate call.
* The LLM's confidence is self-reported, so its ECE shows how honest that number is.
* Self-disagreement is how often the same question gets a different answer the second time. Emotion-S1 is deterministic, so its rate is 0.
