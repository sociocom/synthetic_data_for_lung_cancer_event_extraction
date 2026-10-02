# 09: prompted LLM baselines (reference baselines without fine-tuning)

2-shot inference on the 96 evaluation notes with open-weight 120B LLMs, using the same prompt,
in-context example pool, seed (42), and greedy decoding as `05_few_shot`. No training.

- `config/gptoss.env`: openai/gpt-oss-120b (MXFP4 dequantized to bf16, 3 GPUs)
- `config/weblab.env`: weblab-LLM-M/Weblab-MedLLM-gpt-oss-120b (bf16, 3 GPUs)

`QUANT`, `REASONING_EFFORT`, `FINAL_CHANNEL`, and `INPUT_JSONL_HEAD` are read by `src/inferpipe`
through `scripts/infer/inference.sh`.

```bash
MODEL_CFG=gptoss bash scripts/experiments/09_llm_baselines/infer.sh
RUN_DIR=<run_dir> bash scripts/experiments/09_llm_baselines/score.sh
# or both models in sequence:
bash scripts/experiments/09_llm_baselines/run_all.sh
```
