# exp09: prompted LLM baselines (JMIR revision, Y6 / Z3 / Z4)

2-shot inference on the 96 evaluation notes with large open LLMs, using exactly the few-shot
settings of 05_few_shot (prompt, ICL pool, seed 42, greedy). No training.


Implemented 2026-09-28: `QUANT`, `REASONING_EFFORT`, `FINAL_CHANNEL`, `INPUT_JSONL_HEAD` are read by
`src/inferpipe` via `scripts/infer/inference.sh`. `config/gptoss20b.env` is a 1-GPU pipeline check (not a paper row).

Models: `config/gptoss.env` (openai/gpt-oss-120b, MXFP4 dequantized to bf16, 3 GPUs) and `config/weblab.env`
(weblab-LLM-M/Weblab-MedLLM-gpt-oss-120b, bf16, 3 GPUs). DeepSeek-V4-Flash is planned separately (vLLM).
