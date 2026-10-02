# scripts

Shell wrappers for each stage of the pipeline.

| Directory | Content |
|---|---|
| `generate/` | Synthetic document generation and annotation via the DeepSeek API |
| `train/` | QLoRA fine-tuning (single split and cross-validation) |
| `infer/` | Inference (single split and cross-validation) |
| `score/` | Event-level scoring (single split and cross-validation) |
| `common/` | Run-directory helpers shared by the experiment wrappers |
| `experiments/` | One directory per experiment: `run.sh` selects the stage (`STAGE=train|infer|score|all`) and loads `config/*.env` |

## Experiments

| Directory | Condition in the manuscript |
|---|---|
| `01_zero_shot` | Zero-shot |
| `02_synth_transfer` | FT-Synthetic (training on Dsyn) |
| `03_current_cv` | FT-Manual (5-fold cross-validation on Dmanual) |
| `04_two_stage_cv` | FT-ALL (initialized from the FT-Synthetic adapter, then 5-fold cross-validation on Dmanual) |
| `05_few_shot` | Few-shot SLM baseline (2-shot) |
| `08_multiseed_robustness` | FT-Synthetic, FT-Manual, and FT-ALL repeated with 5 seeds; produces the reported scores |
| `09_llm_baselines` | Prompted 120B LLM baselines (gpt-oss-120b, Weblab-MedLLM-gpt-oss-120b) |

```bash
bash scripts/experiments/01_zero_shot/run.sh
bash scripts/experiments/02_synth_transfer/run.sh
bash scripts/experiments/03_current_cv/run.sh
INIT_ADAPTER_PATH=<exp02_run_dir>/best STAGE=stage2 bash scripts/experiments/04_two_stage_cv/run.sh
bash scripts/experiments/05_few_shot/run.sh
bash scripts/experiments/08_multiseed_robustness/run.sh
bash scripts/experiments/09_llm_baselines/run_all.sh
```

Paths to model checkpoints, input data, and output directories are set in each experiment's
`config/*.env` (placeholders such as `/path/to/...` must be replaced).
