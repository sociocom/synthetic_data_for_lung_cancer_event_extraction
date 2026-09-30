# Experiment 06: Mixed CV (Synthetic + Simulation)

This experiment trains on a mixed train split per fold, while keeping
validation/test as simulation-only.

## Policy

- CV split (`validation/test`) is created from simulation data only.
- Per fold train set:
  - base = `sim_train`
  - synthetic rows are re-sampled every epoch from `synth_pool`
  - sampling size is controlled by `MIX_SYNTH_RATIO:MIX_SIM_RATIO`
- Example:
  - `50:50` means sampled synthetic rows ~= `sim_train` rows.

## Run

- Full pipeline: `bash scripts/experiments/06_mix_cv/run.sh`
- Train only: `STAGE=train bash scripts/experiments/06_mix_cv/run.sh`
- Inference only: `STAGE=infer bash scripts/experiments/06_mix_cv/run.sh`
- Score only: `STAGE=score bash scripts/experiments/06_mix_cv/run.sh`
- Ratio sweep (100/50, 150/50, 200/50): `STAGE=sweep bash scripts/experiments/06_mix_cv/run.sh`

`train.sh` writes the run dir to:
- `scripts/experiments/06_mix_cv/last_run_dir.txt`

`sweep.sh` writes:
- `scripts/experiments/06_mix_cv/last_sweep_runs.txt` (path to latest sweep TSV)
- `<OUTPUT_BASE_DIR>/exp06_mix_cv_sweep_<timestamp>/runs.tsv`

Sweep env overrides:
- `SWEEP_SYNTH_RATIOS` (default: `100 150 200`, comma or space separated)
- `SWEEP_SIM_RATIO` (default: `50`)
- `SWEEP_STAGE` (`train` or `all`, default: `all`)
