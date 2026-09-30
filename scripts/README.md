# scripts

実験用スクリプトの構成です。

## 構成
- `train/`: 学習実処理（非CV/CV）
- `infer/`: 推論実処理（非CV/CV）
- `score/`: 評価実処理（非CV/CV）
- `generate/`: 合成データ生成
- `experiments/`: 実験ラッパー（stage選択 + RUN_DIR解決 + config読込）

## 実行例
- `bash scripts/experiments/01_zero_shot/run.sh`
- `bash scripts/experiments/02_synth_transfer/run.sh`
- `bash scripts/experiments/03_current_cv/run.sh`
- `INIT_ADAPTER_PATH=<exp02_run_dir>/best STAGE=stage2 bash scripts/experiments/04_two_stage_cv/run.sh`
- `bash scripts/experiments/05_few_shot/run.sh`
- `bash scripts/experiments/06_mix_cv/run.sh`
- `bash scripts/experiments/07_synth_holdout_10pct/run.sh`

## 設定変更
各実験の `config/*.env` を編集してください。
