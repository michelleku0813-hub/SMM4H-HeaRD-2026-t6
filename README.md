# TNM Staging Classification from TCGA Pathology Reports

trained checkpoint epoch_7 in outputs_large/ is on huggingface: https://huggingface.co/hsiaoeric/biocliniical-modernbert-tnm-staging

Multi-label classification of T (T1–T4), N (N0–N3), and M (M0–M1) from free-text pathology reports, with optional explainability.

## Setup

```bash
uv sync             # install dependencies
uv sync --extra dev # include dev dependencies (pytest)
```

## Data

- **Reports**: `TCGA_Reports.csv` (columns: `patient_filename`, `text`).
- **Labels**: `TCGA_Metadata/TCGA_T14_patients.csv`, `TCGA_N03_patients.csv`, `TCGA_M01_patients.csv`.

## 1. Data preparation

Join reports with T/N/M metadata, map raw AJCC labels to T14/N03/M01, and create stratified train/val/test splits:

```bash
uv run python src/data_prep.py --reports TCGA_Reports.csv --meta-dir TCGA_Metadata --out-dir data --seed 0
```

Output: `data/train.csv`, `data/val.csv`, `data/test.csv` with columns: `patient_filename`, `case_submitter_id`, `text`, `T`, `N`, `M`, `T_label`, `N_label`, `M_label`.

## 2. Training

Train BioClinical-ModernBERT-large with three heads for T, N, and M:

```bash
uv run python src/train.py --data-dir data --output-dir outputs_large \
  --batch-size 4 --grad-accum-steps 2 --epochs 8 --lr 2e-5
```

Options: `--encoder`, `--max-length`, `--lr`, `--weight-decay`, `--seed`, `--focal-loss`, `--focal-gamma`, `--label-smoothing`, `--no-class-weights`. Best checkpoint (by validation F1 macro avg) is saved to `outputs/best.pt` and config to `outputs/train_config.json`. Inverse-frequency class weights for T, N, M are enabled by default.

## 3. Prediction and submission

Generate predictions on the test set (or any CSV with a `text` column):

```bash
uv run python src/predict.py --checkpoint outputs/best.pt --input-csv data/test.csv --output-csv submission.csv
```

Submission CSV contains `patient_filename` (or `--id-col`), `T_label`, `N_label`, `M_label` (e.g. T2, N1, M0).

## 4. Evaluation

Compute F1 (macro) per label, AUROC (one-vs-rest) where applicable, and exact-match accuracy:

```bash
uv run python src/eval_metrics.py submission.csv data/test.csv --id-col patient_filename --output-metrics metrics.json
```

Predictions and ground-truth CSVs must both have `patient_filename` and `T_label`, `N_label`, `M_label`.

## 5. Explainability (optional)

Add an explanation column with evidence snippets from attention:

```bash
uv run python src/explain.py --input-csv data/test.csv --predictions-csv submission.csv --output-csv explained.csv
```

## Project layout

- `data/` – train/val/test CSVs (from data_prep)
- `src/data_prep.py` – join, label mapping, split
- `src/model.py` – TNMClassifier (encoder + 3 heads)
- `src/dataset.py` – PyTorch Dataset
- `src/train.py` – training loop, validation metrics
- `src/predict.py` – inference, submission CSV
- `src/eval_metrics.py` – F1, AUROC, exact-match
- `src/explain.py` – attention-based evidence snippets
- `configs/default.yaml` – reference config
- `src/constants.py` – shared label mappings and defaults
- `pyproject.toml` – project config and dependencies (uv)

## Reproducing metrics

1. Run data prep (step 1) with `--seed 0`.
2. Train (step 2) with default args and `--seed 0`.
3. Predict on `data/test.csv` (step 3).
4. Run eval_metrics (step 4) comparing submission to `data/test.csv`.

Exact-match and per-label F1/AUROC will be printed and optionally written to `metrics.json`.
