"""
Train TNM staging model: backbone encoder/decoder + three classification heads.
Supports both CE (cross-entropy) and CORAL (ordinal regression) head types.
"""
import argparse
from datetime import datetime
import json
import logging
import os
import platform
import subprocess
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from constants import (
    DEFAULT_ENCODER, DEFAULT_MAX_LENGTH,
    T_NUM_LABELS, N_NUM_LABELS, M_NUM_LABELS,
    DEFAULT_LORA_R, DEFAULT_LORA_ALPHA, DEFAULT_LORA_DROPOUT,
)
from data.dataset import TNMDataset
from models.classifier import TNMClassifier

logger = logging.getLogger(__name__)


def is_output_dir_explicit(argv):
    """Return True if user explicitly passed --output-dir."""
    return any(arg == "--output-dir" or arg.startswith("--output-dir=") for arg in argv)


def get_git_commit_hash():
    """Best-effort git commit hash for reproducibility metadata."""
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Loss helpers
# ---------------------------------------------------------------------------

def masked_ce_loss(criterion, logits, labels, mask):
    """CE loss only on valid samples."""
    if mask.any():
        return criterion(logits[mask], labels[mask])
    return torch.tensor(0.0, device=logits.device, requires_grad=True)


def coral_loss(logits, labels, mask):
    """CORAL ordinal loss: sum of K-1 binary cross-entropies."""
    if not mask.any():
        return torch.tensor(0.0, device=logits.device, requires_grad=True)
    logits = logits[mask]
    labels = labels[mask]
    num_thresholds = logits.shape[1]
    levels = torch.arange(num_thresholds, device=logits.device).float()
    targets = (labels.unsqueeze(1).float() > levels).float()
    return F.binary_cross_entropy_with_logits(logits, targets)


def binary_loss(logits, labels, mask):
    """BCE for binary M head (CORAL mode)."""
    if not mask.any():
        return torch.tensor(0.0, device=logits.device, requires_grad=True)
    logits = logits[mask].squeeze(-1)
    labels = labels[mask].float()
    return F.binary_cross_entropy_with_logits(logits, labels)


# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------

def coral_predict(logits):
    """Predict class from CORAL logits: count thresholds exceeded."""
    probs = torch.sigmoid(logits)
    return (probs > 0.5).sum(dim=1).long()


def binary_predict(logits):
    """Predict class from binary logit."""
    return (torch.sigmoid(logits.squeeze(-1)) > 0.5).long()


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(pred_t, pred_n, pred_m, true_t, true_n, true_m,
                    mask_t=None, mask_n=None, mask_m=None):
    """F1 (macro) per component, exact-match. Respects validity masks."""
    pred_t, pred_n, pred_m = np.array(pred_t), np.array(pred_n), np.array(pred_m)
    true_t, true_n, true_m = np.array(true_t), np.array(true_n), np.array(true_m)

    if mask_t is None:
        mask_t = np.ones(len(true_t), dtype=bool)
    if mask_n is None:
        mask_n = np.ones(len(true_n), dtype=bool)
    if mask_m is None:
        mask_m = np.ones(len(true_m), dtype=bool)

    def _f1(y_true, y_pred, avg):
        return float(f1_score(y_true, y_pred, average=avg, zero_division=0))

    def _prec(y_true, y_pred, avg):
        return float(precision_score(y_true, y_pred, average=avg, zero_division=0))

    def _rec(y_true, y_pred, avg):
        return float(recall_score(y_true, y_pred, average=avg, zero_division=0))

    tt, pt = true_t[mask_t], pred_t[mask_t]
    tn, pn = true_n[mask_n], pred_n[mask_n]
    tm, pm = true_m[mask_m], pred_m[mask_m]

    f1_t = _f1(tt, pt, "macro") if mask_t.any() else 0.0
    f1_n = _f1(tn, pn, "macro") if mask_n.any() else 0.0
    f1_m = _f1(tm, pm, "macro") if mask_m.any() else 0.0

    mi_f1_t = _f1(tt, pt, "micro") if mask_t.any() else 0.0
    mi_f1_n = _f1(tn, pn, "micro") if mask_n.any() else 0.0
    mi_f1_m = _f1(tm, pm, "micro") if mask_m.any() else 0.0

    ma_pr_t = _prec(tt, pt, "macro") if mask_t.any() else 0.0
    ma_re_t = _rec(tt, pt, "macro") if mask_t.any() else 0.0
    ma_pr_n = _prec(tn, pn, "macro") if mask_n.any() else 0.0
    ma_re_n = _rec(tn, pn, "macro") if mask_n.any() else 0.0
    ma_pr_m = _prec(tm, pm, "macro") if mask_m.any() else 0.0
    ma_re_m = _rec(tm, pm, "macro") if mask_m.any() else 0.0

    all_valid = mask_t & mask_n & mask_m
    if all_valid.any():
        exact = float(np.mean(
            (pred_t[all_valid] == true_t[all_valid])
            & (pred_n[all_valid] == true_n[all_valid])
            & (pred_m[all_valid] == true_m[all_valid])
        ))
    else:
        exact = 0.0

    return {
        "f1_t": f1_t, "f1_n": f1_n, "f1_m": f1_m,
        "f1_macro_avg": (f1_t + f1_n + f1_m) / 3,
        "exact_match": exact,
        "micro_f1": (mi_f1_t + mi_f1_n + mi_f1_m) / 3,
        "macro_precision": (ma_pr_t + ma_pr_n + ma_pr_m) / 3,
        "macro_recall": (ma_re_t + ma_re_n + ma_re_m) / 3,
        "n_valid_t": int(mask_t.sum()),
        "n_valid_n": int(mask_n.sum()),
        "n_valid_m": int(mask_m.sum()),
    }


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(data_dir, meta_dir="TCGA_Metadata"):
    """Load train/val data.

    Handles the competition format: columns = patient_filename, text, t, n, m
    where t is 1-indexed (1-4), n is 0-indexed (0-3), m is 0-indexed (0-1),
    and NaN means missing label.

    If ``val.csv`` has no TNM labels, attempts to enrich from TCGA_Metadata.
    """
    train_path = os.path.join(data_dir, "train.csv")
    val_path = os.path.join(data_dir, "val.csv")

    def _normalize(df: pd.DataFrame) -> pd.DataFrame:
        required_cols = ["patient_filename", "text", "t", "n", "m"]
        missing_cols = [c for c in required_cols if c not in df.columns]

        # Ensure expected columns exist so downstream code can always access them.
        # Missing labels are represented by sentinel -1.
        if "patient_filename" not in df.columns:
            df["patient_filename"] = ""
        if "text" not in df.columns:
            raise ValueError("Input CSV must contain a 'text' column.")
        for col in ("t", "n", "m"):
            if col not in df.columns:
                df[col] = -1

        # Normalize t to 0-indexed (T1->0, T2->1, T3->2, T4->3)
        df = df.copy()
        df["t"] = pd.to_numeric(df["t"], errors="coerce") - 1

        # Fill missing with sentinel -1
        for col in ("t", "n", "m"):
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(-1).astype(int)

        if missing_cols:
            logger.warning("CSV missing columns %s; filled with defaults where possible.", missing_cols)
        return df

    def _has_labeled_targets(df: pd.DataFrame) -> bool:
        if not all(col in df.columns for col in ("t", "n", "m")):
            return False
        # Need at least one valid label in each head for meaningful validation.
        return all(pd.to_numeric(df[col], errors="coerce").notna().any() for col in ("t", "n", "m"))

    def _enrich_with_metadata(df: pd.DataFrame, meta_dir: str) -> pd.DataFrame:
        """Join TCGA_Metadata labels into an unlabeled DataFrame."""
        from data.data_prep import map_t_to_t14, map_n_to_n03, map_m_to_m01

        t_path = os.path.join(meta_dir, "TCGA_T14_patients.csv")
        n_path = os.path.join(meta_dir, "TCGA_N03_patients.csv")
        m_path = os.path.join(meta_dir, "TCGA_M01_patients.csv")
        if not all(os.path.exists(p) for p in (t_path, n_path, m_path)):
            return None

        df = df.copy()
        df["case_submitter_id"] = df["patient_filename"].str.split(".").str[0]

        t_df = pd.read_csv(t_path)
        t_df["t"] = t_df["ajcc_pathologic_t"].apply(map_t_to_t14)
        t_df = t_df.dropna(subset=["t"]).astype({"t": int})
        t_df = t_df[["case_submitter_id", "t"]].drop_duplicates(subset=["case_submitter_id"], keep="first")

        n_df = pd.read_csv(n_path)
        n_df["n"] = n_df["ajcc_pathologic_n"].apply(map_n_to_n03)
        n_df = n_df.dropna(subset=["n"]).astype({"n": int})
        n_df = n_df[["case_submitter_id", "n"]].drop_duplicates(subset=["case_submitter_id"], keep="first")

        m_df = pd.read_csv(m_path)
        m_df["m"] = m_df["ajcc_pathologic_m"].apply(map_m_to_m01)
        m_df = m_df.dropna(subset=["m"]).astype({"m": int})
        m_df = m_df[["case_submitter_id", "m"]].drop_duplicates(subset=["case_submitter_id"], keep="first")

        df = df.merge(t_df, on="case_submitter_id", how="left")
        df = df.merge(n_df, on="case_submitter_id", how="left")
        df = df.merge(m_df, on="case_submitter_id", how="left")
        df.drop(columns=["case_submitter_id"], inplace=True)

        # t from metadata is 0-indexed; convert to 1-indexed to match train.csv
        # so _normalize() handles both uniformly
        df["t"] = df["t"].where(df["t"].isna(), df["t"] + 1)
        for col in ("t", "n", "m"):
            df[col] = df[col].fillna(-1).astype(int)

        logger.info(
            "Enriched val from metadata: T=%d, N=%d, M=%d valid out of %d",
            (df["t"] >= 0).sum(), (df["n"] >= 0).sum(), (df["m"] >= 0).sum(), len(df),
        )
        return df

    train_df = _normalize(pd.read_csv(train_path))

    if not os.path.exists(val_path):
        raise FileNotFoundError(f"Validation file not found: {val_path}")

    raw_val_df = pd.read_csv(val_path)
    if _has_labeled_targets(raw_val_df):
        val_df = _normalize(raw_val_df)
        logger.info("Using provided validation split: %s", val_path)
    else:
        logger.info("val.csv has no labels, enriching from %s ...", meta_dir)
        val_df = _enrich_with_metadata(raw_val_df, meta_dir)
        if val_df is None:
            raise FileNotFoundError(
                f"val.csv has no labels and metadata not found at {meta_dir}. "
                "Run data_prep.py --enrich-val first."
            )

    logger.info("Train: %d samples, Val: %d samples", len(train_df), len(val_df))
    for name, part in [("train", train_df), ("val", val_df)]:
        for col in ("t", "n", "m"):
            valid = (part[col] >= 0).sum()
            logger.info("  %s %s: %d valid / %d total", name, col, valid, len(part))
    return train_df, val_df


# ---------------------------------------------------------------------------
# Training / evaluation loops
# ---------------------------------------------------------------------------

def train_epoch(model, loader, optimizer, scheduler, device, grad_accum_steps,
                head_type, criterion_t=None, criterion_n=None, criterion_m=None,
                train_pbar=None):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    for step, batch in enumerate(loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels_t = batch["labels_t"].to(device)
        labels_n = batch["labels_n"].to(device)
        labels_m = batch["labels_m"].to(device)
        mask_t = batch["mask_t"].to(device)
        mask_n = batch["mask_n"].to(device)
        mask_m = batch["mask_m"].to(device)

        token_type_ids = batch.get("token_type_ids")
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(device)

        logits_t, logits_n, logits_m = model(
            input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
        )

        if head_type == "coral":
            loss = (coral_loss(logits_t, labels_t, mask_t)
                    + coral_loss(logits_n, labels_n, mask_n)
                    + binary_loss(logits_m, labels_m, mask_m))
        else:
            loss = (masked_ce_loss(criterion_t, logits_t, labels_t, mask_t)
                    + masked_ce_loss(criterion_n, logits_n, labels_n, mask_n)
                    + masked_ce_loss(criterion_m, logits_m, labels_m, mask_m))

        loss = loss / grad_accum_steps
        loss.backward()

        if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad()
            if train_pbar is not None:
                train_pbar.update(1)

        total_loss += loss.item() * grad_accum_steps
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, device, head_type):
    model.eval()
    preds_t, preds_n, preds_m = [], [], []
    trues_t, trues_n, trues_m = [], [], []
    masks_t, masks_n, masks_m = [], [], []

    for batch in tqdm(loader, desc="Eval", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch.get("token_type_ids")
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(device)

        logits_t, logits_n, logits_m = model(
            input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
        )

        if head_type == "coral":
            preds_t.append(coral_predict(logits_t).cpu().numpy())
            preds_n.append(coral_predict(logits_n).cpu().numpy())
            preds_m.append(binary_predict(logits_m).cpu().numpy())
        else:
            preds_t.append(logits_t.argmax(1).cpu().numpy())
            preds_n.append(logits_n.argmax(1).cpu().numpy())
            preds_m.append(logits_m.argmax(1).cpu().numpy())

        trues_t.append(batch["labels_t"].numpy())
        trues_n.append(batch["labels_n"].numpy())
        trues_m.append(batch["labels_m"].numpy())
        masks_t.append(batch["mask_t"].numpy())
        masks_n.append(batch["mask_n"].numpy())
        masks_m.append(batch["mask_m"].numpy())

    return compute_metrics(
        np.concatenate(preds_t), np.concatenate(preds_n), np.concatenate(preds_m),
        np.concatenate(trues_t), np.concatenate(trues_n), np.concatenate(trues_m),
        np.concatenate(masks_t).astype(bool),
        np.concatenate(masks_n).astype(bool),
        np.concatenate(masks_m).astype(bool),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--encoder", default=DEFAULT_ENCODER)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--head-lr", type=float, default=None,
                        help="Separate LR for classification heads (default: same as --lr)")
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--meta-dir", default="TCGA_Metadata",
                        help="Directory with TCGA metadata CSVs for val label enrichment")
    parser.add_argument("--seed", type=int, default=0)
    # Head type
    parser.add_argument("--head-type", choices=["ce", "coral"], default="ce",
                        help="Classification head type: ce (cross-entropy) or coral (ordinal)")
    parser.add_argument("--no-class-weights-m", action="store_true")
    # LoRA args (set --lora-r 0 to disable)
    parser.add_argument("--lora-r", type=int, default=0)
    parser.add_argument("--lora-alpha", type=int, default=DEFAULT_LORA_ALPHA)
    parser.add_argument("--lora-dropout", type=float, default=DEFAULT_LORA_DROPOUT)
    parser.add_argument("--lora-targets", nargs="+", default=None)
    # W&B
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="tnm-staging")
    parser.add_argument("--wandb-run-name", default=None)
    # Resume
    parser.add_argument("--resume", default=None, metavar="CHECKPOINT")
    cli_args = sys.argv[1:]
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    start_time = datetime.now()
    start_time_str = start_time.strftime("%Y%m%d_%H%M")
    if not is_output_dir_explicit(cli_args):
        args.output_dir = os.path.join(args.output_dir, start_time_str)

    logger.info("Using device: %s", device)
    logger.info("Output directory: %s", args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    # Save train config early so it's available even if training is interrupted
    with open(os.path.join(args.output_dir, "train_config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    wandb_run = None
    if args.wandb:
        import wandb
        wandb_run = wandb.init(
            project=args.wandb_project, name=args.wandb_run_name, config=vars(args),
        )
        logger.info("W&B run: %s", wandb_run.url)

    # ---- Data ----
    train_df, val_df = load_data(args.data_dir, args.meta_dir)

    tokenizer = AutoTokenizer.from_pretrained(args.encoder)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    enc_train = tokenizer(
        train_df["text"].astype(str).tolist(), padding=True, truncation=True,
        max_length=args.max_length, return_tensors="np",
    )
    enc_val = tokenizer(
        val_df["text"].astype(str).tolist(), padding=True, truncation=True,
        max_length=args.max_length, return_tensors="np",
    )

    train_ds = TNMDataset(enc_train, train_df["t"].values, train_df["n"].values, train_df["m"].values)
    val_ds = TNMDataset(enc_val, val_df["t"].values, val_df["n"].values, val_df["m"].values)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=args.eval_batch_size, shuffle=False, num_workers=0)

    # ---- Loss (CE mode only — CORAL computes loss inline) ----
    criterion_t, criterion_n, criterion_m = None, None, None
    if args.head_type == "ce":
        valid_m = train_df["m"].values
        valid_m = valid_m[valid_m >= 0]
        if not args.no_class_weights_m and len(valid_m) > 0:
            m_counts = np.bincount(valid_m, minlength=M_NUM_LABELS)
            m_weights = 1.0 / (m_counts + 1e-6)
            m_weights = m_weights / m_weights.sum() * M_NUM_LABELS
            weight_m = torch.tensor(m_weights, dtype=torch.float32).to(device)
        else:
            weight_m = None
        criterion_t = nn.CrossEntropyLoss()
        criterion_n = nn.CrossEntropyLoss()
        criterion_m = nn.CrossEntropyLoss(weight=weight_m)

    # ---- Model ----
    torch_dtype = torch.bfloat16 if args.lora_r > 0 else torch.float32
    model = TNMClassifier(
        encoder_name=args.encoder,
        t_num_labels=T_NUM_LABELS,
        n_num_labels=N_NUM_LABELS,
        m_num_labels=M_NUM_LABELS,
        dropout=0.1,
        head_type=args.head_type,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_targets=args.lora_targets,
        torch_dtype=torch_dtype,
    ).to(device)

    # Optimizer
    head_lr = args.head_lr or args.lr
    if args.lora_r > 0 and args.head_lr:
        backbone_params = [p for n, p in model.named_parameters()
                           if p.requires_grad and "head" not in n]
        head_params = [p for n, p in model.named_parameters()
                       if p.requires_grad and "head" in n]
        optimizer = torch.optim.AdamW([
            {"params": backbone_params, "lr": args.lr, "weight_decay": args.weight_decay},
            {"params": head_params, "lr": head_lr, "weight_decay": 0.0},
        ])
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Scheduler: linear warmup + cosine decay
    updates_per_epoch = max(1, (len(train_loader) + args.grad_accum_steps - 1) // args.grad_accum_steps)
    total_steps = updates_per_epoch * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    start_epoch = 0
    best_f1 = 0.0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        start_epoch = ckpt.get("epoch", 0) + 1
        best_f1 = ckpt.get("metrics", {}).get("f1_macro_avg", 0.0)
        logger.info("Resumed from %s (epoch %d, best_f1=%.4f)", args.resume, start_epoch, best_f1)

    # ---- Training loop ----
    train_history = []
    best_metrics = None
    initial_step = start_epoch * updates_per_epoch
    train_pbar = tqdm(total=total_steps, initial=initial_step, desc="Train", leave=True)
    for epoch in range(start_epoch, args.epochs):
        loss_avg = train_epoch(
            model, train_loader, optimizer, scheduler, device,
            args.grad_accum_steps, args.head_type,
            criterion_t, criterion_n, criterion_m,
            train_pbar,
        )
        metrics = evaluate(model, val_loader, device, args.head_type)
        logger.info(
            "Epoch %d  loss=%.4f  F1_T=%.4f  F1_N=%.4f  F1_M=%.4f  F1_avg=%.4f  exact=%.4f",
            epoch + 1, loss_avg,
            metrics["f1_t"], metrics["f1_n"], metrics["f1_m"],
            metrics["f1_macro_avg"], metrics["exact_match"],
        )
        if wandb_run is not None:
            wandb_run.log({
                "epoch": epoch + 1,
                "train/loss": loss_avg,
                **{f"val/{k}": v for k, v in metrics.items()},
            })

        train_history.append({
            "epoch": epoch + 1,
            "train_loss": float(loss_avg),
            "lr": float(optimizer.param_groups[0]["lr"]),
            **{k: float(v) if isinstance(v, (int, float, np.floating)) else v for k, v in metrics.items()},
        })

        # Evaluate on training set for overfitting analysis
        train_metrics = evaluate(model, train_loader, device, args.head_type)
        logger.info(
            "Epoch %d  TRAIN  F1_T=%.4f  F1_N=%.4f  F1_M=%.4f  F1_avg=%.4f  exact=%.4f",
            epoch + 1,
            train_metrics["f1_t"], train_metrics["f1_n"], train_metrics["f1_m"],
            train_metrics["f1_macro_avg"], train_metrics["exact_match"],
        )
        if wandb_run is not None:
            wandb_run.log({
                **{f"train/{k}": v for k, v in train_metrics.items()},
            }, commit=False)

        train_history[-1].update({
            f"train_{k}": float(v) if isinstance(v, (int, float, np.floating)) else v
            for k, v in train_metrics.items()
        })

        # Save per-epoch checkpoint with separate head weights
        epoch_dir = os.path.join(args.output_dir, f"epoch_{epoch + 1}")
        os.makedirs(epoch_dir, exist_ok=True)
        if model.use_lora:
            full_state = model.get_trainable_state_dict()
        else:
            full_state = model.state_dict()
        save_dict = {"epoch": epoch, "metrics": metrics, "train_metrics": train_metrics,
                     "model_state_dict": full_state}
        torch.save(save_dict, os.path.join(epoch_dir, "checkpoint.pt"))
        # Save individual head weights
        for head_name in ("head_t", "head_n", "head_m"):
            head_state = {k: v for k, v in full_state.items() if k.startswith(head_name)}
            if head_state:
                torch.save(head_state, os.path.join(epoch_dir, f"{head_name}.pt"))
        logger.info("Saved epoch %d checkpoint to %s", epoch + 1, epoch_dir)

        if metrics["f1_macro_avg"] > best_f1:
            best_f1 = metrics["f1_macro_avg"]
            best_metrics = metrics
            torch.save(save_dict, os.path.join(args.output_dir, "best.pt"))
    train_pbar.close()

    if train_history:
        pd.DataFrame(train_history).to_csv(os.path.join(args.output_dir, "train_metrics.csv"), index=False)

    run_metadata = {
        "start_time": start_time.isoformat(),
        "end_time": datetime.now().isoformat(),
        "timestamp_dir": start_time_str,
        "command": " ".join([sys.executable] + sys.argv),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "transformers_version": __import__("transformers").__version__,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "git_commit": get_git_commit_hash(),
    }

    reproducibility = {
        "args": vars(args),
        "data": {
            "train_samples": len(train_df),
            "val_samples": len(val_df),
            "valid_labels": {
                "train": {
                    "t": int((train_df["t"] >= 0).sum()),
                    "n": int((train_df["n"] >= 0).sum()),
                    "m": int((train_df["m"] >= 0).sum()),
                },
                "val": {
                    "t": int((val_df["t"] >= 0).sum()),
                    "n": int((val_df["n"] >= 0).sum()),
                    "m": int((val_df["m"] >= 0).sum()),
                },
            },
        },
        "training": {
            "updates_per_epoch": updates_per_epoch,
            "total_steps": total_steps,
            "warmup_steps": warmup_steps,
            "effective_batch_size": args.batch_size * args.grad_accum_steps,
            "best_f1_macro_avg": float(best_f1),
            "best_metrics": best_metrics,
        },
        "run_metadata": run_metadata,
    }

    with open(os.path.join(args.output_dir, "train_config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    with open(os.path.join(args.output_dir, "reproducibility.json"), "w") as f:
        json.dump(reproducibility, f, indent=2)
    logger.info("Best f1_macro_avg=%.4f saved to %s/best.pt", best_f1, args.output_dir)
    if wandb_run is not None:
        wandb_run.finish()
    return 0


if __name__ == "__main__":
    sys.exit(main())
