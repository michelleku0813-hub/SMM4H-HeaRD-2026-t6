"""Shared constants for TNM staging classification."""

# Label mappings: index <-> string
T_IDX_TO_LABEL = {0: "T1", 1: "T2", 2: "T3", 3: "T4"}
N_IDX_TO_LABEL = {0: "N0", 1: "N1", 2: "N2", 3: "N3"}
M_IDX_TO_LABEL = {0: "M0", 1: "M1"}

T_LABEL_TO_IDX = {v: k for k, v in T_IDX_TO_LABEL.items()}
N_LABEL_TO_IDX = {v: k for k, v in N_IDX_TO_LABEL.items()}
M_LABEL_TO_IDX = {v: k for k, v in M_IDX_TO_LABEL.items()}

LABEL_TO_IDX = {**T_LABEL_TO_IDX, **N_LABEL_TO_IDX, **M_LABEL_TO_IDX}

# Number of classes per task
T_NUM_LABELS = 4
N_NUM_LABELS = 4
M_NUM_LABELS = 2

# Default model
DEFAULT_ENCODER = "thomas-sounack/BioClinical-ModernBERT-large"
DEFAULT_MAX_LENGTH = 4096

# LoRA defaults
DEFAULT_LORA_R = 16
DEFAULT_LORA_ALPHA = 32
DEFAULT_LORA_DROPOUT = 0.1
DEFAULT_LORA_TARGETS = ["q_proj", "v_proj"]
