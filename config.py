"""
BERT Configuration.

Parameters match those from the original BERT paper:
    Devlin et al., "BERT: Pre-training of Deep Bidirectional Transformers
    for Language Understanding", 2019. https://arxiv.org/abs/1810.04805
"""

from dataclasses import dataclass


@dataclass
class BertConfig:
    # ------------------------------------------------------------------ #
    # Vocabulary / Embedding
    # ------------------------------------------------------------------ #
    # Set to tokenizer.vocab_size after the tokenizer is loaded.
    vocab_size: int = 30_522           # bert-base-uncased exact size
    hidden_size: int = 768             # H  (BERT-Base)
    num_hidden_layers: int = 12        # L
    num_attention_heads: int = 12      # A
    intermediate_size: int = 3072      # 4 * H (FFN inner dimension)
    hidden_act: str = "gelu"           # activation in FFN
    max_position_embeddings: int = 512 # maximum sequence length
    type_vocab_size: int = 1           # single segment (no NSP)

    # ------------------------------------------------------------------ #
    # Regularisation
    # ------------------------------------------------------------------ #
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    layer_norm_eps: float = 1e-12

    # ------------------------------------------------------------------ #
    # Initialisation
    # ------------------------------------------------------------------ #
    initializer_range: float = 0.02

    # ------------------------------------------------------------------ #
    # Special tokens (aligned with BertTokenizer defaults)
    # ------------------------------------------------------------------ #
    pad_token_id: int = 0

    # ------------------------------------------------------------------ #
    # Pre-training objectives
    # ------------------------------------------------------------------ #
    mlm_probability: float = 0.15      # 15 % of tokens are masked

    # ------------------------------------------------------------------ #
    # Training phases (from Appendix A of the paper)
    # Phase 1 — 90 % of steps, seq_len = 128  (fast, captures most signal)
    # Phase 2 — 10 % of steps, seq_len = 512  (fine-tunes on full context)
    # ------------------------------------------------------------------ #
    phase1_max_seq_length: int = 128
    phase2_max_seq_length: int = 512
    phase1_ratio: float = 0.9          # fraction of total steps in phase 1

    # ------------------------------------------------------------------ #
    # Optimiser hyper-parameters (Table 1 / Appendix A)
    # ------------------------------------------------------------------ #
    learning_rate: float = 1e-4
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-6
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    num_train_steps: int = 1_000_000
    warmup_steps: int = 10_000
    train_batch_size: int = 256        # sequences per step (not per GPU)

    # ------------------------------------------------------------------ #
    # MDLM / Diffusion pre-training settings
    # Only used when training_objective = "mdlm"
    # ------------------------------------------------------------------ #
    training_objective: str   = "standard" # "standard" (MLM+NSP) | "mdlm" (diffusion)
    noise_schedule:     str   = "linear"   # "linear" | "cosine"
    time_epsilon:       float = 1e-3       # min timestep — avoids degenerate masking near t=0
    loss_norm_type:     str   = "token"    # "token" | "sequence" | "batch"

    def __post_init__(self):
        assert self.hidden_size % self.num_attention_heads == 0, (
            f"hidden_size ({self.hidden_size}) must be divisible by "
            f"num_attention_heads ({self.num_attention_heads})"
        )
        assert self.training_objective in ("standard", "mdlm"), (
            f"training_objective must be 'standard' or 'mdlm', got '{self.training_objective}'"
        )
        assert self.noise_schedule in ("linear", "cosine"), (
            f"noise_schedule must be 'linear' or 'cosine', got '{self.noise_schedule}'"
        )
