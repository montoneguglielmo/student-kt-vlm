import torch
import torch.nn as nn
import math


class KTTransformer(nn.Module):
    def __init__(
        self,
        d_input: int,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 4,
        d_ff: int = 512,
        dropout: float = 0.1,
        max_seq_len: int = 2048,
    ):
        """
        Args:
            d_input:    Dimension D of the input embedding vectors.
            d_model:    Internal transformer dimension.
            n_heads:    Number of attention heads.
            n_layers:   Number of encoder layers.
            d_ff:       Feed-forward hidden dimension.
            dropout:    Dropout rate.
            max_seq_len: Maximum sequence length for positional encoding.
        """
        super().__init__()

        # Project input embeddings to model dimension
        self.input_proj = nn.Linear(d_input, d_model)

        # Learnable positional encoding
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # Pre-norm for more stable training
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.norm = nn.LayerNorm(d_model)

        # Binary classification head: predict score from last real token
        self.head = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, 1),
        )

    def forward(self, x, attention_mask=None):
        """
        Args:
            x:              (B, S, D) padded history embeddings.
            attention_mask:  (B, S) boolean mask, True = real token, False = padding.

        Returns:
            logits: (B,) raw logits for binary prediction.
        """
        B, S, _ = x.shape

        # Project to model dim and add positional encoding
        positions = torch.arange(S, device=x.device).unsqueeze(0)  # (1, S)
        x = self.input_proj(x) + self.pos_embedding(positions)

        # PyTorch TransformerEncoder wants the INVERSE convention:
        # True = IGNORE this position, so we flip the mask
        src_key_padding_mask = ~attention_mask if attention_mask is not None else None

        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        x = self.norm(x)

        # Extract the last real token per sample (i.e. the exercise_t embedding)
        if attention_mask is not None:
            # Sum of True values per row gives the length, minus 1 for 0-indexing
            last_idx = attention_mask.sum(dim=1) - 1  # (B,)
            x = x[torch.arange(B, device=x.device), last_idx]  # (B, d_model)
        else:
            x = x[:, -1]  # (B, d_model)

        logits = self.head(x).squeeze(-1)  # (B,)
        return logits