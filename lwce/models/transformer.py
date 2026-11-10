"""Compact Transformer encoder for text classification, trained from scratch (no pretraining).

Architecture:
    token embedding (vocab_size -> d_model, padding_idx=pad_idx)
    + learned positional embedding (max_len)
    N x TransformerEncoderLayer (d_model, nhead, dim_feedforward, GELU, batch_first)
    masked mean-pool over non-pad tokens -> (B, d_model)
    LayerNorm -> Dropout -> Linear(d_model, num_classes)

Defaults: d_model=128, nhead=4, num_layers=4, dim_feedforward=256, dropout=0.1.
"""

import torch
import torch.nn as nn


class TextTransformer(nn.Module):
    """Compact from-scratch transformer classifier for (B, L) token id inputs."""

    def __init__(self, vocab_size: int, num_classes: int, pad_idx: int = 0,
                 d_model: int = 128, nhead: int = 4, num_layers: int = 4,
                 dim_feedforward: int = 256, dropout: float = 0.1,
                 max_len: int = 64):
        super().__init__()
        self.pad_idx = pad_idx
        self.max_len = max_len

        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_emb   = nn.Embedding(max_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation='gelu', batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.norm    = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(d_model, num_classes)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.token_emb.weight, std=0.02)
        with torch.no_grad():
            self.token_emb.weight[self.pad_idx].zero_()
        nn.init.normal_(self.pos_emb.weight, std=0.02)
        nn.init.normal_(self.fc.weight, std=0.02)
        nn.init.zeros_(self.fc.bias)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: (B, L) long tensor of token ids, right-padded with pad_idx.

        Returns:
            (B, num_classes) logits.
        """
        B, L = input_ids.shape
        pad_mask = (input_ids == self.pad_idx)                       # (B, L) True at pad
        pos = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, L)

        x = self.token_emb(input_ids) + self.pos_emb(pos)            # (B, L, d_model)
        x = self.dropout(x)
        x = self.encoder(x, src_key_padding_mask=pad_mask)          # (B, L, d_model)

        # masked mean-pool over real (non-pad) tokens
        keep    = (~pad_mask).unsqueeze(-1).float()                  # (B, L, 1)
        summed  = (x * keep).sum(dim=1)                              # (B, d_model)
        counts  = keep.sum(dim=1).clamp(min=1.0)                     # (B, 1)
        pooled  = summed / counts

        pooled = self.dropout(self.norm(pooled))
        return self.fc(pooled)                                       # (B, num_classes)


def build_text_transformer(vocab_size: int, num_classes: int,
                           pad_idx: int = 0, max_len: int = 64) -> TextTransformer:
    """Text transformer backbone with the paper's default size."""
    return TextTransformer(vocab_size, num_classes, pad_idx=pad_idx, max_len=max_len)
