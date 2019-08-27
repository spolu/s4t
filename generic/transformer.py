import torch.nn as nn

from generic.gelu import GeLU


class TransformerBlock(nn.Module):
    def __init__(
            self,
            sequence_max_length,
            hidden_size,
            attention_head_count,
            dropout=0.1,
            attn_mask=None,
    ):
        super(TransformerBlock, self).__init__()

        self.attention = nn.MultiheadAttention(
            hidden_size, attention_head_count, dropout=dropout,
        )
        self.attention_layer_norm = nn.LayerNorm(hidden_size, eps=1e-12)

        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, 4*hidden_size),
            GeLU(),
            nn.Linear(4*hidden_size, hidden_size),
        )
        self.mlp_layer_norm = nn.LayerNorm(hidden_size, eps=1e-12)

        self._attn_mask = attn_mask

    def forward(
            self,
            input_tensor,
    ):
        h = self.attention_layer_norm(input_tensor)
        x, _ = self.attention(
            h, h, h,
            attn_mask=self._attn_mask,
            need_weights=False)
        h = x + h

        h = self.mlp_layer_norm(h)
        x = self.mlp(h)
        h = x + h

        return h
