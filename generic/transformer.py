import math
import torch
import torch.nn as nn

from generic.gelu import GeLU


class SelfAttention(nn.Module):
    def __init__(
            self,
            sequence_max_length,
            hidden_size,
            attention_head_count,
            dropout,
    ):
        super(SelfAttention, self).__init__()

        self.hidden_size = hidden_size
        self.attention_head_count = attention_head_count

        assert self.hidden_size % self.attention_head_count == 0

        self.attention_head_size = int(
            self.hidden_size / self.attention_head_count
        )

        self.query = nn.Linear(self.hidden_size, self.hidden_size)
        self.key = nn.Linear(self.hidden_size, self.hidden_size)
        self.value = nn.Linear(self.hidden_size, self.hidden_size)

        self.attention_dropout = nn.Dropout(dropout)

        self.proj = nn.Linear(hidden_size, hidden_size)
        self.proj_dropout = nn.Dropout(dropout)

    def transpose_for_scores(
            self,
            x,
    ):
        new_x_shape = x.size()[:-1] + (
            self.attention_head_count, self.attention_head_size
        )
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
            self,
            input_tensor,
    ):
        mixed_query = self.query(input_tensor)
        mixed_key = self.key(input_tensor)
        mixed_value = self.value(input_tensor)

        query = self.transpose_for_scores(mixed_query)
        key = self.transpose_for_scores(mixed_key)
        value = self.transpose_for_scores(mixed_value)

        attention_scores = torch.matmul(query, key.transpose(-1, -2)) / \
            math.sqrt(self.attention_head_size)

        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.attention_dropout(attention_probs)

        context = torch.matmul(attention_probs, value)
        context = context.permute(0, 2, 1, 3).contiguous()
        new_context_shape = context.size()[:-2] + (self.hidden_size,)

        context = context.view(*new_context_shape)

        proj = self.proj(context)
        proj = self.proj_dropout(proj)

        return proj


class MLP(nn.Module):
    def __init__(
            self,
            hidden_size,
            dropout,
    ):
        super(MLP, self).__init__()

        self.intermediate = nn.Linear(hidden_size, 4*hidden_size)
        self.gelu = GeLU()
        self.proj = nn.Linear(4*hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(
            self,
            input_tensor,
    ):
        hidden_states = self.intermediate(input_tensor)
        hidden_states = self.gelu(hidden_states)
        output = self.proj(hidden_states)
        output = self.dropout(output)

        return output


class TransformerBlock(nn.Module):
    def __init__(
            self,
            sequence_max_length,
            hidden_size,
            attention_head_count,
            dropout=0.1
    ):
        super(TransformerBlock, self).__init__()

        self.attention = SelfAttention(
            sequence_max_length, hidden_size,
            attention_head_count, dropout
        )
        self.attention_layer_norm = nn.LayerNorm(hidden_size)

        self.mlp = MLP(
            hidden_size, dropout,
        )
        self.mlp_layer_norm = nn.LayerNorm(hidden_size)

        self.apply(self.init_weights)

    def init_weights(
            self,
            module,
    ):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(
            self,
            input_tensor,
    ):
        attention_output = self.attention(input_tensor)
        attention_output = self.attention_layer_norm(
            input_tensor + attention_output
        )

        mlp_output = self.mlp(attention_output)
        mlp_output = self.mlp_layer_norm(
            attention_output + mlp_output
        )

        return mlp_output
