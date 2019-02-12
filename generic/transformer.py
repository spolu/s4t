import math
import torch
import torch.nn as nn

from generic.layer_norm import LayerNorm
from generic.gelu import GeLU


class SelfAttention(nn.Module):
    def __init__(
            self,
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

        self.dropout = nn.Dropout(dropout)

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
        attention_probs = self.dropout(attention_probs)

        context = torch.matmul(attention_probs, value)
        context = context.permute(0, 2, 1, 3).contiguous()
        new_context_shape = context.size()[:-2] + (self.hidden_size,)

        context = context.view(*new_context_shape)

        return context


class SelfOutput(nn.Module):
    def __init__(
            self,
            hidden_size,
            dropout,
    ):
        super(SelfOutput, self).__init__()

        self.dense = nn.Linear(hidden_size, hidden_size)
        self.layer_norm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(
            self,
            hidden_states,
            input_tensor,
    ):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.layer_norm(hidden_states + input_tensor)

        return hidden_states


class Attention(nn.Module):
    def __init__(
            self,
            hidden_size,
            attention_head_count,
            dropout,
    ):
        super(Attention, self).__init__()
        self.self = SelfAttention(hidden_size, attention_head_count, dropout)
        self.output = SelfOutput(hidden_size, dropout)

    def forward(
            self,
            input_tensor,
    ):
        hidden_states = self.self(input_tensor)
        attention_output = self.output(hidden_states, input_tensor)

        return attention_output


class Transformer(nn.Module):
    def __init__(
            self,
            hidden_size,
            attention_head_count,
            intermediate_size,
            dropout=0.1
    ):
        super(Transformer, self).__init__()
        self.attention = Attention(hidden_size, attention_head_count, dropout)

        self.intermediate = nn.Linear(hidden_size, intermediate_size)
        self.dense = nn.Linear(intermediate_size, hidden_size)

        self.layer_norm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.gelu = GeLU()

        self.apply(self.init_weights)

    def init_weights(
            self,
            module,
    ):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, LayerNorm):
            module.beta.data.normal_(mean=0.0, std=0.02)
            module.gamma.data.normal_(mean=0.0, std=0.02)

    def forward(
            self,
            input_tensor,
    ):
        attention_output = self.attention(input_tensor)
        intermediate_output = self.gelu(self.intermediate(attention_output))
        block_output = self.layer_norm(
            attention_output + self.dropout(self.dense(intermediate_output))
        )

        return block_output
