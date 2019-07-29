import torch
import torch.nn as nn

from generic.act import ACT
# from generic.gelu import GeLU
from generic.transformer import TransformerBlock


class T(nn.Module):
    def __init__(
            self,
            config,
    ):
        super(T, self).__init__()

        self.device = torch.device(config.get('device'))

        self.sequence_length = \
            config.get('prooftrace_sequence_length')
        self.hidden_size = \
            config.get('prooftrace_hidden_size')
        self.attention_head_count = \
            config.get('prooftrace_transformer_attention_head_count')

        self.transformer_layer_count = \
            config.get('prooftrace_transformer_layer_count')
        self.transformer_hidden_size = \
            config.get('prooftrace_transformer_hidden_size')

        self.universal_transformer_steps = \
            config.get('prooftrace_universal_transformer_steps')
        self.universal_transformer_act = \
            config.get('prooftrace_universal_transformer_act')

        self.lstm_layer_count = \
            config.get('prooftrace_transformer_layer_count')
        self.lstm_hidden_size = \
            config.get('prooftrace_lstm_hidden_size')

        self.head_hidden_size = \
            config.get('prooftrace_head_hidden_size')

        self.torso_type = \
            config.get("prooftrace_torso_type")

        # self.position_embedding = nn.Embedding(
        #     self.sequence_length, self.hidden_size
        # )

        torso = []

        if self.torso_type == "transformer":
            self.adapter_in = \
                nn.Linear(self.hidden_size, self.transformer_hidden_size)

            for _ in range(self.transformer_layer_count):
                torso += [
                    TransformerBlock(
                        self.sequence_length,
                        self.transformer_hidden_size,
                        self.attention_head_count,
                        dropout=0.0,
                    ),
                ]

            self.torso = nn.Sequential(*torso)

            self.adapter_out = nn.Linear(
                self.transformer_hidden_size, self.head_hidden_size,
            )

        if self.torso_type == "universal_transformer":
            self.adapter_in = \
                nn.Linear(self.hidden_size, self.transformer_hidden_size)
            self.inner_transformer = TransformerBlock(
                self.sequence_length,
                self.transformer_hidden_size,
                self.attention_head_count,
                dropout=0.0,
            )
            self.outer_transformer = TransformerBlock(
                self.sequence_length,
                self.transformer_hidden_size,
                self.attention_head_count,
                dropout=0.0,
            )
            self.adapter_out = nn.Linear(
                self.transformer_hidden_size, self.head_hidden_size,
            )
            if self.universal_transformer_act:
                self.act = ACT(
                    self.device,
                    self.transformer_hidden_size,
                    self.universal_transformer_steps,
                )

        if self.torso_type == "lstm":
            self.adapter_in = \
                nn.Linear(self.hidden_size, self.lstm_hidden_size)
            self.lstm = nn.LSTM(
                self.lstm_hidden_size, self.lstm_hidden_size,
                num_layers=self.lstm_layer_count, batch_first=True,
            )
            self.adapter_out = nn.Linear(
                self.lstm_hidden_size, self.head_hidden_size,
            )

    def parameters_count(
            self,
    ):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(
            self,
            action_embeds,
            argument_embeds,
    ):
        # pos_embeds = torch.arange(
        #     self.sequence_length, dtype=torch.long
        # ).to(self.device)
        # pos_embeds = pos_embeds.unsqueeze(0).expand(
        #    action_embeds.size(0), self.sequence_length,
        # )
        # pos_embeds = self.position_embedding(pos_embeds)

        # hiddens = self.torso(action_embeds + argument_embeds + pos_embeds)

        hiddens = self.adapter_in(action_embeds + argument_embeds)

        if self.torso_type == "transformer":
            hiddens = self.torso(hiddens)

        if self.torso_type == "universal_transformer":
            if self.universal_transformer_act:
                self.act(hiddens, self.inner_transformer)
            else:
                for i in range(self.universal_transformer_steps):
                    hiddens = self.inner_transformer(hiddens)
            hiddens = self.outer_transformer(hiddens)

        if self.torso_type == "lstm":
            hiddens, _ = self.lstm(hiddens)

        hiddens = self.adapter_out(hiddens)

        return hiddens
