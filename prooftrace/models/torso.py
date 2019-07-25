import torch
import torch.nn as nn

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

        self.lstm_layer_count = \
            config.get('prooftrace_transformer_layer_count')
        self.lstm_hidden_size = \
            config.get('prooftrace_lstm_hidden_size')

        self.head_hidden_size = \
            config.get('prooftrace_head_hidden_size')

        # self.position_embedding = nn.Embedding(
        #     self.sequence_length, self.hidden_size
        # )

        torso = []

        if config.get("prooftrace_torso_type") == "transformer":
            self.adapter_in = nn.Linear(
                self.hidden_size, self.transformer_hidden_size,
            )

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

        if config.get("prooftrace_torso_type") == "lstm":
            self.adapter_in = nn.Linear(
                self.hidden_size, self.lstm_hidden_size,
            )
            self.lstm = nn.LSTM(
                self.lstm_hidden_size, self.lstm_hidden_size,
                num_layers=self.lstm_layer_count, batch_first=True,
            )
            self.adapter_out = nn.Linear(
                self.lstm_hidden_size, self.head_hidden_size,
            )

        if config.get("prooftrace_torso_type") == "universal_transformer":
                pass

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

        if self._config.get("prooftrace_torso_type") == "transformer":
            hiddens = self.torso(action_embeds + argument_embeds)

        if self._config.get("prooftrace_torso_type") == "lstm":
            hiddens, _ = self.lstm(hiddens)

        hiddens = self.adapter_out(hiddens)

        return hiddens
