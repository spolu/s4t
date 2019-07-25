import torch
import torch.nn as nn


class ACT(nn.Module):
    def __init__(
            self,
            device,
            hidden_size,
            max_steps,
            threshold=0.9,
    ):
        super(ACT, self).__init__()

        self.device = device
        self.hidden_size = hidden_size
        self.max_steps = max_steps
        self.threshold = threshold

        self.p = nn.Linear(self.hidden_size, 1)
        self.p.bias.data.fill_(1)

    def forward(
            self,
            inputs,
            fn,
    ):
        p = torch.zeros(
            inputs.shape[0], inputs.shape[1],
        ).to(self.device)

        n = torch.zeros(
            inputs.shape[0], inputs.shape[1],
        ).to(self.device)

        state = inputs

        while ((p < self.threshold) & (n < self.max_steps)).byte().any():
            # inputs = inputs + time_enc[:, :inputs.shape[1], :].type_as(inputs.data)
            p = torch.sigmoid(self.p(state)).squeeze(-1)

            running = (p <= self.threshold).float()
            n = n + running

            state = fn(state) * running + state * (1 - running)

        return state
