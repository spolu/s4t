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

        self.step_embedding = nn.Embedding(
            self.max_steps, self.hidden_size
        )

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

        step = 0
        while (
                ((p < self.threshold) & (n < self.max_steps)).byte().any() and
                step < self.max_steps
        ):
            p = torch.sigmoid(self.p(state)).squeeze(-1)

            running = (p <= self.threshold).float()
            n = n + running

            step_embeds = (torch.zeros(
                state.size(0), state.size(1),
            ) + step).long().to(self.device)
            step_embeds = self.step_embedding(step_embeds)

            running = running.unsqueeze(-1)
            state = fn(state + step_embeds) * running + state * (1 - running)

            import pdb; pdb.set_trace()

            step += 1

        return state
