import torch.nn as nn


class Upsample(nn.Module):
    def __init__(
            self,
            hidden_size,
            kernel_size=4,
            stride=2,
    ):
        super(Upsample, self).__init__()

        self.conv_transpose = nn.ConvTranspose1d(
            hidden_size, hidden_size,
            kernel_size, stride,
        )

    def forward(
            self,
            input_tensor,
    ):
        x = input_tensor.transpose(1, 2)
        x = self.conv_transpose(x)
        return x.transpose(1, 2)


class Downsample(nn.Module):
    def __init__(
            self,
            hidden_size,
            kernel_size=4,
            stride=2,
    ):
        super(Downsample, self).__init__()

        self.conv = nn.Conv1d(
            hidden_size, hidden_size,
            kernel_size, stride,
        )

    def forward(
            self,
            input_tensor,
    ):
        x = input_tensor.transpose(1, 2)
        x = self.conv(x)
        return x.transpose(1, 2)
