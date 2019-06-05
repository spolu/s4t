import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader

DEVICE = 'cpu'
BATCH_SIZE = 32
NUM_WORKERS = 8

OUTPUT_SIZE = 100
INPUT_SIZE = 3072
CORE_SIZE = 1024

POPULATION = 32
EPSILON_SCALE = 2
ITERATIONS = 8


class EQI(nn.Module):
    def __init__(
            self,
    ):
        super(EQI, self).__init__()

        self._weights = {
            'itoc': torch.zeros(
                [INPUT_SIZE, CORE_SIZE],
                dtype=torch.int8,
                device=torch.device(DEVICE),
            ),
            'itoc_b': torch.zeros(
                [CORE_SIZE],
                dtype=torch.int8,
                device=torch.device(DEVICE),
            ),
            'ctoc': torch.zeros(
                [CORE_SIZE, CORE_SIZE],
                dtype=torch.int8,
                device=torch.device(DEVICE),
            ),
            'ctoc_b': torch.zeros(
                [CORE_SIZE],
                dtype=torch.int8,
                device=torch.device(DEVICE),
            ),
            'ctoo': torch.zeros(
                [CORE_SIZE, OUTPUT_SIZE],
                dtype=torch.int8,
                device=torch.device(DEVICE),
            ),
            'ctoo_b': torch.zeros(
                [OUTPUT_SIZE],
                dtype=torch.int8,
                device=torch.device(DEVICE),
            )
        }

    def rand(
            self,
            size,
            scale,
    ):
        return ((torch.rand(size) * 2 - 1) * scale).to(
            dtype=torch.int8,
            device=torch.device(DEVICE),
        )

    def epsilon(
            self,
    ):
        return {
            'itoc': self.rand([INPUT_SIZE, CORE_SIZE], EPSILON_SCALE),
            'itoc_b': self.rand([CORE_SIZE], EPSILON_SCALE),
            'ctoc': self.rand([CORE_SIZE, CORE_SIZE], EPSILON_SCALE),
            'ctoc_b': self.rand([CORE_SIZE], EPSILON_SCALE),
            'ctoo': self.rand([CORE_SIZE, OUTPUT_SIZE], EPSILON_SCALE),
            'ctoo_b': self.rand([OUTPUT_SIZE], EPSILON_SCALE),
        }

    def apply(
            self,
            losses,
            epsilons,
    ):
        losses = torch.Tensor(losses)
        losses = -(losses - losses.mean()) / (losses.std() + 1e-5)

        for i in range(len(epsilons)):
            self._weights['itoc'] += losses[i] * epsilons[i]['itoc']
            self._weights['itoc_b'] += losses[i] * epsilons[i]['itoc_b']
            self._weights['ctoc'] += losses[i] * epsilons[i]['ctoc']
            self._weights['ctoc_b'] += losses[i] * epsilons[i]['ctoc_b']
            self._weights['ctoo'] += losses[i] * epsilons[i]['ctoo']
            self._weights['ctoo_b'] += losses[i] * epsilons[i]['ctoo_b']

    def forward(
            self,
            epsilon,
            inputs,
            hiddens,
    ):
        with torch.no_grad():
            inputs = torch.addmm(
                self._weights['itoc_b'] + epsilon['itoc_b'],
                inputs,
                self._weights['itoc'] + epsilon['itoc'],
            )
            hiddens = torch.addmm(
                self._weights['ctoc_b'] + epsilon['ctoc_b'],
                hiddens + inputs,
                self._weights['ctoc'] + epsilon['ctoc'],
            )
            outputs = torch.addmm(
                self._weights['ctoo_b'] + epsilon['ctoo_b'],
                hiddens,
                self._weights['ctoo'] + epsilon['ctoo'],
            )

        return hiddens, outputs


if __name__ == '__main__':
    torch.manual_seed(0)

    dataset = torchvision.datasets.CIFAR100(
        root='~/tmp/micronet/cifar100',
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
        ]),
    )

    dataloader = DataLoader(
        dataset, shuffle=True, num_workers=NUM_WORKERS, batch_size=BATCH_SIZE,
    )

    loss = nn.CrossEntropyLoss()

    model = EQI()
    model.eval()

    for idx, (images, labels) in enumerate(dataloader):
        inputs = (images * 256 - 127).reshape(
            BATCH_SIZE, INPUT_SIZE
        ).to(torch.int8)

        epsilons = [
            model.epsilon() for _ in range(POPULATION)
        ]
        hiddens = [
            torch.zeros(
                [BATCH_SIZE, CORE_SIZE],
                dtype=torch.int8,
                device=torch.device(DEVICE),
            ) for _ in range(POPULATION)
        ]
        outputs = [
            torch.zeros(
                [BATCH_SIZE, OUTPUT_SIZE],
                dtype=torch.int8,
                device=torch.device(DEVICE),
            ) for _ in range(POPULATION)
        ]

        for _ in range(ITERATIONS):
            for i in range(POPULATION):
                hiddens[i], outputs[i] = model(epsilons[i], inputs, hiddens[i])

        losses = [
            loss(outputs[i].to(torch.float32), labels).item()
            for i in range(POPULATION)
        ]

        model.apply(losses, epsilons)

        best = losses[0]
        for i in range(POPULATION):
            if best > losses[i]:
                best = losses[i]

        print("Iteration: idx={} best={}".format(idx, best))
