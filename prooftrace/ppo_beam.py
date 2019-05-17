import argparse
import datetime
import gzip
import math
import os
import pickle
import random
import re
import torch
import typing

from prooftrace.prooftrace import \
    ACTION_TOKENS, PREPARE_TOKENS, INV_ACTION_TOKENS, INV_PREPARE_TOKENS, \
    Action, ProofTraceActions

from prooftrace.models.embedder import E
from prooftrace.models.heads import PH, VH
from prooftrace.models.torso import T

from prooftrace.repl.repl import REPL
from prooftrace.repl.fusion import Thm

from utils.config import Config
from utils.log import Log


class Model:
    def __init__(
            self,
            config: Config,
    ):
        self._config = config

        self._device = torch.device(config.get('device'))

        self._load_dir = config.get('prooftrace_load_dir')

        self._modules = {
            'E': E(self._config).to(self._device),
            'T': T(self._config).to(self._device),
            'PH': PH(self._config).to(self._device),
            'VH': VH(self._config).to(self._device),
        }

    def load(
            self,
    ):
        if self._load_dir:
            Log.out(
                "Loading prooftrace LM", {
                    'load_dir': self._load_dir,
                })
            if os.path.isfile(self._load_dir + "/model_E.pt"):
                self._modules['E'].load_state_dict(
                    torch.load(
                        self._load_dir + "/model_E.pt",
                        map_location=self._device,
                    ),
                )
            if os.path.isfile(self._load_dir + "/model_T.pt"):
                self._modules['T'].load_state_dict(
                    torch.load(
                        self._load_dir + "/model_T.pt",
                        map_location=self._device,
                    ),
                )
            if os.path.isfile(self._load_dir + "/model_PH.pt"):
                self._modules['PH'].load_state_dict(
                    torch.load(
                        self._load_dir + "/model_PH.pt",
                        map_location=self._device,
                    ),
                )
            if os.path.isfile(self._load_dir + "/model_VH.pt"):
                self._modules['VH'].load_state_dict(
                    torch.load(
                        self._load_dir + "/model_VH.pt",
                        map_location=self._device,
                    ),
                )

        self._modules['E'].eval()
        self._modules['T'].eval()
        self._modules['PH'].eval()
        self._modules['VH'].eval()

        return self

    def infer(
            self,
            idx: typing.List[typing.List[int]],
            act: typing.List[typing.List[Action]],
            arg: typing.List[typing.List[Action]],
    ) -> typing.Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor,
        torch.Tensor,
    ]:
        with torch.no_grad():
            action_embeds = self._modules['E'](act)
            argument_embeds = self._modules['E'](arg)

            hiddens = self._modules['T'](action_embeds, argument_embeds)

            heads = torch.cat([
                hiddens[i][idx[i]].unsqueeze(0) for i in range(len(idx))
            ], dim=0)
            targets = torch.cat([
                action_embeds[i][0].unsqueeze(0) for i in range(len(idx))
            ], dim=0)

            prd_actions, prd_lefts, prd_rights = \
                self._modules['PH'](heads, hiddens, targets)
            prd_values = self._modules['VH'](heads, targets)

            return (
                prd_actions, prd_lefts, prd_rights,
                prd_values,
            )


class Candidate:
    def __init__(
            self,
            prd_actions: torch.Tensor,
            prd_lefts: torch.Tensor,
            prd_rights: torch.Tensor,
            value: float,
    ) -> None:
        self._prd_actions = prd_actions
        self._prd_lefts = prd_lefts
        self._prd_rights = prd_rights
        self._value = value


class Beam:
    def __init__(
            self,
            config: Config,
            model: Model,
            repl: REPL,
            ptra: ProofTraceActions,
            theorem: Thm,
    ) -> None:
        self._config = config

        self._width = config.get('prooftrace_beam_width')

        self._model = model

        self._ptras = [ptra.copy() for _ in range(self._width)]
        self._repls = [repl.copy() for _ in range(self._width)]

        index, actions, arguments = self.process_ptra(ptra)
        prd_actions, prd_lefts, prd_rights, prd_values = \
            self._model.infer([index], [actions], [arguments])

        self._head = [
            Candidate(
                prd_actions[0].cpu(),
                prd_lefts[0].cpu(),
                prd_rights[0].cpu(),
                prd_values[0].cpu().item(),
            )
        ]

    def process_ptra(
            self,
            ptra: ProofTraceActions,
    ) -> typing.Tuple[
        int, typing.List[Action], typing.List[Action],
    ]:
        actions = ptra.actions().copy()
        arguments = ptra.arguments().copy()

        index = len(actions)

        empty = Action.from_action('EMPTY', None, None)
        while len(actions) < self._config.get('prooftrace_sequence_length'):
            actions.append(empty)
        while len(arguments) < self._config.get('prooftrace_sequence_length'):
            arguments.append(empty)

        return index, actions, arguments

    def step(
            self,
    ) -> None:
        pass

    def candidates(
            self,
            model: Model,
            beta_width: int,
            sequence_length: int,
    ):
        idx = []
        act = []
        arg = []

        for i in range(self._width):
            actions = self._ptras[i].actions().copy()
            arguments = self._ptras[i].arguments().copy()

            idx.append(len(actions))

            empty = Action.from_action('EMPTY', None, None)
            while len(actions) < sequence_length:
                actions.append(empty)
            while len(arguments) < sequence_length:
                arguments.append(empty)

            act.append(actions)
            arg.append(arguments)

        prd_actions, prd_lefts, prd_rights, prd_values = \
            model.infer(idx, act, arg)

        a_count = min(
            beta_width,
            len(ACTION_TOKENS) - len(PREPARE_TOKENS),
        )
        top_actions = torch.exp(prd_actions.cpu()).topk(a_count)
        top_lefts = torch.exp(prd_lefts.cpu()).topk(beta_width)
        top_rights = torch.exp(prd_rights.cpu()).topk(beta_width)

        candidates = []

        for ia in range(a_count):
            for il in range(beta_width):
                for ir in range(beta_width):

                    action = top_actions[1][ia].item()
                    left = top_lefts[1][il].item()
                    right = top_rights[1][ir].item()

                    if left >= self._ptra.len() or right >= self._ptra.len():
                        continue

                    a = Action.from_action(
                        INV_ACTION_TOKENS[action + len(PREPARE_TOKENS)],
                        self._ptra.arguments()[left],
                        self._ptra.arguments()[right],
                    )

                    if self._ptra.seen(a):
                        continue

                    if not self._repl.valid(a):
                        continue

                    candidates.append((a,
                                       top_actions[0][ia].item() *
                                       top_lefts[0][il].item() *
                                       top_rights[0][ir].item()))


