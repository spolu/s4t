import os
import torch
import typing

from prooftrace.prooftrace import Action, ProofTraceActions

from prooftrace.repl.fusion import Thm
from prooftrace.repl.repl import REPL

from prooftrace.models.embedder import E
from prooftrace.models.heads import PH, VH
from prooftrace.models.torso import T

from utils.config import Config
from utils.log import Log


class SearchModel:
    def __init__(
            self,
            config: Config,
            modules: typing.Dict[str, torch.nn.Module] = None,
    ):
        self._config = config

        self._device = torch.device(config.get('device'))

        if modules is not None:
            assert 'E' in modules
            assert 'T' in modules
            assert 'PH' in modules
            assert 'VH' in modules

            self._modules = modules
        else:
            self._modules = {
                'E': E(self._config).to(self._device),
                'T': T(self._config).to(self._device),
                'PH': PH(self._config).to(self._device),
                'VH': VH(self._config).to(self._device),
            }

    def load(
            self,
    ):
        load_dir = self._config.get('prooftrace_load_dir')

        if load_dir:
            Log.out(
                "Loading prooftrace LM", {
                    'load_dir': load_dir,
                })
            if os.path.isfile(load_dir + "/model_E.pt"):
                self._modules['E'].load_state_dict(
                    torch.load(
                        load_dir + "/model_E.pt",
                        map_location=self._device,
                    ),
                )
            if os.path.isfile(load_dir + "/model_T.pt"):
                self._modules['T'].load_state_dict(
                    torch.load(
                        load_dir + "/model_T.pt",
                        map_location=self._device,
                    ),
                )
            if os.path.isfile(load_dir + "/model_PH.pt"):
                self._modules['PH'].load_state_dict(
                    torch.load(
                        load_dir + "/model_PH.pt",
                        map_location=self._device,
                    ),
                )
            if os.path.isfile(load_dir + "/model_VH.pt"):
                self._modules['VH'].load_state_dict(
                    torch.load(
                        load_dir + "/model_VH.pt",
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

    def value(
            self,
            idx: typing.List[typing.List[int]],
            act: typing.List[typing.List[Action]],
            arg: typing.List[typing.List[Action]],
    ) -> torch.Tensor:
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

            prd_values = self._modules['VH'](heads, targets)

            return prd_values


class Search:
    def __init__(
            self,
            config: Config,
            model: SearchModel,
            ptra: ProofTraceActions,
            repl: REPL,
            target: Thm,
    ) -> None:
        self._config = config
        self._model = model
        self._target = target

    def step(
            self,
            final: bool = False,
            offset: int = 0,
    ) -> typing.Tuple[
        bool, typing.Optional[ProofTraceActions], bool,
    ]:
        raise Exception('Not implemented')
