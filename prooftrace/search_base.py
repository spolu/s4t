import os
import torch
import torch.nn as nn
import typing

from prooftrace.prooftrace import Action, ProofTraceActions

from prooftrace.repl.fusion import Thm
from prooftrace.repl.repl import REPL

from prooftrace.models.embedder import E
from prooftrace.models.heads import PH, VH
from prooftrace.models.torso import T

from utils.config import Config


class SearchModel:
    def __init__(
            self,
            config: Config,
            modules: typing.Dict[str, torch.nn.Module] = None,
    ):
        self._config = config

        self._device = torch.device(config.get('device'))
        self._type = config.get('prooftrace_search_model_type')

        if modules is not None:
            assert 'pE' in modules
            assert 'pT' in modules
            assert 'pH' in modules
            assert 'vE' in modules
            assert 'vT' in modules
            assert 'vH' in modules
            self._modules = modules
        else:
            self._modules = {
                'pE': E(self._config).to(self._device),
                'pT': T(self._config).to(self._device),
                'pH': PH(self._config).to(self._device),
                'vE': E(self._config).to(self._device),
                'vT': T(self._config).to(self._device),
                'vH': VH(self._config).to(self._device),
            }

    def modules(
            self,
    ) -> typing.Dict[str, nn.Module]:
        return self._modules

    def load(
            self,
    ):
        load_dir = self._config.get('prooftrace_load_dir')

        if load_dir:
            if os.path.isfile(load_dir + "/model_pE.pt"):
                self._modules['pE'].load_state_dict(
                    torch.load(
                        load_dir + "/model_pE.pt",
                        map_location=self._device,
                    ),
                )
            if os.path.isfile(load_dir + "/model_pT.pt"):
                self._modules['pT'].load_state_dict(
                    torch.load(
                        load_dir + "/model_pT.pt",
                        map_location=self._device,
                    ),
                )
            if os.path.isfile(load_dir + "/model_pH.pt"):
                self._modules['pH'].load_state_dict(
                    torch.load(
                        load_dir + "/model_pH.pt",
                        map_location=self._device,
                    ),
                )

            if os.path.isfile(load_dir + "/model_vE.pt"):
                self._modules['vE'].load_state_dict(
                    torch.load(
                        load_dir + "/model_vE.pt",
                        map_location=self._device,
                    ),
                )
            if os.path.isfile(load_dir + "/model_vT.pt"):
                self._modules['vT'].load_state_dict(
                    torch.load(
                        load_dir + "/model_vT.pt",
                        map_location=self._device,
                    ),
                )
            if os.path.isfile(load_dir + "/model_vH.pt"):
                self._modules['vH'].load_state_dict(
                    torch.load(
                        load_dir + "/model_vH.pt",
                        map_location=self._device,
                    ),
                )

        return self

    def save(
            self,
    ):
        save_dir = self._config.get('prooftrace_save_dir')

        if save_dir:
            torch.save(
                self._modules['E'].state_dict(),
                save_dir + "/model_E.pt",
            )
            torch.save(
                self._modules['T'].state_dict(),
                save_dir + "/model_T.pt",
            )
            torch.save(
                self._modules['PH'].state_dict(),
                save_dir + "/model_PH.pt",
            )
            torch.save(
                self._modules['VH'].state_dict(),
                save_dir + "/model_VH.pt",
            )

    def infer_values(
            self,
            idx: typing.List[typing.List[int]],
            act: typing.List[typing.List[Action]],
            arg: typing.List[typing.List[Action]],
    ) -> torch.Tensor:
        action_embeds = self._modules['vE'](act)
        argument_embeds = self._modules['vE'](arg)

        hiddens = self._modules['vT'](action_embeds, argument_embeds)

        heads = torch.cat([
            hiddens[i][idx[i]].unsqueeze(0) for i in range(len(idx))
        ], dim=0)
        targets = torch.cat([
            action_embeds[i][0].unsqueeze(0) for i in range(len(idx))
        ], dim=0)

        return self._modules['vH'](heads, targets)

    def infer_actions(
            self,
            idx: typing.List[typing.List[int]],
            act: typing.List[typing.List[Action]],
            arg: typing.List[typing.List[Action]],
    ) -> typing.Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor,
    ]:
        action_embeds = self._modules['pE'](act)
        argument_embeds = self._modules['pE'](arg)

        hiddens = self._modules['pT'](action_embeds, argument_embeds)

        heads = torch.cat([
            hiddens[i][idx[i]].unsqueeze(0) for i in range(len(idx))
        ], dim=0)
        targets = torch.cat([
            action_embeds[i][0].unsqueeze(0) for i in range(len(idx))
        ], dim=0)

        return self._modules['pH'](heads, hiddens, targets)

    def infer(
            self,
            idx: typing.List[typing.List[int]],
            act: typing.List[typing.List[Action]],
            arg: typing.List[typing.List[Action]],
    ) -> typing.Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor,
        torch.Tensor,
    ]:
        prd_values = self.infer_values(idx, act, arg)
        prd_actions, prd_lefts, prd_rights = self.infer_actions(idx, act, arg)

        return (prd_actions, prd_lefts, prd_rights, prd_values)


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
