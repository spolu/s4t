import os
import torch
import torch.nn as nn
import typing

from prooftrace.prooftrace import Action

from prooftrace.models.embedder import E
from prooftrace.models.heads import PH, VH
from prooftrace.models.torso import T

from utils.config import Config
from utils.log import Log


class LModel:
    def __init__(
            self,
            config: Config,
            modules: typing.Dict[str, torch.nn.Module] = None,
    ):
        self._config = config

        self._device = torch.device(config.get('device'))

        if modules is not None:
            assert 'pE' in modules
            assert 'pT' in modules
            assert 'pH' in modules
            self._modules = modules
        else:
            self._modules = {
                'pE': E(self._config).to(self._device),
                'pT': T(self._config).to(self._device),
                'pH': PH(self._config).to(self._device),
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
                Log.out(
                    "Loading pE", {
                        'load_dir': load_dir,
                    })
                self._modules['pE'].load_state_dict(
                    torch.load(
                        load_dir + "/model_pE.pt",
                        map_location=self._device,
                    ),
                )
            if os.path.isfile(load_dir + "/model_pT.pt"):
                Log.out(
                    "Loading pT", {
                        'load_dir': load_dir,
                    })
                self._modules['pT'].load_state_dict(
                    torch.load(
                        load_dir + "/model_pT.pt",
                        map_location=self._device,
                    ),
                )
            if os.path.isfile(load_dir + "/model_pH.pt"):
                Log.out(
                    "Loading pH", {
                        'load_dir': load_dir,
                    })
                self._modules['pH'].load_state_dict(
                    torch.load(
                        load_dir + "/model_pH.pt",
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
                self._modules['pE'].state_dict(),
                save_dir + "/model_pE.pt",
            )
            torch.save(
                self._modules['pT'].state_dict(),
                save_dir + "/model_pT.pt",
            )
            torch.save(
                self._modules['pH'].state_dict(),
                save_dir + "/model_pH.pt",
            )

    def eval(
            self,
    ):
        for m in self._modules:
            self._modules[m].eval()

    def train(
            self,
    ):
        for m in self._modules:
            self._modules[m].train()

    def infer(
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


class VModel:
    def __init__(
            self,
            config: Config,
            modules: typing.Dict[str, torch.nn.Module] = None,
    ):
        self._config = config

        self._device = torch.device(config.get('device'))

        if modules is not None:
            assert 'vE' in modules
            assert 'vT' in modules
            assert 'vH' in modules
            self._modules = modules
        else:
            self._modules = {
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
                self._modules['vE'].state_dict(),
                save_dir + "/model_vE.pt",
            )
            torch.save(
                self._modules['vT'].state_dict(),
                save_dir + "/model_vT.pt",
            )
            torch.save(
                self._modules['vH'].state_dict(),
                save_dir + "/model_vH.pt",
            )

    def eval(
            self,
    ):
        for m in self._modules:
            self._modules[m].eval()

    def train(
            self,
    ):
        for m in self._modules:
            self._modules[m].train()

    def infer(
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
