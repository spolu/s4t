import typing

from prooftrace.prooftrace import ProofTraceActions

from prooftrace.models.model import Model
from prooftrace.repl.fusion import Thm
from prooftrace.repl.repl import REPL

from utils.config import Config


class Search:
    def __init__(
            self,
            config: Config,
            model: Model,
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
