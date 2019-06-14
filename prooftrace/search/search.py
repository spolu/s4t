import typing

from prooftrace.prooftrace import ProofTraceActions, Action

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

    def preprocess_ptra(
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
