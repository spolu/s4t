import typing

from prooftrace.prooftrace import ProofTraceActions, Action, PREPARE_TOKENS

from prooftrace.repl.fusion import Thm
from prooftrace.repl.repl import REPL

from utils.config import Config


class Search:
    def __init__(
            self,
            config: Config,
            ptra: ProofTraceActions,
            repl: REPL,
            target: Thm,
    ) -> None:
        self._config = config
        self._target = target

    def step(
            self,
            offset: int = 0,
            conclusion: bool = False,
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

        index = len(actions)-1
        assert index < self._config.get('prooftrace_sequence_length')

        empty = ptra.actions()[1]
        assert empty.value == PREPARE_TOKENS['EMPTY']

        extract = Action.from_action('EXTRACT', empty, empty)

        while len(actions) < self._config.get('prooftrace_sequence_length'):
            actions.append(extract)
        while len(arguments) < self._config.get('prooftrace_sequence_length'):
            arguments.append(empty)

        return index, actions, arguments
