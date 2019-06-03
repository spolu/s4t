import torch
import typing

from prooftrace.prooftrace import \
    ACTION_TOKENS, PREPARE_TOKENS, INV_ACTION_TOKENS, \
    Action, ProofTraceActions

from prooftrace.repl.repl import REPL
from prooftrace.repl.fusion import Thm
from prooftrace.search_base import Search, SearchModel

from utils.config import Config
from utils.log import Log


class MCTS(Search):
    def __init__(
            self,
            config: Config,
            model: SearchModel,
            ptra: ProofTraceActions,
            repl: REPL,
            target: Thm,
    ) -> None:
        super(MCTS, self).__init__(config, model, ptra, repl, target)
