from .agent import Agent
from ..mcts import MCTS
from ..utilities.timing import time_func

class MCTSAgent(Agent):

    def __init__(self):
        super().__init__()
        self.mcts = None

    def make_new_tree(self, game, num_players):
        self.mcts = MCTS(game, num_players)

    @time_func
    def select_next_action(self, decision, game):
        return self.mcts.explore_and_get_best_action(decision, game)


    