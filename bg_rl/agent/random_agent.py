import random
from .agent import Agent

class RandomAgent(Agent):

    def __init__(self):
        super().__init__()

    def select_next_action(self, decision, game):
        legal_actions = decision.get_legal_actions(game)
        return random.choice(legal_actions)
