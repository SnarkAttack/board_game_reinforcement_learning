from abc import ABC, abstractmethod
from copy import deepcopy

class Player(ABC):

    def __init__(self, player_id, agent):
        self.player_id = player_id
        self.agent = agent

    def __hash__(self):
        return hash(self.player_id)

    def __eq__(self, other):
        return self.player_id == other.player_id

    def select_next_action(self, decision, game):
        return self.agent.select_next_action(decision, game)

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k == 'agent':
                setattr(result, k, v)
            else:
                setattr(result, k, deepcopy(v, memo))
        return result

