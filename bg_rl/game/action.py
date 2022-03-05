from abc import ABC, abstractmethod

class Action(ABC):

    def __init__(self, player):
        self.player = player

    def __hash__(self):
        return hash(str(vars(self)))

    def __eq__(self, other):
        return str(vars(self)) == str(vars(other))

    @abstractmethod
    def is_legal(self, game):
        raise NotImplementedError()

    @abstractmethod
    def perform(self, game):
        raise NotImplementedError()