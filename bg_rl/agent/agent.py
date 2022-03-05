from abc import ABC, abstractmethod

class Agent(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def select_next_action(self, decision, game):
        raise NotImplementedError()