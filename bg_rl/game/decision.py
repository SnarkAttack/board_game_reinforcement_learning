from abc import ABC, abstractmethod

class Decision(ABC):

    def __init__(self, player):
        self.player = player

    def determine_next_action(self, game):
        return self.player.select_next_action(self, game)

    @abstractmethod
    def get_all_possible_actions(self):
        raise NotImplementedError()

    def get_legal_actions(self, game):
        possible_actions = self.get_all_possible_actions()
        return [action for action in possible_actions if action.is_legal(game)]