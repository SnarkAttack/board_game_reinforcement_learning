import sys
from abc import ABC, abstractmethod

from .action import Action
from .player import Player

class Game(ABC):

    def __init__(self):
        self.reactions = []
        self.decisions = []
        self.turns = []
        self.players = {}
        self.action_history = []

        self.curr_turn = None

    def create_players(self, agents, player_count=None):
        """
        Creates game players

        Arguments should be either list of agents, or a single agent and number of players to use that agent for
        
        """
        if isinstance(agents, list):
            for idx, agent in agents:
                self.players[idx] = Player(idx, agent)
        else:
            for idx in range(player_count):
                self.players[idx] = Player(idx, agents)
        return list(self.players.values())

    def get_next_decision(self):
        if len(self.reactions) > 0:
            return self.reactions.pop(0)
        elif len(self.decisions) > 0:
            return self.decisions.pop(0)
        else:
            self.move_to_next_turn()
            return self.get_next_decision()

    def perform_action(self, action: Action):
        action.perform(self)
        self.action_history.append(action)

    def is_player_winner(self, player):
        if player.player_id == self.winner:
            return True
        return False

    @abstractmethod
    def setup_game(self):
        raise NotImplementedError()

    @abstractmethod
    def play_game(self):
        raise NotImplementedError()
        
    @abstractmethod
    def is_done(self):
        raise NotImplementedError()

    @abstractmethod
    def move_to_next_turn(self):
        raise NotImplementedError()

    @abstractmethod
    def get_evaluation_for_player(self, player):
        raise NotImplementedError()

    def get_evaluation(self):
        return {player_id: self.get_evaluation_for_player(player) for player_id, player in self.players.items()}

