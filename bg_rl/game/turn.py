from abc import ABC, abstractmethod

class Turn(ABC):

    def __init__(self, player):
        self.player = player
        self.decisions = []