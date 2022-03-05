from bg_rl.agent.mcts_agent import MCTSAgent
from bg_rl.game import Action, Decision, Game, Player, Turn
from bg_rl.agent import RandomAgent, MCTSAgent
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
import sys

EMPTY_SPACE = -1

class SelectSpaceAction(Action):

    def __init__(self, player, space):
        super().__init__(player)
        self.space = space

    def is_legal(self, game):
        return game.board[self.space] == EMPTY_SPACE

    def perform(self, game):
        game.board[self.space] = self.player.player_id

    def __repr__(self):
        return f"{self.player.player_id} in space {self.space}"

class SelectSpaceDecision(Decision):

    def __init__(self, player):
        super().__init__(player)

    def get_all_possible_actions(self):
        return [SelectSpaceAction(self.player, space) for space in range(4)]

class Connect2Turn(Turn):

    def __init__(self, player):
        super().__init__(player)
        self.decisions = [SelectSpaceDecision(self.player)]

class Connect2Game(Game):

    def __init__(self):
        super().__init__()
        self.board = np.full((4,), EMPTY_SPACE)
        self.winner = EMPTY_SPACE

    @property
    def hash(self):
        return hash(str(self.board)) + sys.maxsize + 1

    def create_players(self, agents, player_count=None):
        """
        Creates game players

        Arguments should be either list of agents, or a single agent and number of players to use that agent for
        
        """
        if isinstance(agents, list):
            for idx, agent in agents:
                self.players[idx] = Connect2Player(idx, agent)
        else:
            for idx in range(player_count):
                self.players[idx] = Connect2Player(idx, agents)
        return list(self.players.values())

    def setup_game(self):
        for player in sorted(self.players.values(), key=lambda x: x.player_id):
            self.turns.append(Connect2Turn(player))

    def play_game(self):
        while self.get_winner() == EMPTY_SPACE and not self.board_is_full():
            next_decision = self.get_next_decision()
            next_action = next_decision.determine_next_action(self)
            self.perform_action(next_action)
            if self.is_winner():
                break

    def is_winner(self):
        for i in range(0, len(self.board)-1):
            if self.board[i] != EMPTY_SPACE and self.board[i] == self.board[i+1]:
                self.winner = self.board[i]
                return True
        return False

    def is_done(self):
        return self.is_winner() or self.board_is_full()

    def move_to_next_turn(self):
        if self.curr_turn is not None:
            self.turns.append(Connect2Turn(self.curr_turn.player))
        self.curr_turn = self.turns.pop(0)
        self.decisions = self.curr_turn.decisions

    def get_winner(self):
        return self.winner

    def board_is_full(self):
        return sum(np.where(self.board == EMPTY_SPACE, 1, 0)) == 0

    def hash_game_state(self):
        return hash(np.array2string(self.board))

    def get_evaluation_for_player(self, player):
        if self.get_winner() == EMPTY_SPACE:
            return 0
        elif player.player_id == self.get_winner():
            return 1
        else:
            return -1

class Connect2Player(Player):

    def __init__(self, player_id, agent):
        super().__init__(player_id, agent)
        assert self.player_id != EMPTY_SPACE

    def __repr__(self):
        return f"Player{self.player_id}"

def main(args):

    wins = {-1: 0, 0: 0, 1: 0}

    game = Connect2Game()
    num_players = 2

    num_games = args.num_games
    status_check = 50

    agent = MCTSAgent()

    game.create_players(agent, num_players)

    game.setup_game()

    agent.make_new_tree(game, num_players)

    move_values = {}

    for i in range(num_games):
        game.play_game()
        wins[game.get_winner()] += 1
        # print(agent.mcts.G.number_of_nodes())
        for child in agent.mcts.get_children(agent.mcts.root):
            first_move = list(zip(*np.where(child['game'].board == 0)))[0]
            if move_values.get(first_move) is None:
                move_values[first_move] = []
            move_values[first_move].append(child['value'][0])
        game = Connect2Game()
        game.create_players(agent, num_players)
        game.setup_game()
        if (i+1)%status_check == 0:
            print(i+1)

    print(wins)

    agent.mcts.visualize_tree()

    x = list(range(num_games))

    for board, move_value_list in move_values.items():
        plt.plot(x, move_value_list, label=board)

    plt.legend()

    plt.savefig(os.path.join('demos', 'images', f'connect2_{num_games}.png'))

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-games", "-n", type=int, default=50, help="Number of games to simulate")

    args = parser.parse_args()

    main(args)

