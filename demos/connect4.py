from distutils.command.config import config
from bg_rl.agent.mcts_agent import MCTSAgent
from bg_rl.game import Action, Decision, Game, Player, Turn
from bg_rl.agent import MCTSAgent
import numpy as np
import matplotlib.pyplot as plt
from bg_rl.utilities.timing import time_func
from scipy.signal import convolve2d

EMPTY_SPACE = -1

class SelectSpaceAction(Action):

    def __init__(self, player, space):
        super().__init__(player)
        self.space = space

    def is_legal(self, game):
        return game.board.is_column_legal(self.space)

    def perform_action(self, game):
        game.board[self.space] = self.player.player_id
        super().perform_action(game)

    def __repr__(self):
        return f"{self.player.player_id} in space {self.space}"

class SelectSpaceDecision(Decision):

    def __init__(self, player):
        super().__init__(player)

    def get_all_possible_actions(self):
        return [SelectSpaceAction(self.player, x) for x in range(7)]

class Connect4Turn(Turn):

    def __init__(self, player):
        super().__init__(player)
        self.decisions = [SelectSpaceDecision(self.player)]

class Connect4Board():

    def __init__(self):

        self.board = np.full((6, 7), EMPTY_SPACE)

    def is_full(self):
        return np.all(sum(np.where(self.board == EMPTY_SPACE, 1, 0)) == 0)

    def is_winner(self, player_ids):
        horizontal_kernel = np.array([1, 1, 1, 1])
        vertical_kernel = np.transpose(horizontal_kernel)
        diag1_kernel = np.eye(4, dtype=np.uint8)
        diag2_kernel = np.fliplr(diag1_kernel)
        detection_kernels = [horizontal_kernel, vertical_kernel, diag1_kernel, diag2_kernel]
        for player_id in player_ids:
            for kernel in detection_kernels:
                if (convolve2d(self.board == player_id, kernel, mode="valid") == 4).any():
                    return player_id
        return EMPTY_SPACE

    def is_column_legal(self, col):
        return self.board[0][col] == EMPTY_SPACE
                

class Connect4Game(Game):

    def __init__(self):
        super().__init__()
        self.board = Connect4Board()
        self.winner = EMPTY_SPACE

    def create_players(self, agents, player_count=None):
        """
        Creates game players

        Arguments should be either list of agents, or a single agent and number of players to use that agent for
        
        """
        if isinstance(agents, list):
            for idx, agent in agents:
                self.players[idx] = Connect4Player(idx, agent)
        else:
            for idx in range(player_count):
                self.players[idx] = Connect4Player(idx, agents)
        return list(self.players.values())

    @time_func
    def setup_game(self):
        for player in sorted(self.players.values(), key=lambda x: x.player_id):
            self.turns.append(Connect4Turn(player))

    @time_func
    def play_game(self):
        while self.get_winner() == EMPTY_SPACE and not self.board_is_full():
            next_decision = self.get_next_decision()
            next_action = next_decision.determine_next_action(self)
            # print(next_action)
            next_action.perform_action(self)
            if self.is_winner():
                break

    def is_winner(self):
        winner = self.board.is_winner()
        if winner != EMPTY_SPACE:
            self.winner = winner
            return True
        return False
                
    def is_done(self):
        return self.is_winner() or self.board.is_full()

    def move_to_next_turn(self):
        if self.curr_turn is not None:
            self.turns.append(Connect4Turn(self.curr_turn.player))
        self.curr_turn = self.turns.pop(0)
        self.decisions = self.curr_turn.decisions

    def get_winner(self):
        return self.winner

    def hash_game_state(self):
        return hash(np.array2string(self.board))

    def get_evaluation_for_player(self, player):
        if self.get_winner() == EMPTY_SPACE:
            return 0
        elif player.player_id == self.get_winner():
            return 1
        else:
            return -1

class Connect4Player(Player):

    def __init__(self, player_id, agent):
        super().__init__(player_id, agent)
        assert self.player_id != EMPTY_SPACE

    def __repr__(self):
        return f"Player{self.player_id}"

if __name__ == "__main__":

    wins = {-1: 0, 0: 0, 1: 0}

    game = Connect4Game()
    num_players = 2

    p1_id = 1
    p2_id = 2

    num_games = 500
    status_check = 10

    agent = MCTSAgent()

    game.create_players(agent, num_players)

    game.setup_game()

    agent.make_new_tree(game, num_players)

    move_values = {}

    for i in range(num_games):
        game.play_game()
        wins[game.get_winner()] += 1
        for child in agent.mcts.get_children(agent.mcts.root):
            first_move = list(zip(*np.where(child['game'].board == 0)))[0]
            if move_values.get(first_move) is None:
                move_values[first_move] = []
            move_values[first_move].append(child['value'][0])
        # print(game.actions)
        game = Connect4Game()
        game.create_players(agent, num_players)
        game.setup_game()
        if (i+1)%status_check == 0:
            print(i+1)

    print(wins)
    # agent.mcts.visualize_tree()

    x = list(range(num_games))

    for board, move_value_list in move_values.items():
        plt.plot(x, move_value_list, label=board)

    plt.legend()
    plt.show()