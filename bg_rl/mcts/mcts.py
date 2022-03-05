import networkx as nx
import matplotlib.pyplot as plt
from copy import deepcopy
from bg_rl.game import Game
import math
import random
from networkx.drawing.nx_pydot import graphviz_layout
from ..utilities.timing import time_func
import time

class MCTS():

    def __init__(self, game: Game, num_players: int):
        self.G = nx.DiGraph()
        self.num_players = num_players
        game_copy = deepcopy(game)
        self.root = self.add_game_as_node(game_copy, None)
        self.expand(self.root)
        
    def add_game_as_node(self, game: Game, parent):
        game_hash = game.hash
        # Check if node already exists and just return if it does
        # If we try to call add_node on an existing node, a new node will 
        # not be created (good) but the existing visits and values counts will
        # be removed (bad)
        try:
            return self.G.nodes[game_hash]
        except KeyError as e:
            self.G.add_node(game_hash,
                            game=game,
                            hash=game_hash,
                            level=parent['level']+1 if parent is not None else 0,
                            visits=0,
                            value={i: 0 for i in range(self.num_players)},
                            t={i: 0 for i in range(self.num_players)})
            return self.G.nodes[game_hash]

    def add_edge_between_nodes(self, node1, node2, action=None):
        hash1 = node1["hash"]
        hash2 = node2["hash"]
        self.G.add_edge(hash1, hash2, action=action)

    def get_parent_visits(self, node):
        return sum([self.G.nodes[n]['visits'] for n in list(self.G.predecessors(node['hash']))])
    
    def get_children(self, node):
        return [self.G.nodes[n] for n in list(self.G.successors(node['hash']))]

    def ucb(self, node, player_id):
        if node["visits"] == 0:
            return math.inf
        parent_visits = self.get_parent_visits(node)
        return node["value"][player_id] + 2*math.sqrt(math.log(parent_visits/node['visits']))

    def expand(self, node):
        game = node["game"]
        next_decision = game.get_next_decision()
        for action in next_decision.get_legal_actions(game):
            game_copy = deepcopy(game)
            game_copy.perform_action(action)
            new_node = self.add_game_as_node(game_copy, node)
            self.add_edge_between_nodes(node, new_node, action=action)

    def rollout(self, node):
        game_copy = deepcopy(node['game'])
        while not game_copy.is_done():
            next_decision = game_copy.get_next_decision()
            actions = next_decision.get_all_possible_actions()
            action = random.choice(actions)
            game_copy.perform_action(action)
        
        return game_copy.get_evaluation()

    def find_leaf_state(self, start_node):
        path_to_node = []
        node = start_node
        while len(self.get_children(node)) > 0:
            path_to_node.append(node)
            node = max(self.get_children(node), key=lambda n: self.ucb(n, self.curr_player_id))
        return node, path_to_node

    def get_specific_child(self, node, action):
        for child in self.get_children(node):
            if self.G.get_edge_data(node['hash'], child['hash'])['action'] == action:
                return child
        return None

    @time_func
    def step(self, parent_nodes_path):

        node, path_to_node = self.find_leaf_state(self.curr_start)

        full_path_from_root = parent_nodes_path + path_to_node + [node]

        game = node['game']
        # If the game is done, just get the evaluation
        if game.is_done():
            ts = game.get_evaluation()

        else:
            # Expand node if we've never visited before
            if node['visits'] == 0:
                self.expand(node)
                ts = self.rollout(node)
            else:
                # Get best looking node (based on ucb) and then rollout that node
                node = max(self.get_children(node), key=lambda n: self.ucb(n, self.curr_player_id))
                ts = self.rollout(node)

        full_path_from_root.reverse()

        while len(full_path_from_root) > 0:
            node = full_path_from_root.pop(0)
            node['visits'] += 1
            for player_id, t in ts.items():
                node['t'][player_id] += t
                node['value'][player_id] = node['t'][player_id]/node['visits']
    
    @time_func
    def explore(self, decision, game, steps=10, max_time=5):
        self.curr_start = self.G.nodes[game.hash]
        self.curr_player = decision.player
        self.curr_player_id = self.curr_player.player_id

        parent_nodes = []
        child = self.root

        action_count = 0

        while child != self.curr_start:
            parent_nodes.append(child)
            child = self.get_specific_child(child, game.action_history[action_count])
            action_count += 1

        start_time = time.time()
        # Run for up to <steps> iterations of searches
        for _ in range(steps):
            self.step(parent_nodes)
            # If the max_time for stepping is exceeded, just cease exploring here.
            # This sets a consistent bound on how long an explore can take, which can
            # allow the explore to run in a set time should you not wish to keep a 
            # player waiting for the MCTS to select a move for an unknown period of time.
            if time.time()-max_time > start_time:
                break

    @time_func
    def get_best_action(self):
        children = self.get_children(self.curr_start)
        player_id = self.curr_player_id
        best_value = max([node['value'][player_id] for node in children])
        best_nodes = [node for node in children if node['value'][player_id] == best_value]
        best_node = random.choice(best_nodes)
        best_action = self.G.get_edge_data(self.curr_start['hash'], best_node['hash'])['action']
        return best_action

    def explore_and_get_best_action(self, decision, game):
        self.explore(decision, game)
        return self.get_best_action()

    def visualize_tree(self):
        labels = {node_hash: self.label_from_node(node_hash) for node_hash in self.G.nodes}
        pos = graphviz_layout(self.G, prog="dot")
        nx.draw(self.G, pos, labels=labels)
        plt.show()

    def label_from_node(self, node_name):
        node = self.G.nodes[node_name]
        rounded_values = {k: round(v, 2) for k, v in node['value'].items()}
        # ucb = {player_id: self.ucb(node, player_id) for player_id in node['game'].players.keys()}
        return f"{node['game'].board}\n{rounded_values}\n{node['visits']}"