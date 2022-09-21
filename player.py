#!/usr/bin/env python3
import random
import numpy as np

from fishing_game_core.game_tree import Node
from fishing_game_core.player_utils import PlayerController
from fishing_game_core.shared import ACTION_TO_STR


class PlayerControllerHuman(PlayerController):
    def player_loop(self):
        """
        Function that generates the loop of the game. In each iteration
        the human plays through the keyboard and send
        this to the game through the sender. Then it receives an
        update of the game through receiver, with this it computes the
        next movement.
        :return:
        """

        while True:
            # send message to game that you are ready
            msg = self.receiver()
            if msg["game_over"]:
                return


class PlayerControllerMinimax(PlayerController):

    def __init__(self):
        super(PlayerControllerMinimax, self).__init__()

    def player_loop(self):
        """
        Main loop for the minimax next move search.
        :return:
        """

        # Generate first message (Do not remove this line!)
        first_msg = self.receiver()

        while True:
            msg = self.receiver()

            # Create the root node of the game tree
            node = Node(message=msg, player=0)

            # Possible next moves: "stay", "left", "right", "up", "down"
            best_move = self.search_best_next_move(initial_tree_node=node)

            # Execute next action
            self.sender({"action": best_move, "search_time": None})

    def search_best_next_move(self, initial_tree_node):
        """
        Use minimax (and extensions) to find best possible next move for player 0 (green boat)
        :param initial_tree_node: Initial game tree node
        :type initial_tree_node: game_tree.Node
            (see the Node class in game_tree.py for more information!)
        :return: either "stay", "left", "right", "up" or "down"
        :rtype: str
        """

        # EDIT THIS METHOD TO RETURN BEST NEXT POSSIBLE MODE USING MINIMAX ###

        # NOTE: Don't forget to initialize the children of the current node
        #       with its compute_and_get_children() method!
        children = initial_tree_node.compute_and_get_children()
        values = [0] * len(children)
        for i, child in enumerate(children):
            values[i] = minimax(child, 1, max_depth=2)
        index = values.index(max(values))  # Careful might have several move with same value

        return ACTION_TO_STR[children[index].move]


# Calculus fcts

def minimax(node, player, max_depth=5):
    
    curr_state = node.state
    remaining_points = sum(list(curr_state.fish_scores.values()))
    # if all fishes have been caught :
    if remaining_points == 0 or node.depth >= max_depth:
        return heuristic(node)  # end of the game (real utility function) or max_depth reached (approxiamtion through heuristic)

    else:
        if player == 0:
            v = - np.inf
            children = node.compute_and_get_children()
            for child in children :
                v = max(v, minimax(child, 1))
            return v
        else:
            v = np.inf
            children = node.compute_and_get_children()
            for child in children :
                v = min(v, minimax(child, 0))
            return v

def heuristic(node):
    """
    Calcultate the heuristic function for a player at a given state
    """
    curr_state = node.state
    curr_score = curr_state.player_scores[0] - curr_state.player_scores[1]
    total_points_collected = curr_state.player_scores[0] + curr_state.player_scores[1]
    remaining_points = sum(list(curr_state.fish_scores.values()))
    # if all fishes have been caught :
    if remaining_points == 0:  # NOT REALLY THE RIGHT CONDITION : TBM !!!
        return curr_score  # end of the game we have the real utility function
    else:
        remaining_fishes = list(curr_state.fish_positions.keys())
        fish_scores = np.array([curr_state.fish_scores[i] for i in remaining_fishes])
        closer_player = np.array(compute_closer_player_to_fish(curr_state))
        mask0 = closer_player == 0
        return curr_score + sum(fish_scores[mask0]) - sum(fish_scores[~mask0])
        

def norm2_distance(position_array, p):
    """
    Calculate the euclidian norm between 2 points in 2D
    """
    return np.sqrt((position_array[:, 1]-p[1]) ** 2 + (position_array[:, 0]-p[0]) ** 2)

def norm2_distance_for_all_fishes(state, player):
    """
    Calculate the distance for all remaining fishes to the position of the player
    """
    fish_positions = np.array([[p[0], p[1]] for p in list(state.fish_positions.values())])
    player_position = state.hook_positions[player]
    if player_position[0] < 9:
        mask = fish_positions[:, 0] > (10 + player_position[0])
        fish_positions[:, 0][mask] = -fish_positions[:, 0][mask]
    elif player_position[0] > 10:
        mask = fish_positions[:, 0] < (player_position[0] - 10)
        fish_positions[:, 0][mask] = -fish_positions[:, 0][mask]
    return norm2_distance(fish_positions, player_position)

def compute_closer_player_to_fish(state):
    distance_to_player_0 = norm2_distance_for_all_fishes(state, 0)
    distance_to_player_1 = norm2_distance_for_all_fishes(state, 1)
    
    closer_player = [0]*len(distance_to_player_0)
    for i in range(len(distance_to_player_0)):
        if distance_to_player_0[i] <= distance_to_player_1[i]:
            closer_player[i] = 1
    
    return closer_player

