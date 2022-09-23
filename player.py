#!/usr/bin/env python3
import numpy as np
import time
import random

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
        t0 = time.time()
        depth_max = 0
        index = 0

        while time.time() - t0 < 0.055 :
            depth_max += 1
            children = initial_tree_node.compute_and_get_children()
            values = np.array([-np.inf] * len(children))
            for i, child in enumerate(children):
                values[i] = minimax(t0, child, 1, -np.inf, np.inf, depth_max)
            if time.time() - t0 < 0.055:
                best_score = max(values)
                if sum(values == best_score):
                    index = random.choice(np.where(values == best_score)[0])
                else:
                    index = values.index(best_score)
            else :
                depth_max -= 1

            # print(f"Search at depth {depth_max} ended at timestamp : {np.round(time.time() - t0, 6)}s")
        
        # print(f"Search stop at depth {depth_max} in {np.round(time.time() - t0, 6)}s")

        return ACTION_TO_STR[children[index].move]


# Calculus fcts
def minimax(t0, node, player, alpha, beta, max_depth=5):
    curr_state = node.state
    remaining_fishes = len(list(curr_state.fish_positions.keys()))
    # If close to timeout we stop the search :
    if time.time() - t0 > 0.055:
        return -np.inf
    # if all fishes have been caught :
    elif remaining_fishes == 0 or node.depth >= max_depth:
        return heuristic(node)  # end of the game (real utility function) or max_depth reached (approxiamtion through heuristic)

    else: 
        children = node.compute_and_get_children()

        if player == 0:
            v = - np.inf
            for child in children:
                v = max(v, minimax(t0, child, 1, alpha, beta, max_depth))
                alpha = max(alpha, v)
                if beta <= alpha:
                    break
        else:
            v = np.inf
            for child in children :
                v = min(v, minimax(t0, child, 0, alpha, beta, max_depth))
                beta = min(beta, v)
                if beta <= alpha:
                    break
        return v

def heuristic(node):
    """
    Calcultate the heuristic function for a player at a given state
    """
    curr_state = node.state
    curr_score = curr_state.player_scores[0] - curr_state.player_scores[1]
    
    if len(list(curr_state.fish_positions.keys())) == 0:
        return curr_score
    
    fish_caught_by_MAX = curr_state.player_caught[0]
    fish_caught_by_MIN = curr_state.player_caught[1]

    closests_fishes = get_closest_fish_for_loop(curr_state)
    if fish_caught_by_MAX != -1:
        curr_score += curr_state.fish_scores[fish_caught_by_MAX]
    else:
        curr_score += curr_state.fish_scores[closests_fishes[0][0]]/2 * ((19+10) - closests_fishes[0][1]) / (19+10)
    if fish_caught_by_MIN != -1:
        curr_score -= curr_state.fish_scores[fish_caught_by_MIN]
    else: 
        curr_score -= curr_state.fish_scores[closests_fishes[1][0]]/2 * ((19+10) - closests_fishes[1][1]) / (19+10)
    
    return curr_score

def norm_distance_for_loop(position_list, p):
    """
    Calculate the norm between 2 points in 2D
    """
    distance = []
    for position in position_list:
        distance.append(abs(position[0]-p[0]) +  abs(position[1]-p[1]))
    return distance

def get_fish_positions(state):
    fish_scores = state.fish_scores
    fish_positions = state.fish_positions
    positions_list = []
    real_indexes_list = []
    for ind in list(fish_positions.keys()):
        if fish_scores[ind] > 0:
            positions_list.append([fish_positions[ind][0], fish_positions[ind][1]])
            real_indexes_list.append(ind)
    if len(positions_list) > 0:
        return real_indexes_list, positions_list
    else:
        return list(state.fish_positions.keys()), [[p[0], p[1]] for p in list(state.fish_positions.values())]

def norm_distance_for_all_fishes_for_loop(state, player):
    """
    Calculate the distance for all remaining fishes to the position of the player
    """
    fish_real_indexes, fish_positions = get_fish_positions(state)
    player_position = state.hook_positions[player]
    opponent_position = state.hook_positions[abs(player - 1)]
    if player_position[0] < opponent_position[0]:
        for i in range(len(fish_positions)):
            if fish_positions[i][0] >= opponent_position[0]:
                fish_positions[i][0] -= 20
    elif player_position[0] > opponent_position[0]:
        for i in range(len(fish_positions)):
            if fish_positions[i][0] <= opponent_position[0]:
                fish_positions[i][0] += 20
    return fish_real_indexes, norm_distance_for_loop(fish_positions, player_position)

def get_closest_fish_for_loop(state):
    fish_real_indexes_MAX, distance_to_MAX = norm_distance_for_all_fishes_for_loop(state, 0)
    fish_real_indexes_MIN, distance_to_MIN = norm_distance_for_all_fishes_for_loop(state, 1)
    dist_min_to_MAX = min(distance_to_MAX)
    dist_min_to_MIN = min(distance_to_MIN)
    ind_closest_fish_to_MAX = fish_real_indexes_MAX[distance_to_MAX.index(dist_min_to_MAX)]  # we take the first one if several fishes are equidistant
    ind_closest_fish_to_MIN = fish_real_indexes_MIN[distance_to_MIN.index(dist_min_to_MIN)]
    return ((ind_closest_fish_to_MAX, dist_min_to_MAX), (ind_closest_fish_to_MIN, dist_min_to_MIN))
