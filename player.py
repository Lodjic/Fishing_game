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
        Function that initiate the minimax algo on the root node to find best possible next move for player 0 (green boat)
        :param initial_tree_node: Initial game tree node
        :type initial_tree_node: game_tree.Node
            (see the Node class in game_tree.py for more information!)
        :return: either "stay", "left", "right", "up" or "down"
        :rtype: str
        """

        t0 = time.time()
        depth_max = 0
        index = 0
        hash_dict = {}

        while time.time() - t0 < 0.055 :
            depth_max += 1
            children = initial_tree_node.compute_and_get_children()
            values = [-np.inf] * len(children)
            for i, child in enumerate(children):
                values[i] = minimax(t0, hash_dict, child, 1, -np.inf, np.inf, depth_max)
            if time.time() - t0 < 0.055:
                best_score = max(values)
                best_score_indexes = where_equal(values, best_score)
                if len(best_score_indexes) > 1:
                    index = random.choice(best_score_indexes)
                else:
                    index = values.index(best_score)
                action = children[index].move
            else :
                depth_max -= 1

            # print(f"Search at depth {depth_max} ended at timestamp : {np.round(time.time() - t0, 6)}s")
        
        # print(f"Search stop at depth {depth_max} in {np.round(time.time() - t0, 6)}s")

        return ACTION_TO_STR[action]



############################################## CALCULUS FUNCTIONS ##############################################

def minimax(t0, hash_table, node, player, alpha, beta, max_depth=5):
    """
    Function that compute the minimax algo on a certain node in order to find best possible next move for player 0 (green boat).
    It is a recursive function that initiate the search at next depth (ie for the children nodes) if there is still enough time.
    In fact we only have 75e-3 seconds to choose an action.
    """
    curr_state = node.state
    remaining_fishes = len(list(curr_state.fish_positions.keys()))
    # If close to timeout we stop the search :
    if time.time() - t0 > 0.055:
        return -np.inf
    # If all fishes have been caught, it is not necesary to go deeper because there is no more action to optimize. So we return the heuristic.
    # Also if we are at the max depth the current node is a terminal leaf so we also return the heuristic
    elif remaining_fishes == 0 or node.depth >= max_depth:
        # To avoid repeated stateswe check if we did not already calculate the value
        hash_code_state = compute_hash_code(node)
        if hash_code_state in hash_table:
            value = hash_table[hash_code_state]
            return value
        else :
            value = heuristic(node)
            hash_table[hash_code_state] = value
            return value  # terminal leaf of the tree because end of the game (real utility function) or max_depth reached (approxiamtion through heuristic)

    # In all other cases we compute the children of the current node and recursively call the minimax fct while using alpha-beta pruning.
    else: 
        children = node.compute_and_get_children()

        if player == 0:
            v = - np.inf
            for child in children:
                v = max(v, minimax(t0, hash_table, child, 1, alpha, beta, max_depth))
                alpha = max(alpha, v)
                if beta <= alpha:
                    break  # alpha-beta pruning
        else:
            v = np.inf
            for child in children :
                v = min(v, minimax(t0, hash_table, child, 0, alpha, beta, max_depth))
                beta = min(beta, v)
                if beta <= alpha:
                    break  # alpha-beta pruning
        return v

def heuristic(node):
    """
    Calcultate the heuristic function for a player at a given state
    """
    # Compute the current score (positive if player 0 leads, negative in the other case)
    curr_state = node.state
    curr_score = curr_state.player_scores[0] - curr_state.player_scores[1]
    
    # If no more fishes to catch, returns the current score
    if len(list(curr_state.fish_positions.keys())) == 0:
        return curr_score
    
    # Gets to know if MAX and MIN have caught any fish
    fish_caught_by_MAX = curr_state.player_caught[0]
    fish_caught_by_MIN = curr_state.player_caught[1]

    # Find the closest fish to MAX and MIN with their respective distance to the player in question
    closests_fishes = get_closest_fish(curr_state)

    # If one of the player gets a fish, we add its score (positively for MAX, negatively for MIN) to the game score. 
    # If not, we add half of the closest fish score (positively for MAX, negatively for MIN) normalized by its distance ot the player
    # (19+10) is the max distance between a hook and a fish in this game if we take into account that the adversary boat can block MAX.
    if fish_caught_by_MAX != -1:
        curr_score += curr_state.fish_scores[fish_caught_by_MAX]
    else:
        curr_score += curr_state.fish_scores[closests_fishes[0][0]]/2 * ((19+10) - closests_fishes[0][1]) / (19+10)
    if fish_caught_by_MIN != -1:
        curr_score -= curr_state.fish_scores[fish_caught_by_MIN]
    else: 
        curr_score -= curr_state.fish_scores[closests_fishes[1][0]]/2 * ((19+10) - closests_fishes[1][1]) / (19+10)
    
    return curr_score

def norm_distance(position_list, p):
    """
    Calculate the norm between 2 points in 2D
    """
    distance = []
    for position in position_list:
        distance.append(abs(position[0]-p[0]) +  abs(position[1]-p[1]))
    return distance

def get_fish_positions(state):
    """
    Return 2 lists : the list of fishes id and the corresponding list of fishes positions
    But it returns those list only for the fishes with positives scores
    If there are no more fishes with positive scores we return those list for all the remaining fishes
    """
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

def norm_distance_for_all_fishes(state, player):
    """
    Calculate the distance between a fish and the player for all the remaining fishes
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
    return fish_real_indexes, norm_distance(fish_positions, player_position)

def get_closest_fish(state):
    """
    Returns the closest fish to MAX and to MIN with their respective corresponding distance to MAX and to MIN
    """
    fish_real_indexes_MAX, distance_to_MAX = norm_distance_for_all_fishes(state, 0)
    fish_real_indexes_MIN, distance_to_MIN = norm_distance_for_all_fishes(state, 1)
    dist_min_to_MAX = min(distance_to_MAX) # we take the first one if several fishes are equidistant
    dist_min_to_MIN = min(distance_to_MIN) # we take the first one if several fishes are equidistant
    score_fishes = [state.fish_scores[i] for i in fish_real_indexes_MAX]
    ind_fish_score_min = fish_real_indexes_MAX[score_fishes.index(min(score_fishes))]
    return ((ind_fish_score_min, dist_min_to_MAX), (ind_fish_score_min, dist_min_to_MIN))

def compute_hash_code(node):
    """
    Computes the hash of a node thanks to the different positions of the players and teh fishes
    """
    hash_code = str(node.state.hook_positions[0][0]) + str(node.state.hook_positions[0][1]) + str(node.state.hook_positions[1][0]) + str(node.state.hook_positions[1][1])
    for i in list(node.state.fish_positions.keys()):
        hash_code += str(node.state.fish_positions[i][0]) + str(node.state.fish_positions[i][1])
    return hash_code

def where_equal(l, condition):
    """
    Finds the indexes at which a list l equals a condition
    """
    indexes = []
    for i in range(len(l)):
        if l[i] == condition:
            indexes.append(i)
    return indexes