import numpy as np
import random

PASS = 0
BET = 1
NUM_ACTIONS = 2

node_map = {}

class Node:
    def __init__(self):
        self.info_set = ""
        self.regret_sum = np.zeros(NUM_ACTIONS)
        self.strategy = np.zeros(NUM_ACTIONS)
        self.strategy_sum = np.zeros(NUM_ACTIONS)

    def __str__(self):
        return f"{self.info_set}: {self.get_average_strategy()}"
    
    # Returns current mixed strategy through regret-matching
    # and adds it to the sum of all strategies so far.
    def get_strategy(self, realization_weight):
        normalizing_sum = 0
        
        for a in range(NUM_ACTIONS):
            self.strategy[a] = self.regret_sum[a] if self.regret_sum[a] > 0 else 0
        
        normalizing_sum = np.sum(self.strategy)

        if normalizing_sum > 0:
            self.strategy /= normalizing_sum
        else:
            self.strategy = np.full(NUM_ACTIONS, 1 / NUM_ACTIONS)

        self.strategy_sum += (realization_weight * self.strategy)

        return self.strategy
    
    # Returns weighted average over all strategies using the sum of strategies:
    # the average strategy is what converges to Nash
    def get_average_strategy(self):
        normalizing_sum = np.sum(self.strategy_sum)

        if normalizing_sum > 0:
            avg_strategy = self.strategy_sum / normalizing_sum
        else:
            avg_strategy = np.full(NUM_ACTIONS, 1 / NUM_ACTIONS)

        return avg_strategy
    
def train(iterations):
    cards = [1, 2, 3]
    util = 0
    for i in range(iterations):
        random.shuffle(cards)
        util += cfr(cards, "", 1, 1)
    
    print(f"Average game value: {util / iterations}")
    
    for key in sorted(node_map): 
        print(key, node_map[key])

def cfr(cards: list[int], history: str, p0: float, p1: float):
    plays = len(history)
    player = plays % 2
    opponent = 1 - player

    if plays > 1:
        terminal_pass = history[plays - 1] == "p"
        double_bet = history[plays - 2:] == "bb"
        is_player_card_higher = cards[player] > cards[opponent]

        if terminal_pass:
            if history == "pp":
                return 1 if is_player_card_higher else -1
            
            else:
                return 1
        
        if double_bet:
            return 2 if is_player_card_higher else -2

    info_set = str(cards[player]) + history

    node = node_map.get(info_set)
    if node is None:
        node = Node()
        node.info_set = info_set
        node_map[info_set] = node
    
    strategy = node.get_strategy(p0 if player == 0 else p1)
    util = np.zeros(NUM_ACTIONS)
    node_util = 0

    for a in range(NUM_ACTIONS):
        next_history = history + ("p" if a == PASS else "b")
        if player == 0:
            util[a] = -cfr(cards, next_history, p0 * strategy[a], p1)
        else:
            util[a] = -cfr(cards, next_history, p0, p1 * strategy[a])
        
        node_util += (strategy[a] * util[a])

    regret = util - node_util
    node.regret_sum = node.regret_sum + (p1 if player == 0 else p0) * regret

    return node_util

train(100000)