"""Rock-Paper-Scissors regret-matching intro for the CFR project."""

import numpy as np
import random

NUM_ACTIONS = 3

class Player:
    """Regret-matching agent for a single-decision game."""

    def __init__(self):
        self.regret_sum = np.zeros(NUM_ACTIONS)
        self.strategy = np.zeros(NUM_ACTIONS)
        self.strategy_sum = np.zeros(NUM_ACTIONS)

    # Returns current mixed strategy through regret-matching and adds
    # it to the sum of all strategies so far.
    def get_strategy(self):
        normalizing_sum = 0
        
        for a in range(NUM_ACTIONS):
            self.strategy[a] = self.regret_sum[a] if self.regret_sum[a] > 0 else 0
        
        normalizing_sum = np.sum(self.strategy)

        if normalizing_sum > 0:
            self.strategy /= normalizing_sum
        else:
            self.strategy = np.full(NUM_ACTIONS, 1 / NUM_ACTIONS)

        self.strategy_sum += self.strategy

        return self.strategy
    
    # Sample an action from the mixed strategy via inverse-CDF:
    # walk the cumulative probability until it exceeds a uniform draw r.
    def get_action(self, strat):
        r = random.random()
        a = 0
        cum_prob = 0

        while a < (NUM_ACTIONS - 1):
            cum_prob += strat[a]

            if r < cum_prob:
                break
            a += 1
        
        return a
    
    # Returns average over all strategies using the sum of strategies:
    # the average strategy is what converges to Nash
    def get_average_strategy(self):
        normalizing_sum = np.sum(self.strategy_sum)

        if normalizing_sum > 0:
            avg_strategy = self.strategy_sum / normalizing_sum
        else:
            avg_strategy = np.full(NUM_ACTIONS, 1 / NUM_ACTIONS)

        return avg_strategy
            
player1 = Player()
player2 = Player()
        
iterations = 100000

for i in range(iterations):
    action_utility1 = np.zeros(NUM_ACTIONS)
    action_utility2 = np.zeros(NUM_ACTIONS)

    player1.get_strategy()
    player2.get_strategy()

    action_p1 = player1.get_action(player1.strategy)
    action_p2 = player2.get_action(player2.strategy)

    # RPS payoff encoding (Rock=0, Paper=1, Scissors=2):
    # tie with same action -> 0; the action (a+1)%3 beats a -> +1; (a-1)%3 loses -> -1.
    action_utility1[action_p2] = 0
    action_utility1[0 if action_p2 == NUM_ACTIONS - 1 else action_p2 + 1] = 1
    action_utility1[NUM_ACTIONS - 1 if action_p2 == 0 else action_p2 - 1] = -1

    action_utility2[action_p1] = 0
    action_utility2[0 if action_p1 == NUM_ACTIONS - 1 else action_p1 + 1] = 1
    action_utility2[NUM_ACTIONS - 1 if action_p1 == 0 else action_p1 - 1] = -1

    # Adds regret of current iteration to sum of regrets:
    # regret defined as the utility an action would have 
    # earned minus the utility from the actual action
    player1.regret_sum += (action_utility1 - action_utility1[action_p1])
    player2.regret_sum += (action_utility2 - action_utility2[action_p2])

print(player1.get_average_strategy())
print(player2.get_average_strategy())

        
