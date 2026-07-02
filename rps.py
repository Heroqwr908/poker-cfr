import numpy as np
import random as rd

NUM_ACTIONS = 3

class Player:
    def __init__(self):
        self.regretSum = np.zeros(3)
        self.strategy = np.zeros(3)
        self.strategySum = np.zeros(3)

    # Returns current mixed strategy through regret-matching
    def getStrategy(self):
        normalizingSum = 0
        
        for a in range(NUM_ACTIONS):
            self.strategy[a] = self.regretSum[a] if self.regretSum[a] > 0 else 0
        
        normalizingSum = np.sum(self.strategy)

        if normalizingSum > 0:
            self.strategy /= normalizingSum
        else:
            self.strategy = np.full(NUM_ACTIONS, 1 / NUM_ACTIONS)

        self.strategySum += self.strategy

        return self.strategy
    
    # Picks action randomly proportional to current strategy
    def getAction(self, strat):
        r = rd.random()
        a = 0
        cum_prob = 0

        while a < (NUM_ACTIONS - 1):
            cum_prob += strat[a]

            if r < cum_prob:
                break
            a += 1
        
        return a
    
    def getAverageStrategy(self):
        avgStrategy = np.zeros(3)
        normalizingSum = np.sum(self.strategySum)

        if normalizingSum > 0:
            avgStrategy = self.strategySum / normalizingSum
        else:
            avgStrategy = np.full(NUM_ACTIONS, 1 / NUM_ACTIONS)

        return avgStrategy
            
player1 = Player()
player2 = Player()
        
iterations = 100000

for i in range(iterations):
    actionUtility1 = np.zeros(3)
    actionUtility2 = np.zeros(3)

    player1.strategy = player1.getStrategy()
    player2.strategy = player2.getStrategy()

    action_p1 = player1.getAction(player1.strategy)
    action_p2 = player2.getAction(player2.strategy)

    actionUtility1[action_p2] = 0
    actionUtility1[0 if action_p2 == NUM_ACTIONS - 1 else action_p2 + 1] = 1
    actionUtility1[NUM_ACTIONS - 1 if action_p2 == 0 else action_p2 - 1] = -1

    actionUtility2[action_p1] = 0
    actionUtility2[0 if action_p1 == NUM_ACTIONS - 1 else action_p1 + 1] = 1
    actionUtility2[NUM_ACTIONS - 1 if action_p1 == 0 else action_p1 - 1] = -1

    player1.regretSum += (actionUtility1 - actionUtility1[action_p1])
    player2.regretSum += (actionUtility2 - actionUtility2[action_p2])

print(player1.getAverageStrategy())
print(player2.getAverageStrategy())

        
