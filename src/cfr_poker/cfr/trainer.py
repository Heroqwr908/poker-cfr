"""Counterfactual Regret Minimization (CFR) trainer.

Game-agnostic implementation of vanilla (full-tree, non-sampled) CFR.
The trainer walks the full game tree each iteration, applies
regret-matching at every information set, and accumulates a running
average strategy that converges toward a Nash equilibrium for
two-player zero-sum imperfect-information games.

The trainer knows nothing about any specific game. All rules
(terminal detection, payoffs, legal actions, whose turn it is, chance
outcomes, and information-set keys) are supplied by a `game` object
conforming to the Game interface, so the same engine can solve Kuhn,
Leduc, etc. without modification.

Reference:
    Neller, T. & Lanctot, M. (2013). An Introduction to Counterfactual
    Regret Minimization.
"""
import numpy as np
from cfr_poker.games.base import Game

class Node:
    """Regret and strategy bookkeeping for a single information set.

    Stores cumulative regrets and the running average strategy for a 
    single information set. The node is game-independent; 
    it only knows the number of legal actions.

    Parameters
    ----------
    n_actions : int
        Number of legal actions at this information set. Fixes the
        width of all regret/strategy arrays; must be identical every
        time this info set is reached (array slot i must always mean
        the same action).
    """

    def __init__(self, n_actions: int):
        self.n_actions = n_actions
        self.info_set = ""              # human-readable key, for debugging/printing
        self.regret_sum = np.zeros(self.n_actions)    # cumulative regrets
        self.strategy = np.zeros(self.n_actions)      # current regret-matched strategy
        self.strategy_sum = np.zeros(self.n_actions)  # cumulative average-strategy weights

    def __str__(self):
        return f"{self.info_set}: {self.get_average_strategy()}"

    def compute_strategy(self, realization_weight: float) -> None:
        """Compute the current regret-matched strategy and accumulate it.

        Action probabilities are set proportional to positive regret;
        if no action has positive regret, falls back to uniform. The
        resulting strategy is added to the running sum, weighted by
        this player's reach probability, so the average reflects how
        often each info set is actually played.

        Parameters
        ----------
        realization_weight : float
            The acting player's own reach probability for this info
            set on this iteration.
        """
        self.strategy = np.maximum(self.regret_sum, 0)
        normalizing_sum = np.sum(self.strategy)

        if normalizing_sum > 0:
            self.strategy /= normalizing_sum          
        else:
            self.strategy = np.full(self.n_actions, 1 / self.n_actions)  # no positive regret -> play uniformly

        # Weight by own reach so rarely-reached info sets contribute less to the average.
        self.strategy_sum += (realization_weight * self.strategy)
    
    def get_strategy(self) -> np.ndarray:
        """Return the most recently computed strategy, without updating.

        Returns
        -------
        numpy.ndarray
            The current mixed strategy. This is the node's live array,
            not a copy -- callers should treat it as read-only and not
            mutate it in place.
        """
        return self.strategy

    def get_average_strategy(self) -> np.ndarray:
        """Return the reach-weighted average strategy over all iterations.

        This averaged strategy is what converges to the equilibrium.

        Returns
        -------
        numpy.ndarray
            The normalized average strategy, or uniform if this info
            set was never reached.
        """
        normalizing_sum = np.sum(self.strategy_sum)

        if normalizing_sum > 0:
            avg_strategy = self.strategy_sum / normalizing_sum
        else:
            avg_strategy = np.full(self.n_actions, 1 / self.n_actions)

        return avg_strategy


class CFRTrainer:
    """Drives CFR self-play over a game tree to learn an equilibrium.

    Owns the table of information-set Nodes and the recursive tree
    traversal. Delegates all game rules to the injected `game` object.

    Parameters
    ----------
    game : object
        A game implementing the Game interface
    """

    def __init__(self, game: Game):
        self.node_map = {}    # maps info-set key -> Node; each trainer owns its own table
        self.game = game

    def train(self, iterations: int) -> None:
        """Run CFR self-play for a fixed number of iterations.

        Prints the average game value and the learned average strategy
        at every information set.

        Parameters
        ----------
        iterations : int
            Number of full-tree CFR passes to run.

        Notes
        -----
        The printed "average game value" is a running mean of the
        per-iteration root utility under each iteration's *current*
        strategy -- not the value of the final averaged strategy -- so
        it only trends toward the true value and
        will not match it exactly.
        """
        util = 0
        for i in range(iterations):
            util += self.cfr("", 1, 1)  # start from the empty history

        print(f"Average game value: {util / iterations}")

        for key in sorted(self.node_map):
            print(self.node_map[key])

    def cfr(self, history: str, p0: float, p1: float) -> float:
        """Recursively compute counterfactual utility for one history.

        Walks the subtree rooted at `history`, updating regrets and the
        strategy sum at each decision node reached. Returns the
        expected utility of this history from the perspective of the
        player about to act.

        Parameters
        ----------
        history : str
            Encoded sequence of actions (and chance outcomes) so far.
        p0, p1 : float
            Reach probabilities of players 0 and 1 respectively -- the
            product of their own action probabilities along `history`.

        Returns
        -------
        float
            Expected utility of `history` from the perspective of the
            player to act at `history`. The caller negates this to
            convert it to its own perspective (two-player zero-sum).
        """
        # Base case: game is over, return the terminal payoff.
        if self.game.is_terminal(history):
            return self.game.get_utility(history)

        # Chance node (e.g. the deal): sample an outcome and recurse without touching regrets.
        elif self.game.is_chance_node(history):
            a = self.game.sample_outcome(history)
            next_history = history + a
            return self.cfr(next_history, p0, p1)

        player = self.game.get_player(history)
        info_set = self.game.get_info_set(history)
        actions = self.game.get_actions(history)

        # Fetch this info set's Node, creating it on first visit.
        node = self.node_map.get(info_set)
        if node is None:
            node = Node(len(actions))
            node.info_set = info_set
            self.node_map[info_set] = node

        # Current strategy, weighted into the average by the acting player's own reach.
        node.compute_strategy(p0 if player == 0 else p1)
        strategy = node.get_strategy()
        util = np.zeros(len(actions))   # utility of each action, from the acting player's view
        node_util = 0                   # expected utility of this node under `strategy`

        for i, a in enumerate(actions):
            next_history = history + a
            # Negate the child's value: zero-sum, and the child's acting player is the opponent.
            # Advance only the acting player's reach by the probability of the action taken.
            if player == 0:
                util[i] = -self.cfr(next_history, p0 * strategy[i], p1)
            else:
                util[i] = -self.cfr(next_history, p0, p1 * strategy[i])

            node_util += (strategy[i] * util[i])    

        # Counterfactual regret: how much better each action was than the node's expected value,
        # weighted by the *opponent's* reach (the counterfactual "how likely we got here").
        regret = util - node_util
        node.regret_sum = node.regret_sum + (p1 if player == 0 else p0) * regret

        return node_util