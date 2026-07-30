"""Exact best-response and exploitability computation.

Computes the exact value of a player's best response against a fixed
strategy profile in a two-player zero-sum extensive-form game. Combining the best
response for both players yields NashConv and exploitability, the standard
distance-from-equilibrium metric used to verify that a solver is
converging.

The implementation:
- gathers information-set histories with reach probability
- evaluates the tree with memoized recursion
- computes NashConv and exploitability

"""
import numpy as np
from cfr_poker.games.base import Game


class BestResponse:
    """Computes exact best responses and exploitability against a fixed profile.

    A single instance can be reused across many calls; per-computation state
    (gathered histories and caches) is reset at the start of each
    ``best_response`` call, so results from different players or profiles never
    contaminate each other.

    Parameters
    ----------
    game : Game
        A game implementing the Game interface. Supplies the tree structure,
        chance outcomes, terminal payoffs, information-set keys, and legal
        actions.
    """

    def __init__(self, game: Game):
        self.game = game
        self.br_player = 0          # player we are computing the best response for
        self.strategies = {}        # fixed profile the best response plays against
        self.value_cache = {}       # history -> best-response value (memoized)
        self.action_cache = {}      # info set -> chosen best-response action (memoized)
        self.infoset_histories = {} # info set -> list of (history, reach probability)

    def best_response(self, strategies: dict, br_player: int) -> float:
        """Compute the best-response value for one player against a fixed profile.

        Collects information-set histories, evaluates the best-response value, and 
        finally forces a best-response action to be cached for *every* information 
        set -- including those the value recursion never reached because they lie 
        below an action the best responder does not take (see Notes).

        Parameters
        ----------
        strategies : dict
            Maps every information-set key to a strategy: an array of action
            probabilities aligned with ``game.get_actions``. Must cover every
            information set reachable in the tree by the opponent.
        br_player : int
            The player whose best response is being computed.

        Returns
        -------
        float
            The value of the game to ``br_player`` when they best-respond and the
            other player follows ``strategies``, from ``br_player``'s perspective.

        Notes
        -----
        The value recursion only descends through the best responder's *chosen*
        action at each information set, so it never visits information sets that
        sit below a non-chosen action. The trailing loop fills those in so that
        ``get_best_response_strategy`` returns a complete pure strategy. It is
        unnecessary if only the returned value is needed.
        """
        self.value_cache = {}
        self.action_cache = {}
        self.infoset_histories = {}
        self.strategies = strategies
        self.br_player = br_player

        self._collect_infoset_histories("", 1)
        best_response_value = self._best_response_value("")
        # Force a decision for every info set, including off the best-response path.
        for info_set in self.infoset_histories:
            if info_set not in self.action_cache:
                self._best_action(info_set)

        return best_response_value

    def compute_exploitability(self, strategies: dict) -> float:
        """Return the exploitability of a two-player profile.

        Parameters
        ----------
        strategies : dict
            The profile to evaluate, in the same format as ``best_response``,
            but must contain strategies for both players.

        Returns
        -------
        float
            NashConv / 2, where NashConv is the sum of both players'
            best-response values.
        """
        nash_conv = self.best_response(strategies, 0) + self.best_response(strategies, 1)
        exploitability = nash_conv / 2
        return exploitability

    def get_best_response_strategy(self) -> dict:
        """Return the pure best-response strategy from the most recent call.

        Returns
        -------
        dict
            Maps information-set key to the chosen best-response action, as
            populated by the last ``best_response`` call.

        Notes
        -----
        Reflects whichever player was passed to the most recent ``best_response``
        call. After ``compute_exploitability`` this is player 1's best response,
        since that was the last computed. Call ``best_response`` for the player
        you want immediately before reading this.
        """
        return self.action_cache

    def _collect_infoset_histories(self, history: str, reach: float) -> None:
        """Collect histories belonging to each information set.

        Recurses into *every* action at every node (no decisions are made here),
        threading the reach probability down the tree. These histories are later 
        used to evaluate each candidate action across the entire information set, 
        ensuring a single action is chosen consistently for all indistinguishable histories.

        Parameters
        ----------
        history : str
            Encoded sequence of actions and chance outcomes so far.
        reach : float
            Reach probability of ``history``: the product of the opponent's
            action probabilities and chance probabilities along the path. The
            best responder's own probabilities are never multiplied in.
        """
        if self.game.is_terminal(history):
            return

        # Chance: branch into every outcome, scaling reach by its probability.
        elif self.game.is_chance_node(history):
            for a, probability in self.game.enumerate_chance(history).items():
                next_history = history + a
                next_reach = probability * reach
                self._collect_infoset_histories(next_history, next_reach)
            return

        is_br_player_turn = (self.br_player == self.game.get_player(history))
        actions = self.game.get_actions(history)
        info_set = self.game.get_info_set(history)

        if is_br_player_turn:
            self.infoset_histories.setdefault(info_set, []).append((history, reach))
            # The best responder's own action probabilities are omitted from
            # reach probability, so recurse without changing `reach`.
            for a in actions:
                next_history = history + a
                self._collect_infoset_histories(next_history, reach)
            return

        else:
            # Opponent: scale reach by the opponent's probability of each action.
            for i, a in enumerate(actions):
                next_history = history + a
                next_reach = reach * self.strategies[info_set][i]
                self._collect_infoset_histories(next_history, next_reach)
            return

    def _best_response_value(self, history: str) -> float:
        """Evaluate a history from the best responder's perspective.

        Terminal values are converted to the best responder's perspective; all
        internal nodes forward or sum child values with no further negation.
        Best-responder nodes follow the single action chosen by ``_best_action``;
        opponent nodes take the strategy-weighted average over actions.

        Parameters
        ----------
        history : str
            Encoded sequence of actions and chance outcomes so far.

        Returns
        -------
        float
            Expected value from ``history`` onward, assuming the best responder
            plays its best response and the opponent follows ``strategies``.
        """
        if self.game.is_terminal(history):
            # Convert the terminal payoff into the best responder's perspective.
            if self.game.get_player(history) == self.br_player:
                return self.game.get_utility(history)
            else:
                return -self.game.get_utility(history)

        # Expected value over all chance outcomes.
        elif self.game.is_chance_node(history):
            value = 0
            for a, probability in self.game.enumerate_chance(history).items():
                next_history = history + a
                value += probability * self._best_response_value(next_history)
            return value

        is_br_player_turn = (self.br_player == self.game.get_player(history))
        value = 0
        actions = self.game.get_actions(history)
        info_set = self.game.get_info_set(history)

        if is_br_player_turn:
            # Memoized by full history (safe under perfect recall).
            if history in self.value_cache:
                return self.value_cache[history]
            # Follow the single best action for this info set.
            a = self._best_action(info_set)
            next_history = history + a

            value = self._best_response_value(next_history)
            self.value_cache[history] = value
            return value

        else:
            for i, a in enumerate(actions):
                next_history = history + a
                value += self.strategies[info_set][i] * self._best_response_value(next_history)

            return value

    def _best_action(self, info_set: str) -> str:
        """Choose the best-response action for an information set.

        Scores each legal action as the reach-weighted sum of its resulting value
        across every history in the information set, then returns the argmax. The
        sum over histories is what makes the choice a valid best response: the
        best responder cannot distinguish these histories, so a single action
        must be chosen for all of them.

        Parameters
        ----------
        info_set : str
            The information-set key to decide.

        Returns
        -------
        str
            The chosen action token, also cached in ``action_cache``.

        Notes
        -----
        Legal actions are read from a representative history (any member of the
        information set has the same legal actions). Ties are broken by taking the
        first argmax in ``get_actions`` order.
        """
        if info_set in self.action_cache:
            return self.action_cache[info_set]

        # Any member history has the same legal actions as the whole info set.
        actions = self.game.get_actions(self.infoset_histories[info_set][0][0])
        action_values = []

        # Score each action: reach-weighted sum over every history
        # in the information set.
        for a in actions:
            action_value = 0
            for history, reach in self.infoset_histories[info_set]:
                action_value += reach * self._best_response_value(history + a)
            action_values.append(action_value)

        max_action_value = max(action_values)
        max_action_index = action_values.index(max_action_value)
        best_action = actions[max_action_index]
        self.action_cache[info_set] = best_action
        return best_action