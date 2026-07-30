import random
from cfr_poker.games.base import Game

class Kuhn(Game):
    """Kuhn poker implementation for CFR training.

    Kuhn poker is a minimal two-player, zero-sum poker variant,
    A three-card deck (ranks 1-3) is shuffled and one private card is dealt
    to each player. After a single betting round -- each player may check or
    bet, and a bet may be called or folded to -- the higher card wins at
    showdown. Antes are implicit in the +/-1 and +/-2 payoffs.

    State representation
    --------------------
    The full game state is encoded in the `history` string:

    - The deal is the first two characters, positionally: `history[0]` is
      player 0's card and `history[1]` is player 1's. This is lossless --
      the exact deal is always recoverable, which terminal payoffs require.
    - Betting actions ("p" for pass/check/fold, "b" for bet/call) follow.
    - An information-set key (see `get_info_set`) projects this down to one
      player's view by keeping their own card and the public betting while
      dropping the opponent's hidden card.
    """

    def __init__(self):
        self.actions = ["p", "b"]  

    def enumerate_chance(self, history: str) -> dict[str, float]:
        """Returns the dictionary with all possible deal outcomes.

        Kuhn has 6 possible deals each with equal chance of happening,
        hence each has chance 1/6.
        """
        return {"12": 1/6, "13": 1/6, "21": 1/6, "23": 1/6, "31": 1/6, "32": 1/6}

    def sample_outcome(self, history: str) -> str:
        """Sample a new private card assignment.

        Kuhn has one chance node: the initial deal. The sampled cards are
        stored in the first 2 characters of the history.
        """
        cards = [1, 2, 3]
        random.shuffle(cards)
        return f"{cards[0]}{cards[1]}"

    def is_terminal(self, history: str) -> bool:
        """Check whether Kuhn betting has reached a terminal state.

        A game ends after a fold, a checked-down showdown, or a called bet.
        The leading card tokens are excluded when counting actions.
        """
        plays = len(history) - 2   # betting actions only; the leading "d" doesn't count

        if plays > 1:
            return history[-1] == "p" or history[-2:] == "bb"
        else:
            return False

    def is_chance_node(self, history: str) -> bool:
        """Return whether history is at the initial deal node.

        The empty history represents Kuhn's only chance event: dealing cards.
        """
        return history == ""

    def get_utility(self, history: str) -> float:
        """Return terminal payoff from the CFR reference player's view.

        The reference player is the player returned by get_player(history).
        This matches the trainer's value propagation convention.
        Assumes history is terminal.

        Kuhn payoffs are:
        - +/-1 for folds and checked-down showdowns
        - +/-2 for called bets
        """
        player = self.get_player(history)   # reference player = who would act next
        opponent = 1 - player

        terminal_pass = history[-1] == "p"    # ends in =check or fold
        called_bet = history[-2:] == "bb"     # bet followed by call
        is_player_card_higher = int(history[player]) > int(history[opponent])

        if terminal_pass:
            if history[2:] == "pp":     # both checked -> showdown for the antes only
                return 1 if is_player_card_higher else -1

            else:                       # a pass here is a fold; `player` is the non-folder (winner)
                return 1

        if called_bet:                  # showdown after a call
            return 2 if is_player_card_higher else -2

    def get_player(self, history: str) -> int:
        """Return the player acting at this history.

        The 2 character deal prefix is skipped, so the first betting 
        action belongs to player 0.
        """
        plays = len(history) - 2
        player = plays % 2
        return player

    def get_info_set(self, history: str) -> str:
        """Return Kuhn's information-set key.

        The key contains the player's private card and the public betting
        history, while excluding the opponent's hidden card.
        """
        player = self.get_player(history)

        # Information sets contain only information available to this player
        return history[player] + history[2:]   

    def get_actions(self, history: str) -> list[str]:
        """Return legal actions.

        Action ordering is fixed because CFR stores regrets and strategies
        by action index.
        """
        return self.actions
