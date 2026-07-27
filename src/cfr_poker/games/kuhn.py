import random
from base import Game
from cfr_poker.cfr.trainer import CFRTrainer

class Kuhn(Game):
    """Kuhn poker implementation for CFR training.

    Uses chance sampling: the deck is shuffled once per iteration and
    private cards are stored internally. Public betting history is encoded
    separately from the sampled cards.
    """

    def __init__(self):
        self.actions = ["p", "b"]   
        self.cards = [1, 2, 3]

    def sample_outcome(self, history: str) -> str:
        """Sample a new private card assignment.

        Kuhn has one chance node: the initial deal. The sampled cards are
        stored in self.cards and the public history only receives the token
        "d" to indicate that the deal occurred.
        """
        random.shuffle(self.cards)
        return "d"

    def is_terminal(self, history: str) -> bool:
        """Check whether Kuhn betting has reached a terminal state.

        A game ends after a fold, a checked-down showdown, or a called bet.
        The leading deal token is excluded when counting actions.
        """
        plays = len(history) - 1   # betting actions only; the leading "d" doesn't count

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
        is_player_card_higher = self.cards[player] > self.cards[opponent]

        if terminal_pass:
            if history[1:] == "pp":     # both checked -> showdown for the antes only
                return 1 if is_player_card_higher else -1

            else:                       # a pass here is a fold; `player` is the non-folder (winner)
                return 1

        if called_bet:                  # showdown after a call
            return 2 if is_player_card_higher else -2

    def get_player(self, history: str) -> int:
        """Return the player acting at this history.

        The deal token is ignored, so the first betting action belongs
        to player 0.
        """
        plays = len(history) - 1
        player = plays % 2
        return player

    def get_info_set(self, history: str) -> str:
        """Return Kuhn's information-set key.

        The key contains the player's private card and the public betting
        history, while excluding the opponent's hidden card.
        """
        player = self.get_player(history)

        # Information sets contain only information available to this player
        return str(self.cards[player]) + history   

    def get_actions(self, history: str) -> list[str]:
        """Return legal actions.

        Action ordering is fixed because CFR stores regrets and strategies
        by action index.
        """
        return self.actions

if __name__ == "__main__":
    trainer = CFRTrainer(Kuhn())
    trainer.train(1000)