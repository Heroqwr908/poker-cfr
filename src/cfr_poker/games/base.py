from abc import ABC, abstractmethod

class Game(ABC):
    """Abstract interface for games used by the CFR trainer.

    A game implementation defines the game tree structure, including
    chance events, terminal states, utilities, information sets, and
    legal actions.
    """

    @abstractmethod
    def sample_outcome(self, history: str) -> str:
        """Sample one chance outcome at a chance node.

        Draws a single outcome (e.g. a card dealt) according to 
        its probability and returns its encoded token.

        Parameters
        ----------
        history : str
            Encoded sequence of actions (and chance outcomes) so far;
            must be a chance node.

        Returns
        -------
        str
            Encoded token for the sampled outcome.
        """

    @abstractmethod
    def is_terminal(self, history: str) -> bool:
        """Return whether the history is a terminal (end-of-game) node.

        Parameters
        ----------
        history : str
            Encoded sequence of actions (and chance outcomes) so far.

        Returns
        -------
        bool
            True if play has ended at this history (payoffs defined),
            False otherwise.
        """

    @abstractmethod
    def is_chance_node(self, history: str) -> bool:
        """Return whether the history is a chance node.

        Parameters
        ----------
        history : str
            Encoded sequence of actions (and chance outcomes) so far.

        Returns
        -------
        bool
            True if this history is chance node,
            False otherwise.
        """

    @abstractmethod
    def get_utility(self, history: str) -> float:
        """Return the utility of the provided terminal history.
        
        Parameters
        ----------
        history : str
            Encoded sequence of actions (and chance outcomes) so far, 
            must be terminal.

        Returns
        -------
        float
            Utility of the provided terminal history.
         
        Notes
        -----
        Payoff returned from the perspective of the player to act 
        at the terminal history.
        """

    @abstractmethod
    def get_player(self, history: str) -> int:
        """Return the current player.
        
        Parameters
        ----------
        history : str
            Encoded sequence of actions (and chance outcomes) so far.

        Returns
        -------
        int
            Player whose turn it is according to the provided history.
        """

    @abstractmethod
    def get_info_set(self, history: str) -> str:
        """Return the info set the history is part of.
        
        Parameters
        ----------
        history : str
            Encoded sequence of actions (and chance outcomes) so far.

        Returns
        -------
        str
            Encoded string of the info set.
        """

    @abstractmethod
    def get_actions(self, history: str) -> list[str]:
        """Return all legal actions at the info set of the provided history.
        
        Parameters
        ----------
        history : str
            Encoded sequence of actions (and chance outcomes) so far.

        Returns
        -------
        list[str]
            List of all legal actions at the info set of the provided history.
        
        Notes
        -----
        Actions must be returned in stable consistent order.
        """