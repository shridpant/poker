# players/base.py

class Player:
    """
    Abstract base class for an agent/player.
    Subclasses should implement get_action().
    """

    def get_action(self, state, legal_actions):
        """
        Given the current game state and a list of legal action codes,
        choose and return one action (integer code).
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    def record_local_transition(self, transition):
        """
        Override this to store transitions locally for federated learning.
        By default, do nothing.
        """
        pass