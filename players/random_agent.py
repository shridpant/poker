# players/random_agent.py

import random

class RandomPlayer:
    """A simple random agent for Kuhn poker."""
    
    def get_action(self, card, available_actions, round_num, chips_remaining):
        """
        Choose a random action from available actions.
        
        Args:
            card: The player's card
            available_actions: Dictionary of available actions
            round_num: Current betting round (1 or 2)
            chips_remaining: The number of chips the player has
            
        Returns:
            tuple: (Index of chosen action, raise amount)
        """
        action_idx = random.choice(list(available_actions.keys()))
        if action_idx == 4:  # raise
            # Pick a random raise between 1 and whatever the player has
            raise_amount = random.randint(1, chips_remaining)
            return (4, raise_amount)
        return (action_idx, 0)