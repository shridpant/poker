# players/random_agent.py

import random
from players.base import Player

class RandomPlayer(Player):
    """A simple random agent for Kuhn poker."""
    
    def get_action(self, card, available_actions, round_num, chips_remaining, public_state):
        """
        Choose a random action from available actions.
        
        Args:
            card: The player's card
            available_actions: Dictionary of available actions
            round_num: Current betting round (1 or 2)
            chips_remaining: The number of chips the player has
            public_state: The public state of the game
            
        Returns:
            tuple: (Index of chosen action, raise amount)
        """
        action_idx = random.choice(list(available_actions.keys()))
        if action_idx == 4:  # raise
            # Determine minimum raise amount
            min_raise = 1
            if public_state and 'min_raise' in public_state:
                min_raise = public_state['min_raise']
                
            # Pick a random raise between min_raise and whatever the player has
            max_raise = chips_remaining
            if min_raise > max_raise:
                # If we can't meet minimum raise, choose another action
                valid_actions = [a for a in available_actions.keys() if a != 4]
                if valid_actions:
                    return (random.choice(valid_actions), 0)
                # If raise is the only option, use all remaining chips
                return (4, max_raise)
                
            raise_amount = random.randint(min_raise, max_raise)
            return (4, raise_amount)
        return (action_idx, 0)