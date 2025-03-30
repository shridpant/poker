from players.base import Player
from utilities import write_federated_data
import random

class FederatedPlayer(Player):
    def __init__(self, player_id):
        self.player_id = player_id
        self.local_transitions = []

    def get_action(self, card, available_actions, round_num, chips_remaining):
        # Pick a random action from the available options
        action_idx = random.choice(list(available_actions.keys()))
        if action_idx == 4:  # raise
            raise_amount = random.randint(1, chips_remaining)
            return (4, raise_amount)
        return (action_idx, 0)

    def record_local_transition(self, transition):
        """
        Store transition data in a standardized format for federated learning.
        
        Args:
            transition: Dictionary with state, action, reward, done fields
        """
        # Convert state to a serialized representation if it's a dict
        state = transition.get('state', {})
        action = transition.get('action', 0)
        reward = transition.get('reward', 0)
        done = transition.get('done', False)
        
        # Format the transition for CSV storage
        formatted_transition = {
            'local_state': str(state),  # Convert state dict to string
            'local_action': action,
            'local_reward': reward, 
            'local_done': done
        }
        
        # Append to local transitions list
        self.local_transitions.append(formatted_transition)

    def flush_local_transitions(self):
        filename = f"logs/game_data/federated_player_{self.player_id}_data.csv"
        write_federated_data(self.local_transitions, filename)
        self.local_transitions.clear()
