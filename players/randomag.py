import random

class RandomAgent:
    def get_action(self, card, available_actions, round_num, chips, public_state):
        choice = random.choice(list(available_actions.keys()))
        if choice == 4:
            return (4, public_state.get('min_raise', 1))
        return (choice, None)
