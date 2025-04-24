class ConservativeAgent:
    """Check if possible; otherwise call; otherwise fold."""
    def get_action(self, card, available_actions, round_num, chips, public_state):
        if 0 in available_actions: return (0, None)
        if 2 in available_actions: return (2, None)
        if 3 in available_actions: return (3, None)
        # fallback
        k = list(available_actions.keys())[0]
        return (k, None)
