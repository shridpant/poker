class AggressiveAgent:
    """Bet or raise whenever you can; else call; else check; else fold."""
    def get_action(self, card, available_actions, round_num, chips, public_state):
        if 1 in available_actions: return (1, None)
        if 4 in available_actions: return (4, public_state.get('min_raise',1))
        if 2 in available_actions: return (2, None)
        if 0 in available_actions: return (0, None)
        if 3 in available_actions: return (3, None)
        # fallback
        k = list(available_actions.keys())[0]
        return (k, None)
