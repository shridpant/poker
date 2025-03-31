# players/human_agent.py

from players.base import Player

# Mapping from action codes to descriptive strings (for display).
ACTION_DESCRIPTIONS = {
    0: "check",
    1: "bet",
    2: "call",
    3: "fold",
    4: "raise"
}

class HumanPlayer:
    """Represents a human-controlled player who uses console input for actions."""
    
    def get_action(self, card, available_actions, round_num, chips_remaining, public_state):
        """
        Get input from a human user.

        Args:
            card: The player's card
            available_actions: Dictionary of available actions
            round_num: Current betting round (1 or 2)
            chips_remaining: Number of chips the player has remaining
            public_state: The public state of the game

        Returns:
            tuple: (Index of chosen action, raise amount)
        """
        print(f"Your card: {card}")
        print(f"You are Player {public_state['player_id']}")  # Show the player their ID
        
        # Display useful game state information
        print(f"Pot: {public_state['pot_size']}")
        
        # Display each player's chips and bets clearly, highlighting your own status
        print("\nPlayer status:")
        for i in range(public_state['num_players']):
            player_status = f"Player {i}: " 
            if i == public_state['player_id']:
                player_status += "YOU - "  # Highlight the player's own status
            player_status += f"{public_state['chip_counts'][i]} chips"
            player_status += f" (bet: {public_state['current_bets'][i]})"
            if public_state['folded_players'][i]:
                player_status += " [FOLDED]"
            if i == public_state['current_player']:
                player_status += " <- YOUR TURN"
            print(player_status)
        
        # Display betting history in a readable format
        if public_state['betting_history']:
            print(f"Previous actions: {', '.join(public_state['betting_history'])}")
        else:
            print("You are the first to act.")
            
        # Show which players have folded
        folded = [i for i, has_folded in enumerate(public_state['folded_players']) if has_folded]
        if folded:
            print(f"Players who folded: {', '.join(map(str, folded))}")
        
        moves_str = ", ".join([f"{idx}: {desc.split(' - ')[0]}" for idx, desc in available_actions.items()])
        print(f"Available moves: {moves_str}")
        
        while True:
            try:
                action_idx = int(input("Enter your move: "))
                if action_idx in available_actions:
                    if action_idx == 4:  # If raise is selected
                        min_raise = 1  # Default
                        if public_state and 'min_raise' in public_state:
                            min_raise = public_state['min_raise']
                            
                        while True:
                            try:
                                raise_amount = int(input(f"Enter raise amount (minimum {min_raise}): "))
                                if raise_amount < min_raise:
                                    print(f"Error: Minimum raise is {min_raise}.")
                                    continue
                                if raise_amount > chips_remaining:
                                    print(f"Error: You only have {chips_remaining} chips.")
                                    continue
                                break
                            except ValueError:
                                print("Please enter a valid number.")
                        
                        return action_idx, raise_amount
                    
                    return action_idx, 0  # For other actions, no raise amount needed
                else:
                    print("Invalid choice. Please select one of the listed moves.")
            except ValueError:
                print("Please enter an integer corresponding to your move.")