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
    
    def get_action(self, card, available_actions, round_num, chips_remaining):
        """
        Get input from a human user.

        Args:
            card: The player's card
            available_actions: Dictionary of available actions
            round_num: Current betting round (1 or 2)
            chips_remaining: Number of chips the player has remaining

        Returns:
            tuple: (Index of chosen action, raise amount)
        """
        print(f"Your card: {card}")
        moves_str = ", ".join([f"{idx}: {desc.split(' - ')[0]}" for idx, desc in available_actions.items()])
        print(f"Available moves: {moves_str}")
        
        while True:
            try:
                choice = int(input("Enter your move: "))
                if choice in available_actions:
                    if choice == 4:  # raise
                        while True:
                            try:
                                amt = int(input("Enter raise amount: "))
                                if 1 <= amt <= chips_remaining:
                                    return (4, amt)
                                else:
                                    print(f"Invalid amount. You have {chips_remaining} chip(s).")
                            except ValueError:
                                print("Enter an integer amount.")
                    return (choice, 0)
                else:
                    print("Invalid choice. Please select one of the listed moves.")
            except ValueError:
                print("Please enter an integer corresponding to your move.")