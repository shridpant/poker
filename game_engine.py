# game_engine.py

import time
import random
import csv
import sys
from datetime import datetime

# import pyspiel

from utilities import (
    log_to_console_and_store, 
    append_to_log_file, 
    write_transitions_to_csv
)
from players.human_agent import HumanPlayer
from players.federated_agent import FederatedPlayer

"""
Mappings for card names & actions
"""
# This dictionary maps indexes to card names in Kuhn Poker.
CARD_NAMES = {0: "J", 1: "Q", 2: "K"}

# Possible betting actions
#   0 -> check
#   1 -> bet
#   2 -> call
#   3 -> fold
#   4 -> raise
ACTION_CODES = {0: "check", 1: "bet", 2: "call", 3: "fold", 4: "raise"}

class KuhnPokerEngine:
    """Manages the flow of a Kuhn Poker game, from dealing to resolution."""
    def __init__(self, player0, player1, player2=None, delay=0.5, num_players=2, auto_rounds=None):
        self.players = [player0, player1]
        if player2 is not None:
            self.players.append(player2)
        self.delay = delay
        self.num_players = num_players
        self.auto_rounds = auto_rounds
        self.chips = [10] * num_players
        self.current_hand = 0
        self.transitions = []  # For RL data collection
        self.log_file = None
        
    def log(self, message):
        """Log a message with timestamp."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {message}")
        
    def run_game(self):
        """Run the entire game session."""
        self.log("\n🎲 POKER GAME SESSION STARTED 🎲")
        print("=" * 60)
        self.log(f"Current chip counts before session: "
                 f"{', '.join([f'Player {i}: {self.chips[i]}' for i in range(self.num_players)])}")
        sys.stdout.flush()  # Ensure logs are shown
        try:
            while True:
                self.current_hand += 1
                self.run_round()
                
                # Check if we should continue automatically or ask
                if self.auto_rounds is not None and self.current_hand >= self.auto_rounds:
                    break
                elif self.auto_rounds is None:
                    print("\n" + "-" * 40)
                    response = input("Play another round? (y/n): ")
                    if response.lower() != 'y':
                        break
        finally:
            print("\n" + "=" * 60)
            self.log("🏁 GAME SESSION ENDED 🏁")
            self.log(f"Final chip counts: {', '.join([f'Player {i}: {self.chips[i]}' for i in range(self.num_players)])}")
            self.write_transitions()
            
    def run_round(self):
        """
        Run a single round of Kuhn poker.
        """
        print("\n" + "=" * 60)
        self.log(f"=== Starting Hand {self.current_hand} ===")
        print("=" * 60)
        
        self.log(f"Starting Hand {self.current_hand} with chip counts: "
                 f"{', '.join([f'Player {i}: {self.chips[i]}' for i in range(self.num_players)])}")
        sys.stdout.flush()
        
        # Initialize round variables
        self.pot = 0
        self.current_bets = [0] * self.num_players
        self.folded = [False] * self.num_players
        self.cards = []
        chips_before_round = self.chips[:]  # Track starting chips for diff
        
        # Ante
        ante = 1
        for i in range(self.num_players):
            self.chips[i] -= ante
            self.pot += ante
        self.log(f"All players ante {ante} unit. Pot is now {self.pot}.")

        # Deal cards
        if self.num_players == 3:
            # For 3 players, use four cards (J, Q, K, A)
            deck = ["J", "Q", "K", "A"]
            random.shuffle(deck)
            # Deal 3 cards, keep one face-down (unused)
            for i in range(self.num_players):
                self.cards.append(deck[i])
                self.log(f"Chance node: Dealt card {self.cards[i]} to Player {i}.")
            self.hidden_card = deck[3]
            # Log the unused card (optional: if you want players to see it right away)
            self.log(f"One card is face-down (hidden). For debugging: {self.hidden_card}")
        else:
            # Default 2-player or fallback
            deck = ["J", "Q", "K"]
            random.shuffle(deck)
            for i in range(self.num_players):
                self.cards.append(deck[i])
                self.log(f"Chance node: Dealt card {self.cards[i]} to Player {i}.")

        time.sleep(self.delay)
        
        # Conduct betting rounds
        print("\n" + "-" * 40 + " BETTING ROUND 1 " + "-" * 40)
        first_round_actions, last_acted_player = self.betting_round(1)
        
        if all(a == "check" for a in first_round_actions):
            self.log("All players checked. Going to showdown.")
            self.showdown(chips_before_round)
            return

        if self.num_players == 2 and any(a in ["bet", "call", "raise"] for a in first_round_actions):
            self.showdown(chips_before_round)
            return

        # If not everyone folded, do the second betting round
        active_players = [i for i in range(self.num_players) if not self.folded[i]]
        if len(active_players) > 1:
            print("\n" + "-" * 40 + " BETTING ROUND 2 " + "-" * 40)
            # Start with the next player after the last one who acted in round 1
            next_player = (last_acted_player + 1) % self.num_players
            # Skip any folded players for the starting position
            while self.folded[next_player]:
                next_player = (next_player + 1) % self.num_players
            self.log(f"Second round starts with Player {next_player} (continuing clockwise)")
            second_round_actions, _ = self.betting_round(2, starting_player=next_player)

        # Now do final showdown
        self.showdown(chips_before_round)
        
    def betting_round(self, round_num, starting_player=0):
        """Conduct a betting round."""
        self.log(f"Starting {'First' if round_num == 1 else 'Second'} Betting Round.")
        
        actions = []
        current_player = starting_player
        highest_bet = 0
        players_acted = 0
        all_players_acted = False
        last_bettor = None  # Keep track of the last player who bet or raised
        last_acted_player = starting_player  # Track who acted last for next round
        active_players_count = len([i for i in range(self.num_players) if not self.folded[i]])
        starting_active_players = active_players_count  # Remember how many players we started with

        while not all_players_acted:
            # Skip folded players
            if self.folded[current_player]:
                current_player = (current_player + 1) % self.num_players
                continue
                
            # Determine available actions
            available_actions = {}
            bet_diff = max(0, highest_bet - self.current_bets[current_player])

            # Build list of possible actions, filtering out those not feasible
            if bet_diff == 0:
                # Can check if you have >=0 chips
                if self.chips[current_player] > 0:
                    available_actions[0] = "check - No bet on the table. You may check or bet."
                    # Bet is 1 if they have at least 1 chip
                    if self.chips[current_player] >= 1:
                        available_actions[1] = "bet - You may bet 1 unit."
                else:
                    # No chips => must effectively skip (check if no bet or fold if needed)
                    available_actions[0] = "check - No chips left."
            else:
                # There's a call to pay
                if self.chips[current_player] >= bet_diff:
                    available_actions[2] = f"call - Call the bet of {bet_diff} unit(s)."
                # Fold is always possible
                available_actions[3] = "fold - Fold your hand."
                # Raise check
                if highest_bet > 0 and self.chips[current_player] > bet_diff:
                    available_actions[4] = "raise - Raise by betting an additional amount."

            # Display player turn indicator before determining actions
            print(f"\n[PLAYER {current_player}'S TURN]")
            
            # Display available actions for the current player
            self.log(f"Available actions for Player {current_player} :")
            for idx, desc in available_actions.items():
                self.log(f"{idx}: {desc}")
            
            # Get player action; now returns (action_idx, raise_amount)
            action_idx, raise_amount = self.players[current_player].get_action(
                self.cards[current_player],
                available_actions,
                round_num,
                self.chips[current_player]
            ) if hasattr(self.players[current_player], 'get_action') else (0, None)
            
            # Process the chosen action
            if bet_diff == 0:  # No bet difference
                if action_idx == 0:  # Check
                    self.log(f"Player {current_player} checks.")
                    actions.append("check")
                else:  # Bet
                    bet_amount = min(1, self.chips[current_player])
                    if bet_amount <= 0:
                        self.log(f"Player {current_player} has no chips left - forced to fold.")
                        self.folded[current_player] = True
                        actions.append("fold")
                    else:
                        self.chips[current_player] -= bet_amount
                        self.current_bets[current_player] += bet_amount
                        self.pot += bet_amount
                        highest_bet = self.current_bets[current_player]
                        self.log(f"Player {current_player} bets. Pot is now {self.pot}.")
                        actions.append("bet")
                        last_bettor = current_player
                        players_acted = 1  # Reset action count when there's a new bet
            else:  # There is a bet to call/fold/raise
                if action_idx == 3:  # Fold
                    self.log(f"Player {current_player} folds.")
                    self.folded[current_player] = True
                    actions.append("fold")
                    active_players_count -= 1
                    if active_players_count <= 1:
                        all_players_acted = True
                elif action_idx == 2:  # Call
                    call_amount = min(bet_diff, self.chips[current_player])
                    if call_amount <= 0:
                        self.log(f"Player {current_player} doesn't have enough chips to call - forced to fold.")
                        self.folded[current_player] = True
                        actions.append("fold")
                        active_players_count -= 1
                    else:
                        self.chips[current_player] -= call_amount
                        self.current_bets[current_player] += call_amount
                        self.pot += call_amount
                        self.log(f"Player {current_player} calls. Pot is now {self.pot}.")
                        actions.append("call")
                        
                        # Check if all active players have had a chance to act after the last bet/raise
                        # and have equal bets
                        active_players = [i for i in range(self.num_players) if not self.folded[i]]
                        all_bets_equal = all(self.current_bets[p] == highest_bet for p in active_players)
                        
                        # If all players have acted since the last bet and bets are equal, round is complete
                        if all_bets_equal and (players_acted >= starting_active_players):
                            all_players_acted = True
                            
                elif action_idx == 4:  # Raise
                    # First pay call
                    call_needed = bet_diff
                    call_used = min(call_needed, self.chips[current_player])
                    self.chips[current_player] -= call_used
                    self.current_bets[current_player] += call_used
                    self.pot += call_used

                    # Now add the raise
                    min_extra = 1
                    max_extra = self.chips[current_player]
                    raise_used = min(max_extra, max(min_extra, raise_amount or 1))
                    if raise_used <= 0:
                        self.log(f"Player {current_player} can't raise further - forced to call or fold.")
                        if self.chips[current_player] > 0:
                            self.chips[current_player] -= self.chips[current_player]
                            self.current_bets[current_player] += self.chips[current_player]
                            self.pot += self.chips[current_player]
                            self.log(f"Player {current_player} calls with remaining chips. Pot is now {self.pot}.")
                            actions.append("call")
                        else:
                            self.folded[current_player] = True
                            actions.append("fold")
                            active_players_count -= 1
                    else:
                        self.chips[current_player] -= raise_used
                        self.current_bets[current_player] += raise_used
                        self.pot += raise_used
                        highest_bet = self.current_bets[current_player]
                        self.log(f"Player {current_player} raises by {raise_used} unit(s). Pot is now {self.pot}.")
                        actions.append("raise")
                        last_bettor = current_player
                        # Reset players_acted counter when there's a new bet
                        players_acted = 1
            
            # After processing the chosen action, only record local transitions if agent is FRL
            if isinstance(self.players[current_player], FederatedPlayer):
                transition_data = {
                    "local_state": "example_state",
                    "local_action": action_idx,
                    "local_reward": 0,
                    "local_done": False
                }
                self.players[current_player].record_local_transition(transition_data)
            
            # Move to next player and track if everyone has acted
            players_acted += 1
            current_player = (current_player + 1) % self.num_players
            
            # Check if everyone has folded 
            active_players = [i for i in range(self.num_players) if not self.folded[i]]
            if len(active_players) <= 1:
                all_players_acted = True
                continue
            
            # Check for round completion under different conditions:
            
            # 1. When everyone has checked (no bets made)
            if highest_bet == 0 and players_acted >= starting_active_players:
                all_players_acted = True
                continue
                
            # 2. After a bet/raise: everyone after the bettor needs to act and bets are equal
            equal_bets = all(self.current_bets[player] == highest_bet for player in active_players)
            if last_bettor is not None:
                # We made at least one complete rotation after the last bet
                if players_acted > active_players_count and equal_bets:
                    all_players_acted = True
            
            time.sleep(self.delay)
            sys.stdout.flush()
        
        # At the end, after processing all actions, update last_acted_player
        last_acted_player = (current_player - 1) % self.num_players
        
        # Add last_acted_player to the return value
        return actions, last_acted_player
        
    def showdown(self, chips_before_round):
        """
        Determine the winner and distribute the pot.
        """
        print("\n" + "-" * 40 + " SHOWDOWN " + "-" * 40)
        self.log("=== Showdown ===")
        
        # Show cards of non-folded players
        reveal_message = "Cards revealed: "
        active_players = []
        for i in range(self.num_players):
            if not self.folded[i]:
                reveal_message += f"Player {i}: {self.cards[i]}, "
                active_players.append(i)
        self.log(reveal_message.rstrip(", "))
        
        # If there's a hidden card in 3-player mode, log it at the showdown
        if self.num_players == 3 and hasattr(self, 'hidden_card'):
            self.log(f"(** Debug/Reveal **) The unused hidden card was: {self.hidden_card}")
        
        # Determine winner
        if len(active_players) == 1:
            winner = active_players[0]
            self.log(f"Player {winner} wins the pot of {self.pot} by default (others folded).")
        else:
            # Compare cards (in Kuhn poker, higher card wins)
            card_ranks = {"J": 1, "Q": 2, "K": 3, "A": 4}
            best_card = -1
            winner = -1
            
            for player in active_players:
                card_rank = card_ranks.get(self.cards[player], 0)
                if card_rank > best_card:
                    best_card = card_rank
                    winner = player
                    
            self.log(f"Player {winner} wins the pot of {self.pot} with {self.cards[winner]}.")
        
        # Award pot to winner
        self.chips[winner] += self.pot
        
        # Show per-round diff
        for i in range(self.num_players):
            diff = self.chips[i] - chips_before_round[i]
            if diff > 0:
                self.log(f"Player {i} gained {diff} chip(s) this round.")
            elif diff < 0:
                self.log(f"Player {i} lost {-diff} chip(s) this round.")
            else:
                self.log(f"Player {i}'s chips did not change this round.")
        
        # Display chip counts
        chip_status = ", ".join([f"Player {i}: {self.chips[i]}" for i in range(self.num_players)])
        self.log(f"Chip counts after Hand {self.current_hand}: {chip_status}")
        
    def write_transitions(self):
        """Write collected RL data to a CSV file."""
        if self.transitions:
            with open('logs/game_data/rl_data.csv', 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['state', 'action', 'next_state', 'reward'])
                writer.writerows(self.transitions)
            self.log("Transitions written to logs/game_data/rl_data.csv.")
        
        # Flush local transitions for any FederatedPlayer
        for i, p in enumerate(self.players):
            if isinstance(p, FederatedPlayer):
                p.flush_local_transitions()