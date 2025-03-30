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

class RLDataLogger:
    """Handles RL transitions and data exporting."""
    def __init__(self):
        # Store transitions here instead of the engine
        self.transitions = []

    def record_transition(self, transition):
        self.transitions.append(transition)

    def write_transitions(self, log_callback):
        if self.transitions:
            from utilities import write_transitions_to_csv
            write_transitions_to_csv(self.transitions, log_callback)
            # Also export training data
            self.export_training_data(log_callback)
        
    def export_training_data(self, log_callback, filename="logs/game_data/training_data.csv"):
        if not self.transitions:
            log_callback("No transitions to export.")
            return
            
        import os
        import ast
        
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # Card mapping to numeric values
        card_map = {"J": 1, "Q": 2, "K": 3, "A": 4}
        
        # Prepare structured data
        training_data = []
        
        for t in self.transitions:
            try:
                # Parse the state string into a dictionary
                if isinstance(t["state"], str):
                    state = ast.literal_eval(t["state"])
                else:
                    state = t["state"]
                    
                # Feature vector
                features = {
                    # Convert cards to numeric values
                    "player_card": card_map.get(state.get(f"player{t['current_player']}_card", ""), 0),
                    "pot_ratio": state.get("pot", 0) / 10.0,  # Normalize pot size
                    "chips_ratio": float(state.get("chips", "0;0").split(";")[min(t["current_player"], len(state.get("chips", "0;0").split(";"))-1)]) / 10.0 if state.get("chips") else 0.0,
                    "round": state.get("round", 1) / 5.0,  # Normalize round number
                    "is_first_round": 1.0 if state.get("stage") == "first" else 0.0,
                    # One-hot encode betting position
                    "position_p0": 1.0 if t["current_player"] == 0 else 0.0,
                    "position_p1": 1.0 if t["current_player"] == 1 else 0.0,
                    "position_p2": 1.0 if t["current_player"] == 2 else 0.0,
                    # Add betting history features
                    "history_bet_count": state.get("betting_history", "").count("bet") / 5.0,
                    "history_raise_count": state.get("betting_history", "").count("raise") / 5.0,
                    "history_fold_count": state.get("betting_history", "").count("fold") / 5.0,
                }
                
                # One-hot encode action
                action_vec = [0] * 5  # 5 possible actions
                action_idx = int(t.get("chosen_action", 0))
                if 0 <= action_idx < 5:
                    action_vec[action_idx] = 1
                    
                # Row with features, action, reward, done
                row = {
                    "player_id": t["current_player"],
                    "episode": f"{t['session_id']}_{t['round']}",
                    "step": t["decision_index"],
                    **features,
                    "action_check": action_vec[0],
                    "action_bet": action_vec[1], 
                    "action_call": action_vec[2],
                    "action_fold": action_vec[3],
                    "action_raise": action_vec[4],
                    "reward": t["reward"],
                    "done": 1 if t["done"] else 0
                }
                
                training_data.append(row)
            except Exception as e:
                log_callback(f"Error processing transition: {e}")
        
        # Write
        if training_data:
            import csv
            with open(filename, 'w', newline='') as csvfile:
                fieldnames = list(training_data[0].keys())
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(training_data)
                log_callback(f"Training data exported to {filename}")
        else:
            log_callback("No valid training data to export.")

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
        self.rlogger = RLDataLogger()
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
            self._deal_cards_3player()
        else:
            self._deal_cards_2player()

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
        
    def _deal_cards_2player(self):
        # Default 2-player or fallback
        deck = ["J", "Q", "K"]
        random.shuffle(deck)
        for i in range(self.num_players):
            self.cards.append(deck[i])
            self.log(f"Chance node: Dealt card {self.cards[i]} to Player {i}.")

    def _deal_cards_3player(self):
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
            
            # Record transition for RL training (for ALL players)
            session_id = datetime.now().strftime("%Y%m%d%H%M%S")
            transition_data = {
                "session_id": session_id,
                "round": self.current_hand,
                "decision_index": players_acted,
                "stage": "first" if round_num == 1 else "second",
                "current_player": current_player,
                "state": {
                    "round": self.current_hand,
                    "stage": "first" if round_num == 1 else "second",
                    "current_player": current_player,
                    "player0_card": self.cards[0] if 0 < len(self.cards) else -1,
                    "player1_card": self.cards[1] if 1 < len(self.cards) else -1,
                    "player2_card": self.cards[2] if 2 < len(self.cards) and self.num_players > 2 else -1,
                    "pot": self.pot,
                    "chips": ";".join([str(c) for c in self.chips[:2]]),  # Only storing first two players' chips for simplicity
                    "betting_history": "".join(str(a) for a in actions),
                },
                "legal_actions": list(available_actions.keys()),
                "chosen_action": action_idx,
                "reward": 0,  # Reward will be updated at showdown
                "done": False
            }
            self.rlogger.record_transition(transition_data)
            
            # FederatedPlayer specific code
            if isinstance(self.players[current_player], FederatedPlayer):
                local_transition = {
                    "local_state": "example_state",
                    "local_action": action_idx,
                    "local_reward": 0,
                    "local_done": False
                }
                self.players[current_player].record_local_transition(local_transition)
            
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
        
        # Calculate rewards (chip differences) for each player
        rewards = {}
        for i in range(self.num_players):
            rewards[i] = self.chips[i] - chips_before_round[i]
        
        # Update the transitions with proper rewards and done status
        # Group transitions by player and round
        player_round_transitions = {}
        for transition in self.rlogger.transitions:
            if transition["round"] == self.current_hand:
                player_id = transition["current_player"]
                round_key = f"{player_id}_{transition['round']}_{transition['stage']}"
                if round_key not in player_round_transitions:
                    player_round_transitions[round_key] = []
                player_round_transitions[round_key].append(transition)
        
        # For each player, mark only their last action in this round as terminal
        for round_key, transitions in player_round_transitions.items():
            # Sort by decision index to find the last action
            sorted_transitions = sorted(transitions, key=lambda t: t["decision_index"])
            if sorted_transitions:
                # Set rewards for all transitions
                player_id = sorted_transitions[0]["current_player"]
                reward = rewards.get(player_id, 0)
                
                # Set all transitions to have the reward but only the last one is terminal
                for t in sorted_transitions[:-1]:
                    t["reward"] = reward
                    t["done"] = False
                    
                # Mark the last transition as terminal
                sorted_transitions[-1]["reward"] = reward
                sorted_transitions[-1]["done"] = True
        
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
        """Write collected RL data to CSV files."""
        self.rlogger.write_transitions(self.log)
        
        # Flush local transitions for any FederatedPlayer
        for i, p in enumerate(self.players):
            if hasattr(p, 'flush_local_transitions'):
                p.flush_local_transitions()