# KuhnPokerEngine.py

import time
import random
import sys
from datetime import datetime

from engine.utilities import (
    flush_federated_transitions,
    RLDataLogger,
    log_message,
    record_local_transition_if_applicable
)

"""
Mappings for card names & actions
"""
# This dictionary maps indexes to card names in Kuhn Poker.
CARD_NAMES = {0: "J", 1: "Q", 2: "K", 3: "A"}
CHIP_TOTAL = 100 # Total chips each player starts with

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
        self.chips = [CHIP_TOTAL] * num_players
        self.current_hand = 0
        self.rlogger = RLDataLogger()
        self.log_file = None

    def log(self, message):
        """Wrapper around the utility logging function."""
        log_message(message)

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
        self.all_in = [False] * self.num_players  # Track which players are all-in
        self.cards = []
        chips_before_round = self.chips[:]  # Track starting chips for diff
        
        # Ante
        ante = 1
        players_with_chips = 0
        for i in range(self.num_players):
            if self.chips[i] >= ante:
                players_with_chips += 1
        
        # Only collect ante if there are at least 2 players with chips (to avoid forced folds)
        if players_with_chips >= 2:
            for i in range(self.num_players):
                if self.chips[i] >= ante:
                    self.chips[i] -= ante
                    self.pot += ante
                else:
                    self.log(f"Player {i} doesn't have enough chips for ante - forced to fold.")
                    self.folded[i] = True
        else:
            # If only one or no players have chips, don't collect ante
            for i in range(self.num_players):
                if self.chips[i] < ante:
                    self.log(f"Player {i} doesn't have enough chips for ante - forced to fold.")
                    self.folded[i] = True
        
        # Check if we have enough active players to continue
        active_players = [i for i in range(self.num_players) if not self.folded[i]]
        if len(active_players) <= 1:
            if active_players:
                winner = active_players[0]
                self.log(f"Only Player {winner} has enough chips to play. Player {winner} wins the pot of {self.pot} by default.")
                self.chips[winner] += self.pot
            else:
                self.log("No players have enough chips to play this hand.")
            
            # Use showdown to handle rewards and transition updates
            self.showdown(chips_before_round)
            return

        self.log(f"All active players ante {ante} unit. Pot is now {self.pot}.")

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
        last_raise_amount = 1  # Default minimum raise

        # public_state for more comprehensive information
        public_state = {
            "pot_size": self.pot,
            "current_bets": self.current_bets.copy(),
            "chip_counts": self.chips.copy(),
            "betting_history": [],  # Will be a list of action strings like ["check", "bet", "call"]
            "folded_players": self.folded.copy(),
            "num_players": self.num_players,
            "highest_bet": highest_bet,
            "last_bettor": last_bettor,
            "current_player": current_player,
            "round_num": round_num,
            "min_raise": last_raise_amount
        }

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
            
            # Update public_state with the latest information before calling get_action
            public_state["pot_size"] = self.pot
            public_state["current_bets"] = self.current_bets.copy()
            public_state["chip_counts"] = self.chips.copy()
            public_state["betting_history"] = actions.copy()
            public_state["folded_players"] = self.folded.copy()
            public_state["highest_bet"] = highest_bet
            public_state["last_bettor"] = last_bettor
            public_state["current_player"] = current_player
            public_state["player_id"] = current_player  # Explicitly tell the player which player they are
            
            # Get player action; now returns (action_idx, raise_amount)
            action_idx, raise_amount = (
                self.players[current_player].get_action(
                    self.cards[current_player],
                    available_actions,
                    round_num,
                    self.chips[current_player],
                    public_state
                )
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
                        
                        # Mark player as all-in if they can't fully match the bet
                        if call_amount < bet_diff:
                            self.all_in[current_player] = True
                            self.log(f"Player {current_player} calls all-in with {call_amount} chip(s). Pot is now {self.pot}.")
                        else:
                            self.log(f"Player {current_player} calls. Pot is now {self.pot}.")
                        
                        actions.append("call")
                        
                        # Check if all active players have had a chance to act after the last bet/raise
                        # and have equal bets or are all-in
                        active_players = [i for i in range(self.num_players) if not self.folded[i]]
                        
                        # Consider a player's bet "equal" if they're all-in with whatever they could contribute
                        all_bets_equal = all(self.current_bets[p] == highest_bet or self.all_in[p] for p in active_players)
                        
                        # If all players have acted since the last bet and bets are equal or players are all-in, round is complete
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
                    min_extra = 1  # Always set minimum raise to 1
                    max_extra = self.chips[current_player]
                    raise_used = min(max_extra, max(min_extra, raise_amount or min_extra))
                    if raise_used <= 0:
                        self.log(f"Player {current_player} can't raise further - forced to call or fold.")
                        if self.chips[current_player] > 0:
                            remaining_chips = self.chips[current_player]  # Store before setting to zero
                            self.chips[current_player] = 0
                            self.current_bets[current_player] += remaining_chips
                            self.pot += remaining_chips
                            self.log(f"Player {current_player} calls all-in with remaining {remaining_chips} chip(s). Pot is now {self.pot}.")
                            actions.append("call")
                            self.all_in[current_player] = True  # Mark as all-in
                        else:
                            self.folded[current_player] = True
                            actions.append("fold")
                            active_players_count -= 1
                    else:
                        self.chips[current_player] -= raise_used
                        self.current_bets[current_player] += raise_used
                        self.pot += raise_used
                        highest_bet = self.current_bets[current_player]
                        
                        # Check if player is all-in after the raise
                        if self.chips[current_player] == 0:
                            self.all_in[current_player] = True
                            self.log(f"Player {current_player} raises all-in by {raise_used} unit(s). Pot is now {self.pot}.")
                        else:
                            self.log(f"Player {current_player} raises by {raise_used} unit(s). Pot is now {self.pot}.")
                            
                        actions.append("raise")
                        last_bettor = current_player
                        # Reset players_acted counter when there's a new bet
                        players_acted = 1
                        # Update last_raise_amount so future raises must meet or exceed this
                        last_raise_amount = 1  # Always keep minimum raise as 1
                        public_state["min_raise"] = last_raise_amount
            
            # After processing the chosen action, only record local transitions if agent is FRL
            record_local_transition_if_applicable(self.players[current_player], action_idx)
            
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
                    "chips": ";".join([str(c) for c in self.chips]),  # Include all players' chips
                    "betting_history": ",".join(actions),  # More readable format with commas
                    "highest_bet": highest_bet,
                    "last_bettor": last_bettor if last_bettor is not None else -1
                },
                "legal_actions": list(available_actions.keys()),
                "chosen_action": action_idx,
                "reward": 0,  # Reward will be updated at showdown
                "done": False
            }
            self.rlogger.record_transition(transition_data)
            
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
            # Consider a player's bet "equal" if they're all-in with whatever they could contribute
            active_players = [i for i in range(self.num_players) if not self.folded[i]]
            equal_bets = all(self.current_bets[player] == highest_bet or self.all_in[player] for player in active_players)
            
            if last_bettor is not None:
                # Make at least one complete rotation after the last bet
                if players_acted >= active_players_count and equal_bets:
                    all_players_acted = True
            
            time.sleep(self.delay)
            sys.stdout.flush()
        
        # At the end, after processing all actions, update last_acted_player
        last_acted_player = (current_player - 1) % self.num_players
        
        # Add last_acted_player to the return value
        return actions, last_acted_player
        
    def showdown(self, chips_before_round):
        """
        Determine the winner and distribute the pot with side pot support.
        """
        print("\n" + "-" * 40 + " SHOWDOWN " + "-" * 40)
        self.log("=== Showdown ===")
        
        active_players = []
        # Show cards of non-folded players, but only if cards have been dealt
        if self.cards:
            reveal_message = "Cards revealed: "
            for i in range(self.num_players):
                if not self.folded[i]:
                    reveal_message += f"Player {i}: {self.cards[i]}, "
                    active_players.append(i)
            self.log(reveal_message.rstrip(", "))
            
            # If there's a hidden card in 3-player mode, log it at the showdown
            if self.num_players == 3 and hasattr(self, 'hidden_card'):
                self.log(f"(** Debug/Reveal **) The unused hidden card was: {self.hidden_card}")
        else:
            # If no cards were dealt, just collect active players
            for i in range(self.num_players):
                if not self.folded[i]:
                    active_players.append(i)
        
        # Only handle side pots in 3-player mode when multiple players remain
        if self.num_players == 3 and any(self.all_in) and len(active_players) > 1:
            self.distribute_with_side_pots(active_players, chips_before_round)
        else:
            # Original showdown logic for standard cases
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

    def distribute_with_side_pots(self, active_players, chips_before_round):
        """
        Distribute the pot considering side pots for all-in players.
        """
        # Sort active players by their bet amounts (all-in amounts)
        active_players_by_bet = sorted(active_players, key=lambda p: self.current_bets[p])
        
        # Card ranks for determining winners
        card_ranks = {"J": 1, "Q": 2, "K": 3, "A": 4}
        
        remaining_pot = self.pot
        processed_amount = 0
        eligible_players = active_players.copy()
        
        # Process each potential side pot
        for i, current_all_in_player in enumerate(active_players_by_bet):
            # Skip if player isn't all-in (will be part of the main pot)
            if not self.all_in[current_all_in_player]:
                continue
                
            current_bet = self.current_bets[current_all_in_player]
            
            # Calculate this side pot amount
            side_pot_amount = 0
            for player in range(self.num_players):
                # How much of this player's bet goes into this side pot
                if not self.folded[player]:
                    contribution = min(self.current_bets[player], current_bet) - processed_amount
                    if contribution > 0:
                        side_pot_amount += contribution
            
            if side_pot_amount <= 0:
                continue  # Skip if no money in this side pot
                
            # For this side pot, find the winner among eligible players
            if len(eligible_players) == 1:
                # Only one player eligible for this side pot
                winner = eligible_players[0]
            else:
                # Find highest card among eligible players
                best_card = -1
                winner = -1
                for player in eligible_players:
                    card_rank = card_ranks.get(self.cards[player], 0)
                    if card_rank > best_card:
                        best_card = card_rank
                        winner = player
            
            # Award this side pot to the winner
            self.chips[winner] += side_pot_amount
            remaining_pot -= side_pot_amount
            
            self.log(f"Player {winner} wins side pot of {side_pot_amount} with {self.cards[winner]}.")
            
            # Remove current all-in player from eligible players for next side pots
            if current_all_in_player in eligible_players:
                eligible_players.remove(current_all_in_player)
                
            # Update processed amount for next side pot
            processed_amount = current_bet
        
        # Handle the main pot (if anything remains)
        if remaining_pot > 0 and eligible_players:
            if len(eligible_players) == 1:
                winner = eligible_players[0]
            else:
                # Find highest card among remaining eligible players
                best_card = -1
                winner = -1
                for player in eligible_players:
                    card_rank = card_ranks.get(self.cards[player], 0)
                    if card_rank > best_card:
                        best_card = card_rank
                        winner = player
                        
            self.chips[winner] += remaining_pot
            self.log(f"Player {winner} wins main pot of {remaining_pot} with {self.cards[winner]}.")
        
        # Calculate rewards for RL updates
        rewards = {}
        for i in range(self.num_players):
            rewards[i] = self.chips[i] - chips_before_round[i]
        
        # Update transitions with rewards - copied from original showdown method
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
            sorted_transitions = sorted(transitions, key=lambda t: t["decision_index"])
            if sorted_transitions:
                player_id = sorted_transitions[0]["current_player"]
                reward = rewards.get(player_id, 0)
                
                for t in sorted_transitions[:-1]:
                    t["reward"] = reward
                    t["done"] = False
                    
                sorted_transitions[-1]["reward"] = reward
                sorted_transitions[-1]["done"] = True
    
    def write_transitions(self):
        """Write collected RL data to CSV files."""
        self.rlogger.write_transitions(self.log)
        
        # Flush local transitions for any FederatedPlayer
        for i, p in enumerate(self.players):
            if hasattr(p, 'local_transitions'):
                flush_federated_transitions(p)