import torch
import torch.nn.functional as F
import torch.optim as optim
from typing import Dict, List, Tuple, Any, Optional,
import itertools
import random
import matplotlib.pyplot as plt
import numpy as np
import os
import time
from collections import defaultdict, deque
import pandas as pd
import json
import copy


random.seed(1337)
np.random.seed(1999)
torch.manual_seed(42)

# Constants
NUM_PLAYERS = 3
CARDS = ('J', 'Q', 'K', 'A')
NUM_CARDS = len(CARDS)
CARD_MAP = {card: i for i, card in enumerate(CARDS)} # J=0, Q=1, K=2, A=3

class Actions:
    PASS, BET, RAISE, FOLD, CALL = 0, 1, 2, 3, 4
    LIST = (PASS, BET, RAISE, FOLD, CALL) # Tuple
    MAP_STR = {PASS: 'p', BET: 'b', RAISE: 'r', FOLD: 'f', CALL: 'c'}
    MAP_INT = {v: k for k, v in MAP_STR.items()}
NUM_ACTIONS = len(Actions.LIST)

# Game parameters
ANTE = 1.0
BET_AMOUNT = 1.0
RAISE_AMOUNT = 2.0 # Total amount after raising (Ante + Bet + Raise increment)
MAX_HISTORY_LEN = 6
INFOSER_FEATURE_SIZE = NUM_CARDS + (NUM_ACTIONS * MAX_HISTORY_LEN) # 4 + 5*6 = 34

class Kuhn3PlayerEnvironment:
    """
    A robust implementation of the 3-Player Kuhn Poker environment with a raise action.

    Rules:
    - 4 Cards: J, Q, K, A
    - 3 Players (Player 0, Player 1, Player 2)
    - Ante: 1 unit per player (Pot starts at 3)
    - Actions: Pass/Check(0), Bet(1), Raise(2), Fold(3), Call(4)
             Bet costs 1 unit. Raise costs an additional 1 unit (total bet 2).
    - Max 1 raise per betting round.
    - Game ends when only one player remains or betting concludes followed by showdown.
    """
    def __init__(self):
        self.card_permutations: List[Tuple[str, ...]] = list(itertools.permutations(CARDS, NUM_PLAYERS))
        # Cache for terminal utilities: Key=(history_tuple, sorted_cards_tuple) -> Tuple[float, float, float]
        self._terminal_utilities_cache: Dict[Tuple[Tuple[str,...], Tuple[str,...]], Tuple[float, float, float]] = {}

    def _get_initial_state(
            self,
            cards: Tuple[str, str, str]
            )-> Dict[str, Any]:
        """
        Initializes the game state with the given cards.
        The state dictionary contains:
        - cards: tuple of cards for each player
        - player_turn: index of the player whose turn it is
        - bets: list of bets for each player
        - active_players: set of players who haven't folded
        - history_actions: list of actions taken in the current round
        - pot: total amount in the pot
        - num_raises: number of raises in the current round
        - betting_closed: boolean indicating if betting is closed
        - is_terminal: boolean indicating if the game has ended
        - terminal_utility: utility for each player at the terminal state

        Args:
            cards: Tuple[str, str, str], Tuple of cards for each player.
        
        Returns:
            Dict[str, Any]: The initial state dictionary.
        """
        """Returns the initial state dictionary after dealing cards."""
        return {
            "cards": cards,
            "player_turn": 0,
            "bets": [0.0] * NUM_PLAYERS, # Bets committed in this round (excluding ante)
            "active_players": set(range(NUM_PLAYERS)), # Players who haven't folded
            "history_actions": [], # List of (player_idx, action_int)
            "pot": ANTE * NUM_PLAYERS,
            "num_raises": 0, # Count raises in the current round
            "betting_closed": False,
            "is_terminal": False,
            "terminal_utility": None
        }

    def get_current_player(
            self,
            state: Dict[str, Any]
            ) -> int:
         """
         Returns the index of the player whose turn it is.
         Args:
            state: Dict[str, Any], The current game state.
         
         Returns:
            The index of the player whose turn it is.
         """
         return state["player_turn"]

    def get_legal_actions(
            self,
            state: Dict[str, Any]
            ) -> List[int]:
        """
        Returns the list of legal action integers for the current state.

        Args:
            state: Dict[str, Any], The current game state.
        
        Returns:
            List[int]: A list of legal action integers.
        """
        if state["is_terminal"]:
            return []

        player = state["player_turn"]
        if player not in state["active_players"]:
             # Should ideally not happen if turn logic is correct, but handle defensively
             return []

        bets = state["bets"]
        current_bet = bets[player]
        max_bet = max(bets[idx] for idx in state["active_players"])
        num_raises = state["num_raises"]
        can_raise = num_raises == 0

        legal = []

        # Can always fold (unless everyone else already folded, which is terminal)
        if len(state["active_players"]) > 1:
             legal.append(Actions.FOLD)

        if current_bet == max_bet:
            # Player is facing no bet or has matched the highest bet
            legal.append(Actions.PASS) # Check
            if max_bet == 0: # Can open bet
                 legal.append(Actions.BET)
            elif can_raise: # Matched a bet, can raise if limit not reached
                 legal.append(Actions.RAISE)
        else:
             # Player is facing a bet/raise
             legal.append(Actions.CALL) # Call the difference
             if can_raise and max_bet < RAISE_AMOUNT: # Can raise if limit not reached and facing a bet (not already a raise)
                 legal.append(Actions.RAISE)

        # Filter out invalid actions (e.g., trying to raise when already max bet)
        final_legal = []
        for action in legal:
            if action == Actions.RAISE and max_bet >= RAISE_AMOUNT: # Cannot raise if current max bet is already the raised amount
                 continue
            final_legal.append(action)

        # Ensure PASS is only valid if check is possible (no bet pending)
        if Actions.PASS in final_legal and max_bet > current_bet:
            final_legal.remove(Actions.PASS)
            # If PASS was the *only* option other than FOLD, something is wrong. Add CALL back if removed implicitly.
            if Actions.CALL not in final_legal and len(state["active_players"]) > 1:
                 if max_bet > current_bet: # Facing a bet
                      final_legal.append(Actions.CALL)

        # If only FOLD is possible, it implies an error or specific end-game state not handled
        if not final_legal and len(state["active_players"]) > 1:
             print(f"Warning: No legal actions derived for state: {state}") # Debug help
             # This might happen if a player must call all-in but logic doesn't handle it.
             # For basic Kuhn, let's assume this doesn't occur.
             return [Actions.FOLD] # Default to fold if logic fails

        return sorted(list(set(final_legal)))


    def _get_next_player(
            self,
            state: Dict[str, Any]
            ) -> int:
        """
        Find the next active player in sequence.

        Args:
            state: Dict[str, Any], The current game state.
        
        Returns:
            int: The index of the next active player.
        """
        current_player = state["player_turn"]
        next_p = (current_player + 1) % NUM_PLAYERS
        while next_p not in state["active_players"]:
             next_p = (next_p + 1) % NUM_PLAYERS
             if next_p == current_player: # Cycled all the way around
                 return -1 # Should indicate betting closed or error
        return next_p


    def step(
            self,
            state: Dict[str, Any], 
            action: int
            ) -> Dict[str, Any]:
        """
        Applies an action to the state, returning the new state.
        Does not modify the input state dictionary.

        Args:
            state: Dict[str, Any], The current game state.
            action: int, The action to apply.
        
        Returns:
            Dict[str, Any]: The new game state after applying the action.
        """
        if state["is_terminal"]:
            # print("Warning: Step called on terminal state.")
            return state
        if action not in self.get_legal_actions(state):
            legal_strs = [Actions.MAP_STR.get(a, '?') for a in self.get_legal_actions(state)]
            action_str = Actions.MAP_STR.get(action, '?')
            history_str = self.get_history_string(state)
            raise ValueError(f"Illegal action '{action_str}' ({action}) for history '{history_str}'. Legal: {legal_strs}. State: {state}")

        new_state = copy.deepcopy(state) # Work on a copy
        player = new_state["player_turn"]
        bets = new_state["bets"]
        max_bet = max(bets[idx] for idx in new_state["active_players"]) if new_state["active_players"] else 0

        new_state["history_actions"].append((player, action))

        if action == Actions.FOLD:
            new_state["active_players"].remove(player)
            # Don't reset player's bet, their contribution stays in pot
        elif action == Actions.PASS:
            # Bet remains the same (should be equal to max_bet or 0)
            pass # No change in bets or pot needed
        elif action == Actions.CALL:
            amount_to_call = max_bet - bets[player]
            new_state["bets"][player] += amount_to_call
            new_state["pot"] += amount_to_call
        elif action == Actions.BET:
            amount_to_bet = BET_AMOUNT - bets[player] # Should be BET_AMOUNT if bets[player] was 0
            new_state["bets"][player] += amount_to_bet
            new_state["pot"] += amount_to_bet
            max_bet = new_state["bets"][player] # Update max_bet conceptually for next player
        elif action == Actions.RAISE:
            # Raise brings the total bet *or this player up to RAISE_AMOUNT
            amount_to_raise = RAISE_AMOUNT - bets[player]
            new_state["bets"][player] += amount_to_raise
            new_state["pot"] += amount_to_raise
            new_state["num_raises"] += 1
            max_bet = new_state["bets"][player] # Update max_bet

        # Check for Round End / Terminal State
        remaining_active = new_state["active_players"]
        if len(remaining_active) == 1:
            new_state["is_terminal"] = True
            new_state["betting_closed"] = True
            new_state["terminal_utility"] = self._calculate_payoffs(new_state)
            new_state["player_turn"] = -1
            return new_state

        # Check if betting round is closed
        # Betting closes if:
        # 1. All active players have acted since the last bet/raise.
        # 2. All active players have committed the same amount (or folded).
        # 3. Or, if everyone just checked around (passed when max_bet was 0).
        current_max_bet_active = max(bets[idx] for idx in remaining_active) if remaining_active else 0
        all_bets_equal = all(new_state["bets"][idx] == current_max_bet_active for idx in remaining_active)

        action_closed_round = False
        if action in [Actions.CALL, Actions.PASS]:
             if all_bets_equal:
                 action_closed_round = True

        # Special case: Initial check-around p-p-p
        if len(new_state["history_actions"]) == NUM_PLAYERS and all(a == Actions.PASS for _, a in new_state["history_actions"]):
             action_closed_round = True

        if action_closed_round:
             new_state["betting_closed"] = True
             new_state["is_terminal"] = True # In Kuhn, betting closes means showdown
             new_state["terminal_utility"] = self._calculate_payoffs(new_state)
             new_state["player_turn"] = -1
        else:
             # Find next player
             new_state["player_turn"] = self._get_next_player(new_state)
             if new_state["player_turn"] == -1: # Should not happen if not terminal, indicates error
                  print(f"Error: Could not find next player in non-terminal state: {new_state}")
                  # Force terminal state to avoid infinite loop
                  new_state["is_terminal"] = True
                  new_state["betting_closed"] = True
                  new_state["terminal_utility"] = self._calculate_payoffs(new_state)

        return new_state


    def is_terminal(self, state: Dict[str, Any]) -> bool:
        """Checks if the state represents a terminal state."""
        return state["is_terminal"]

    def _calculate_payoffs(self, state: Dict[str, Any]) -> Tuple[float, ...]:
        """
        Internal helper to calculate payoffs based on the terminal state.
        Returns tuple (payoff_p0, payoff_p1, payoff_p2).
        """
        # Use cache if available
        history_tuple = tuple(Actions.MAP_STR[a] for _, a in state["history_actions"])
        # Sort cards for canonical cache key, but use original order for winner determination
        cards_tuple_sorted = tuple(sorted(state["cards"]))
        cache_key = (history_tuple, cards_tuple_sorted)

        if cache_key in self._terminal_utilities_cache:
             return self._terminal_utilities_cache[cache_key]


        payoffs = [0.0] * NUM_PLAYERS
        active_players = list(state["active_players"]) # Who remains?
        cards = state["cards"]
        pot = state["pot"]

        if len(active_players) == 1:
            # Winner by fold
            winner = active_players[0]
            for p in range(NUM_PLAYERS):
                amount_contributed = ANTE + state["bets"][p] # Ante + bets this round
                if p == winner:
                     payoffs[p] = pot - amount_contributed
                else:
                     payoffs[p] = -amount_contributed
        else:
            # Showdown
            # Find winner(s) among active players
            winning_rank = -1
            winners = []
            for p in active_players:
                 rank = CARD_MAP[cards[p]]
                 if rank > winning_rank:
                      winning_rank = rank
                      winners = [p]
                 elif rank == winning_rank:
                      winners.append(p) # Handle ties

            # Split pot among winners
            prize_per_winner = pot / len(winners)

            for p in range(NUM_PLAYERS):
                amount_contributed = ANTE + state["bets"][p]
                if p in winners:
                    payoffs[p] = prize_per_winner - amount_contributed
                elif p in active_players: # Lost at showdown
                     payoffs[p] = -amount_contributed
                else: # Folded earlier
                     payoffs[p] = -amount_contributed


        # Verify zero-sum
        if abs(sum(payoffs)) > 1e-6:
            print(f"Warning: Payoffs do not sum to zero! State: {state}, Payoffs: {payoffs}")

        result = tuple(payoffs)
        self._terminal_utilities_cache[cache_key] = result
        return result

    def get_utility(self, state: Dict[str, Any], player: int) -> float:
        """
        Returns the utility for the specified player at a terminal state.
        Raises ValueError if the state is not terminal or player index is invalid.
        """
        if not state["is_terminal"]:
            raise ValueError(f"get_utility called on non-terminal state: {state}")
        if player < 0 or player >= NUM_PLAYERS:
            raise ValueError(f"Invalid player index {player} requested for utility.")

        if state["terminal_utility"] is None:
             # Should have been calculated when state became terminal, but calculate defensively
             state["terminal_utility"] = self._calculate_payoffs(state)

        return state["terminal_utility"][player]

    def get_history_string(self, state: Dict[str, Any]) -> str:
        """Returns the history string representation (e.g., 'pbfr')."""
        return "".join(Actions.MAP_STR[action] for _, action in state["history_actions"])

    def get_infoset(self, state: Dict[str, Any], player: int) -> str:
        """
        Returns the information set string for the player.
        Format: PlayerCard + HistoryString
        """
        if player < 0 or player >= NUM_PLAYERS:
            raise ValueError(f"Invalid player index {player} requested for infoset.")
        # cards is expected to be (p0_card, p1_card, p2_card)
        history_str = self.get_history_string(state)
        return state["cards"][player] + history_str

    def get_all_card_deals(self) -> List[Tuple[str, ...]]:
        """Returns all possible card dealings (permutations)."""
        return self.card_permutations

    def get_infoset_tensor(self, infoset_str: str, device: torch.device) -> torch.Tensor:
        """Converts infoset string to a fixed-size tensor."""
        card = infoset_str[0]
        history = infoset_str[1:]
        card_idx = CARD_MAP[card]
        card_one_hot = F.one_hot(torch.tensor(card_idx), num_classes=NUM_CARDS)

        # Encode history actions
        history_indices = []
        for action_char in history:
             action_int = Actions.MAP_INT.get(action_char)
             if action_int is not None: # Should always be found
                  history_indices.append(action_int)
             else:
                  print(f"Warning: Unknown action character '{action_char}' in history '{history}'")


        if history_indices:
            if len(history_indices) > MAX_HISTORY_LEN:
                 history_indices = history_indices[:MAX_HISTORY_LEN]
                 # print(f"Warning: History length exceeded MAX_HISTORY_LEN. Truncated.")

            history_one_hot = F.one_hot(torch.tensor(history_indices), num_classes=NUM_ACTIONS)
            flat_history = history_one_hot.view(-1) # Shape: [len(history) * NUM_ACTIONS]
        else:
            flat_history = torch.empty(0, dtype=torch.float32)

        padding_needed = (NUM_ACTIONS * MAX_HISTORY_LEN) - len(flat_history)
        if padding_needed < 0:
            print(f"Error: Negative padding needed for history. len={len(flat_history)}")
            padding_needed = 0
            flat_history = flat_history[:(NUM_ACTIONS * MAX_HISTORY_LEN)]


        history_padding = torch.zeros(padding_needed, dtype=torch.float32)

        # Concatenate history and padding
        final_history_tensor = torch.cat((flat_history.float(), history_padding))

        # Concatenate card features and history features
        infoset_tensor = torch.cat((card_one_hot.float(), final_history_tensor))

        # Add batch dimension and send to device
        return infoset_tensor.to(device).unsqueeze(0) # Shape: [1, INFOSER_FEATURE_SIZE]

# ===================================
# CFR Policy
# ===================================
def get_cfr_policy(
        env: Kuhn3PlayerEnvironment, # Type hint updated
        v_network: torch.nn.Module,
        infoset_str: str,
        legal_actions: List[int],
        device: torch.device
        ) -> Dict[int, float]:
    """ Gets CFR policy from V network (Regret Matching). """
    if not legal_actions: return {}
    infoset_tensor = env.get_infoset_tensor(infoset_str, device)
    with torch.no_grad():
        # V network output represents *cumulative* regret estimates for each action
        regret_outputs = v_network(infoset_tensor).squeeze(0) # Shape [NUM_ACTIONS]

    # Perform regret matching on legal actions
    positive_regrets = {action: max(regret_outputs[action].item(), 0.0) for action in legal_actions}
    sum_positive_regrets = sum(positive_regrets.values())

    policy = {}
    if sum_positive_regrets > 1e-6:
        for action in legal_actions:
            policy[action] = positive_regrets[action] / sum_positive_regrets
    else:
        num_legal = len(legal_actions)
        policy = {action: 1.0 / num_legal for action in legal_actions}

    # Normalize again to be safe
    final_sum = sum(policy.values())
    if abs(final_sum - 1.0) > 1e-5 and final_sum > 1e-6:
       # print(f"Warning: CFR policy normalization needed. Sum={final_sum}, Policy={policy}")
       for action in legal_actions: policy[action] /= final_sum
    elif final_sum == 0 and len(legal_actions)>0:
        num_legal = len(legal_actions)
        policy = {action: 1.0 / num_legal for action in legal_actions}
    return policy


# ==================================
# Average Policy
# ===================================
def get_avg_policy(
        env: Kuhn3PlayerEnvironment,
        pi_network: torch.nn.Module,
        infoset_str: str, legal_actions: List[int],
        device: torch.device
        )-> Dict[int, float]:
    """ Gets average policy from Pi network (applying softmax). """
    if not legal_actions: return {}
    infoset_tensor = env.get_infoset_tensor(infoset_str, device)
    with torch.no_grad():
        # Apply softmax to Pi network logits
        action_logits = pi_network(infoset_tensor).squeeze(0) # Shape [NUM_ACTIONS]
        action_probs = F.softmax(action_logits, dim=0)

    # Select probabilities for legal actions
    legal_probs = {action: action_probs[action].item() for action in legal_actions}
    prob_sum = sum(v for v in legal_probs.values() if v > 0) # Sum only positive probs

    policy = {}
    if prob_sum > 1e-6: # Use tolerance
        # Normalize probabilities over legal actions
        for action in legal_actions:
            policy[action] = max(0.0, legal_probs[action]) / prob_sum # Ensure non-negative
        # Renormalize for safety due to potential floating point issues
        final_sum = sum(policy.values())
        if abs(final_sum - 1.0) > 1e-5 and final_sum > 1e-6:
            # print(f"Warning: Avg policy normalization needed. Sum={final_sum}, Policy={policy}")
            for action in legal_actions: policy[action] /= final_sum
    else:
        # Fallback to uniform if sum is too small or no legal actions have probability > 0
        num_legal = len(legal_actions)
        if num_legal > 0:
            policy = {action: 1.0 / num_legal for action in legal_actions}
    return policy


# =======================================
# Regret Estimation & Expected Utility 
# =======================================
memo_reach_prob = {}

def get_others_reach_probability(
        target_state: Dict[str, Any],
        target_player: int,
        pi_network: torch.nn.Module,
        environment: Kuhn3PlayerEnvironment,
        device: torch.device,
        memo: Dict[Tuple[str, Tuple[str,...], int], float]
        ) -> float:
    """
    Calculate the product of reach probabilities of all players *other than* target_player
    to reach the target_state, assuming they play according to pi_network (average policy).
    Uses memoization.

    Args:
        target_state: Dict[str, Any], The target state to reach.
        target_player: int, The player whose reach probability we are calculating.
        pi_network: torch.nn.Module, The network for average policy.
        environment: Kuhn3PlayerEnvironment, The game environment.
        device: torch.device, The device for PyTorch operations.
        memo: Dict[Tuple[str, Tuple[str,...], int], float], Memoization dictionary.
    
    Returns:
        float: The reach probability for the target player.
    """
    history_tuple = tuple(Actions.MAP_STR[a] for _, a in target_state["history_actions"])
    cards_tuple = tuple(sorted(target_state["cards"]))
    memo_key = (history_tuple, cards_tuple, target_player)

    if memo_key in memo:
        return memo[memo_key]

    # Probability of this specific deal (cards)
    initial_deal_prob = 1.0 / len(environment.card_permutations)
    reach_prob = initial_deal_prob

    # Simulate history from the start, multiplying probabilities
    # Need to reconstruct intermediate states, which is inefficient.
    # Better: Pass history list and reconstruct state progressively?
    # Or, ideally, the caller provides the path probability to the *parent* state.

    # Let's recalculate from scratch for simplicity here, although less efficient.
    current_cards = target_state["cards"]
    sim_state = environment._get_initial_state(current_cards)

    for player_idx, action in target_state["history_actions"]:
        # Whose turn was it at sim_state?
        acting_player = environment.get_current_player(sim_state)
        if acting_player != player_idx:
            # This indicates an issue with state progression or history tracking
            print(f"Error: History action player {player_idx} doesn't match state turn {acting_player}")
            reach_prob = 0.0
            break

        if acting_player != target_player:
            # This action was taken by someone else. Multiply their policy prob.
            infoset = environment.get_infoset(sim_state, acting_player)
            legal_actions = environment.get_legal_actions(sim_state)
            policy = get_avg_policy(environment, pi_network, infoset, legal_actions, device)
            action_prob = policy.get(action, 0.0)

            reach_prob *= action_prob
            if reach_prob <= 1e-9: # Probability vanished
                reach_prob = 0.0
                break

        # Advance the simulation state
        sim_state = environment.step(sim_state, action)
        if sim_state["is_terminal"] and len(history_tuple) > len(sim_state["history_actions"]):
             # Reached terminal state earlier than expected in simulation
             reach_prob = 0.0
             break

    memo[memo_key] = reach_prob
    return reach_prob


def get_expected_utility_regret_calc(
    current_state: Dict[str, Any],
    player_p: int, # The player whose utility we are calculating
    v_network: torch.nn.Module, # Player p uses V network (CFR policy)
    pi_network: torch.nn.Module, # Opponents use Pi network (Avg policy)
    environment: Kuhn3PlayerEnvironment,
    device: torch.device,
    memo: Dict[Tuple[str, Tuple[str,...], int], float] # Memoization dictionary
) -> float:
    """
    Calculates the expected utility for player_p starting from current_state.
    Player p plays according to CFR policy (derived from v_network).
    Other players play according to average policy (pi_network).
    Uses memoization.
    """
    if environment.is_terminal(current_state):
        return environment.get_utility(current_state, player_p)

    # Create memo key: (history_string, sorted_cards, player_p)
    history_str = environment.get_history_string(current_state)
    cards_tuple = tuple(sorted(current_state["cards"]))
    memo_key = (history_str, cards_tuple, player_p)

    if memo_key in memo:
        return memo[memo_key]

    current_player = environment.get_current_player(current_state)
    legal_actions = environment.get_legal_actions(current_state)
    if not legal_actions:
         # Should only happen in terminal states, handled above
         print(f"Warning: No legal actions in non-terminal state for regret calc: {current_state}")
         return 0.0 # Or handle as error

    current_infoset = environment.get_infoset(current_state, current_player)
    expected_value = 0.0

    # Determine policy based on whose turn it is
    if current_player == player_p:
        # Our target player uses the CFR policy (derived from V net)
        strategy_dist = get_cfr_policy(environment, v_network, current_infoset, legal_actions, device)
    else:
        # Other players use the average policy (Pi net)
        strategy_dist = get_avg_policy(environment, pi_network, current_infoset, legal_actions, device)

    # Calculate expected value by summing over legal actions
    for action in legal_actions:
        prob = strategy_dist.get(action, 0.0)
        if prob > 1e-9: # Only explore actions with non-negligible probability
            next_state = environment.step(current_state, action)
            expected_value += prob * get_expected_utility_regret_calc(
                next_state, player_p, v_network, pi_network, environment, device, memo
            )

    memo[memo_key] = expected_value
    return expected_value


def estimate_instantaneous_regrets(
    I_t_state: Dict[str, Any], # The state corresponding to infoset I_t
    pi_network: torch.nn.Module,
    v_network: torch.nn.Module,
    environment: Kuhn3PlayerEnvironment,
    all_possible_hands: List[Tuple[str,...]],
    device: torch.device
) -> Optional[Dict[int, float]]:
    """
    Estimates instantaneous regrets r_t(I, a) for all actions 'a' at the infoset I
    represented by I_t_state. This is the core of the Deep CFR update for the V network.
    Returns None if calculation is not possible (e.g., terminal state).
    """
    if environment.is_terminal(I_t_state):
        return None # No regrets at terminal states

    player_p = environment.get_current_player(I_t_state)
    player_p_card = I_t_state["cards"][player_p]
    history_at_I_t_str = environment.get_history_string(I_t_state)
    infoset_str = player_p_card + history_at_I_t_str # The actual infoset string I_t

    legal_actions_at_I_t = environment.get_legal_actions(I_t_state)
    if not legal_actions_at_I_t:
        # Should only happen if terminal, but check defensively
        print(f"Warning: No legal actions for regret estimation at non-terminal state: {I_t_state}")
        return None

    total_others_reach_prob_sum = 0.0
    # Stores sum over hands of: other_reach_prob * (cf_utility(action) - cf_utility(current_policy))
    cumulative_regrets: Dict[int, float] = {action: 0.0 for action in legal_actions_at_I_t}
    valid_hand_count = 0

    # --- Memoization Scope ---
    # Clear/create memo dictionaries needed for calculations within this function
    memo_reach_prob.clear() # Global memo for reach probabilities
    memo_exp_utility = {} # Local memo for expected utility calls within this estimation

    # Iterate over all possible complete hands (deals)
    for hand_cards in all_possible_hands:
        # Does this hand match the partial information (player_p's card) of the infoset?
        if hand_cards[player_p] != player_p_card:
            continue # Skip hands inconsistent with the infoset

        # Construct the state corresponding to this *specific* hand reaching the *same history*
        # This requires simulating the history with these specific cards to ensure it's possible.
        # This is complex. Let's *assume* I_t_state already correctly reflects the history,
        # and we just need to evaluate counterfactuals *given* this state but *weighted*
        # by the probability of reaching this state with this specific hand assignment.

        # Create a state copy representing this specific hand at the current history point
        current_state_for_hand = copy.deepcopy(I_t_state)
        current_state_for_hand["cards"] = hand_cards # Set the full hand

        # Calculate the reach probability of *other* players for this specific hand
        # Requires the state object corresponding to history_at_I_t_str and hand_cards
        others_reach_prob = get_others_reach_probability(
            current_state_for_hand, player_p, pi_network, environment, device, memo_reach_prob
        )

        if others_reach_prob <= 1e-9: # If others couldn't reach this state with this hand, skip
            continue

        valid_hand_count += 1
        total_others_reach_prob_sum += others_reach_prob

        # --- Calculate Counterfactual Values (Expected Utility) for each action ---
        # v(I, a) = Expected utility if player p takes action 'a' at infoset I,
        #           and thereafter p plays CFR policy (V net) and others play Avg policy (Pi net).
        cf_values_per_action_hand: Dict[int, float] = {}

        for action_prime in legal_actions_at_I_t:
            # Simulate taking action 'a' from the current state with this hand
            next_state_after_action = environment.step(current_state_for_hand, action_prime)

            # Calculate the expected utility from the resulting state onwards
            cf_values_per_action_hand[action_prime] = get_expected_utility_regret_calc(
                next_state_after_action, player_p, v_network, pi_network, environment, device, memo_exp_utility
            )

        # Calculate the expected value of the current infoset under the current CFR policy
        current_cfr_policy_at_I_t = get_cfr_policy(environment, v_network, infoset_str, legal_actions_at_I_t, device)
        v_I_t_hand = 0.0
        for action_double_prime in legal_actions_at_I_t:
            prob = current_cfr_policy_at_I_t.get(action_double_prime, 0.0)
            # Use the cf_value already computed for this action
            v_I_t_hand += prob * cf_values_per_action_hand.get(action_double_prime, 0.0) # Default to 0 if action missing? Should not happen.

        for action_prime in legal_actions_at_I_t:
            regret_contribution = others_reach_prob * (cf_values_per_action_hand[action_prime] - v_I_t_hand)
            cumulative_regrets[action_prime] += regret_contribution

    # Normalize the accumulated regrets by the total reach probability sum
    if valid_hand_count == 0 or total_others_reach_prob_sum <= 1e-9:
        # print(f"Warning: Could not estimate regrets for infoset {infoset_str}. Reach prob sum: {total_others_reach_prob_sum}")
        return None # Cannot compute regrets if no valid hands reach this state

    # Average instantaneous regret
    final_regrets = {action: cum_regret / total_others_reach_prob_sum
                     for action, cum_regret in cumulative_regrets.items()}

    return final_regrets


# --- Replay Buffers ---
class ReservoirBuffer:
    """Reservoir buffer using deque for potential efficiency gains."""
    def __init__(self, capacity):
        self.capacity = int(capacity) # Ensure integer capacity
        self.memory = deque(maxlen=self.capacity) # Use deque
        self.position = 0 # Tracks total number of items pushed

    def push(self, item):
        # Deque automatically handles capacity limit
        self.memory.append(item)
        self.position += 1

    def sample(self, batch_size):
        # Ensure batch_size is not larger than current memory size
        actual_batch_size = min(batch_size, len(self.memory))
        if actual_batch_size <= 0:
             return []
        # random.sample works directly on deques
        return random.sample(self.memory, actual_batch_size)

    def __len__(self):
        return len(self.memory)

# --- Network Definition (Unchanged conceptually) ---
class KuhnNetwork(torch.nn.Module):
    """ Simple MLP for Kuhn Poker """
    def __init__(self, input_size, output_size, hidden_size=128): # Default hidden size increase
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, output_size)
        )
        self.output_size = output_size # Store for reference if needed

    def forward(self, x):
        # Input x shape: [batch_size, input_size]
        out = self.net(x)
        # Output shape: [batch_size, output_size]
        # Softmax for Pi network is applied *outside* this forward pass (in get_avg_policy)
        # V network output (regrets) are used directly.
        return out


# ===================================
# Exploitability Calculation
# ===================================
memo_br = {}
def get_best_response_value_recursive(
    current_state: Dict[str, Any],
    br_player: int,
    fixed_policy_net: torch.nn.Module,
    environment: Kuhn3PlayerEnvironment,
    device: torch.device,
    memo: Dict[Tuple[str, Tuple[str,...], int], float]
) -> float:
    """
    Calculates the value for br_player assuming they play optimally
    against the fixed_policy_net played by all other players.
    Uses memoization.

    Args:
        current_state: Dict[str, Any], The current game state.
        br_player: int, The player index for the best response calculation.
        fixed_policy_net: torch.nn.Module, The network for the fixed policy (Π).
        environment: Kuhn3PlayerEnvironment, The game environment.
        device: torch.device, The device for PyTorch operations.
        memo: Dict[Tuple[str, Tuple[str,...], int], float], Memoization dictionary.
    
    Returns:
        float: The value for the best response player.
    """
    if environment.is_terminal(current_state):
        return environment.get_utility(current_state, br_player)

    # Create memo key: (history_string, sorted_cards, br_player)
    history_str = environment.get_history_string(current_state)
    cards_tuple = tuple(sorted(current_state["cards"]))
    memo_key = (history_str, cards_tuple, br_player)

    if memo_key in memo:
        return memo[memo_key]

    current_player_turn = environment.get_current_player(current_state)
    legal_actions = environment.get_legal_actions(current_state)

    if not legal_actions:
         print(f"Warning: No legal actions in non-terminal state for BR calc: {current_state}")
         if environment.is_terminal(current_state):
              return environment.get_utility(current_state, br_player)
         else: 
              return 0.0

    node_value = 0.0
    if current_player_turn == br_player:
        # Best response player maximizes their value
        best_action_value = -float('inf')
        for action in legal_actions:
            next_state = environment.step(current_state, action)
            action_value = get_best_response_value_recursive(
                next_state, br_player, fixed_policy_net, environment, device, memo
            )
            best_action_value = max(best_action_value, action_value)
        node_value = best_action_value
    else:
        # Fixed policy player's turn
        # Use the average policy (Pi network) for this player
        fixed_player = current_player_turn
        current_infoset = environment.get_infoset(current_state, fixed_player)
        strategy_dist = get_avg_policy(environment, fixed_policy_net, current_infoset, legal_actions, device)

        expected_value = 0.0
        for action in legal_actions:
            prob = strategy_dist.get(action, 0.0)
            if prob > 1e-9:
                next_state = environment.step(current_state, action)
                expected_value += prob * get_best_response_value_recursive(
                     next_state, br_player, fixed_policy_net, environment, device, memo
                 )
        node_value = expected_value

    memo[memo_key] = node_value
    return node_value


def calculate_exploitability(
    pi_network: torch.nn.Module, # The average policy network (Π) to evaluate
    environment: Kuhn3PlayerEnvironment,
    device: torch.device
) -> float:
    """
    Calculates the average exploitability of the pi_network in the N-player game.
    Defined as the average best-response value against the fixed policy Π.
    Exploitability = (1/N) * Sum_{i=0..N-1} [ Value(BR_i vs Π_{-i}) ]

    Args:
        pi_network: torch.nn.Module, The average policy network (Π).
        environment: Kuhn3PlayerEnvironment, The game environment.
        device: torch.device, The device for PyTorch operations.
    
    Returns:
        float: The average exploitability of the pi_network.
    """
    total_br_value_sum = 0.0
    all_hands = environment.get_all_card_deals()
    num_deals = len(all_hands)
    if num_deals == 0: return 0.0

    pi_network.eval()

    # Calculate BR value for each player against the others playing Pi
    for br_player_idx in range(NUM_PLAYERS):
        player_br_value_sum = 0.0
        memo_br.clear() # Clear memoization for each player's BR calculation

        for cards in all_hands:
            initial_state = environment._get_initial_state(cards)
            # Start recursion from the initial state after dealing
            player_br_value_sum += get_best_response_value_recursive(
                initial_state, br_player_idx, pi_network, environment, device, memo_br
            )

        # Average BR value for this player over all deals
        avg_br_value_player = player_br_value_sum / num_deals
        total_br_value_sum += avg_br_value_player
        # print(f"Avg BR Value for Player {br_player_idx}: {avg_br_value_player:.6f}")

    # Average exploitability = average BR value across all players
    exploitability = total_br_value_sum / NUM_PLAYERS
    return exploitability


# ====================================
# Plotting Function
# ====================================
def plot_results(
        results: dict,
        window_size: int,
        filename_prefix: str
        ) -> None:
    """
    Plots rewards and exploitability.

    Args:
        results (dict): Results dictionary containing episodes, rewards, and exploitability.
        window_size (int): Window size for rolling average.
        filename_prefix (str): Prefix for the output plot filename.
    """
    if not results['episodes']:
        print("No results to plot.")
        return

    episodes = results['episodes']
    # Use average reward if available, otherwise P0
    rewards_key = 'average_rewards' if 'average_rewards' in results else 'p0_rewards'
    rewards_label = 'Average Reward per Episode' if rewards_key == 'average_rewards' else 'P0 Reward per Episode'

    raw_rewards = results[rewards_key]
    cum_rewards = results['cumulative_' + rewards_key] # Match the key used in training
    exploitability = results['exploitability']
    exploit_episodes = results['exploitability_episodes']

    plt.style.use('seaborn-v0_8-darkgrid') # Nicer style
    fig, axes = plt.subplots(3, 1, figsize=(12, 18), sharex=True) # 3 rows, 1 column

    # 1. Raw Rewards (with rolling average)
    axes[0].plot(episodes, raw_rewards, label=f'Raw {rewards_label}', alpha=0.3)
    if len(raw_rewards) >= window_size:
        # Ensure window size is not larger than data length for convolution
        effective_window = min(window_size, len(raw_rewards))
        rolling_avg = np.convolve(raw_rewards, np.ones(effective_window)/effective_window, mode='valid')
        # Adjust episode indices for rolling average plot
        start_index = effective_window - 1
        axes[0].plot(episodes[start_index:], rolling_avg, label=f'Smoothed Reward ({effective_window}-ep avg)', color='red')
    axes[0].set_ylabel("Reward")
    axes[0].set_title("Episode Rewards")
    axes[0].legend()
    axes[0].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False) # Hide x-axis labels except last

    # 2. Cumulative Rewards
    axes[1].plot(episodes, cum_rewards, label=f'Cumulative {rewards_label}', color='green')
    axes[1].set_ylabel("Total Reward")
    axes[1].set_title("Cumulative Reward Over Training")
    axes[1].legend()
    axes[1].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False) # Hide x-axis labels except last


    # 3. Exploitability
    if exploit_episodes:
        axes[2].plot(exploit_episodes, exploitability, label='Average Exploitability', marker='o', linestyle='-', color='purple')
        axes[2].set_xlabel("Episode")
        axes[2].set_ylabel("Exploitability (Avg BR value)")
        axes[2].set_title("Policy Exploitability Over Training")
        axes[2].legend()
        # Exploitability might not always decrease smoothly, log scale can be helpful
        try:
             axes[2].set_yscale('log')
             axes[2].grid(True, which="both", ls="--")
        except ValueError:
             print("Warning: Could not set y-axis to log scale for exploitability (possibly zero or negative values).")
             axes[2].grid(True, ls="--")

    else:
        axes[2].text(0.5, 0.5, 'No exploitability data recorded', horizontalalignment='center', verticalalignment='center', transform=axes[2].transAxes)
        axes[2].set_xlabel("Episode")


    plt.tight_layout()
    plot_filename = f"{filename_prefix}_training_plots.png"
    plt.savefig(plot_filename)
    print(f"Saved plots to {plot_filename}")
    plt.close(fig) # Close the figure after saving


# ===================================
# Training loop
# ===================================
def train(config):
    """ NFSP-DeepCFR Training Loop for N-Player Kuhn """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    num_episodes = config.get('num_episodes', 300_000) # May need more for 3p
    sl_buffer_size = config.get('sl_buffer_size', 2_000_000) # Larger buffer might be needed
    adv_buffer_size = config.get('adv_buffer_size', 2_000_000) # Larger buffer
    batch_size_sl = config.get('batch_size_sl', 512) # Larger batch size
    batch_size_adv = config.get('batch_size_adv', 512) # Larger batch size
    update_pi_every = config.get('update_pi_every', 50) # Episodes update frequency
    update_v_every = config.get('update_v_every', 50) # Episodes update frequency
    learning_rate_pi = config.get('learning_rate_pi', 0.001)
    learning_rate_v = config.get('learning_rate_v', 0.005) # V might need different LR
    eta = config.get('eta', 0.1) # Anticipatory parameter
    exploitability_every = config.get('exploitability_every', 5000) # Episodes
    reward_window_size = config.get('reward_window_size', 1000) # Smoother average
    save_every = config.get('save_every', 25000) # Episodes
    results_dir = config.get('results_dir', 'kuhn_nfsp_deepcfr_3p_results')
    hidden_size = config.get('hidden_size', 128) # Increased default

    os.makedirs(results_dir, exist_ok=True)
    config_filename = os.path.join(results_dir, "config.json")
    with open(config_filename, 'w') as f:
         json.dump(config, f, indent=4)
    print(f"Saved config to {config_filename}")

    filename_prefix = os.path.join(results_dir, f"kuhn_3p_nfsp_dcfr_{int(time.time())}")

    # Initialize components
    environment = Kuhn3PlayerEnvironment()
    all_hands = environment.get_all_card_deals()

    # Shared networks for all players
    pi_network = KuhnNetwork(INFOSER_FEATURE_SIZE, NUM_ACTIONS, hidden_size).to(device) # Π network (Average Policy)
    v_network = KuhnNetwork(INFOSER_FEATURE_SIZE, NUM_ACTIONS, hidden_size).to(device) # V network (Advantage/Regret)

    optimizer_pi = optim.Adam(pi_network.parameters(), lr=learning_rate_pi)
    optimizer_v = optim.Adam(v_network.parameters(), lr=learning_rate_v)

    pi_step_size = 5_000
    v_step_size = 5_000
    scheduler_pi = optim.lr_scheduler.StepLR(optimizer_pi, step_size=pi_step_size, gamma=0.5)
    scheduler_v = optim.lr_scheduler.StepLR(optimizer_v, step_size=v_step_size, gamma=0.95)


    # Use ReservoirBuffer with deque
    sl_memory = ReservoirBuffer(sl_buffer_size) # M_SL: Stores (infoset_str, action) from CFR policy
    adv_memory = ReservoirBuffer(adv_buffer_size) # M_V: Stores (infoset_str, episode, regret_dict)

    # Training results storage
    results = {
        'episodes': [],
        'p0_rewards': [], 'cumulative_p0_rewards': [],
        'average_rewards': [], 'cumulative_average_rewards': [], # Track average reward
        'exploitability': [], 'exploitability_episodes': []
    }
    cumulative_reward_p0 = 0.0
    cumulative_reward_avg = 0.0
    start_time = time.time()

    print("Starting Training...")
    for episode in range(1, num_episodes + 1):
        # Sample initial state (deal cards)
        cards = random.choice(all_hands)
        game_state = environment._get_initial_state(cards)

        # Gameplay Loop
        while not environment.is_terminal(game_state):
            current_player = environment.get_current_player(game_state)
            # Check if player is active (should always be true if turn logic is correct)
            if current_player not in game_state["active_players"]:
                 print(f"Error: Player {current_player} is not active but it's their turn. State: {game_state}")
                 # Handle error, maybe force terminal state or skip turn?
                 # Forcing terminal might be safest to avoid loops.
                 game_state["is_terminal"] = True
                 game_state["terminal_utility"] = environment._calculate_payoffs(game_state) # Calculate payoff based on current bets/folds
                 break


            infoset = environment.get_infoset(game_state, current_player)
            legal_actions = environment.get_legal_actions(game_state)
            if not legal_actions:
                # This *can* happen legitimately if a player is forced all-in and called,
                # but the environment state should become terminal before this check.
                # If it happens here, it implies a potential issue in state transition or terminal check.
                 print(f"Warning: No legal actions in non-terminal state during gameplay loop. State: {game_state}")
                 if not environment.is_terminal(game_state): # Double check if state became terminal unexpectedly
                     game_state["is_terminal"] = True # Force terminal
                     game_state["terminal_utility"] = environment._calculate_payoffs(game_state)
                 break # Exit loop

            # 1. Select Action using Mixed Strategy (η)
            if random.random() < eta: # Use CFR policy (from V network)
                action_policy = get_cfr_policy(environment, v_network, infoset, legal_actions, device)
                action_source = 'cfr'
            else: # Use Average policy (from Pi network)
                action_policy = get_avg_policy(environment, pi_network, infoset, legal_actions, device)
                action_source = 'avg'

            # Sample action from the chosen policy
            action_list = list(action_policy.keys())
            prob_list = list(action_policy.values())

            # Ensure probabilities are valid for sampling
            if not action_list: # No legal actions? Should have been caught earlier.
                 print(f"Error: Policy has no actions. Legal actions: {legal_actions}, Policy: {action_policy}")
                 break # Exit episode

            if abs(sum(prob_list) - 1.0) > 1e-5 or any(p < 0 for p in prob_list):
                 # print(f"Warning: Invalid probability distribution for sampling. Sum={sum(prob_list)}. Policy: {action_policy}. Forcing uniform.")
                 # Fallback to uniform sampling over legal actions
                 chosen_action = random.choice(action_list)
            else:
                 try:
                     chosen_action = random.choices(action_list, weights=prob_list, k=1)[0]
                 except ValueError as e:
                     print(f"Warning: ValueError during action sampling. {e}. Policy: {action_policy}. Falling back to uniform.")
                     # Ensure action_list is not empty before choosing
                     if action_list:
                         chosen_action = random.choice(action_list)
                     else:
                         # This case implies a deeper issue where legal_actions existed but policy mapping failed.
                         print(f"CRITICAL Error: Cannot choose action, empty action_list derived from policy. State: {game_state}")
                         break # Exit episode prematurely

            # Store Experience for SL (Pi network)
            # Store (infoset, action) pair if action came from the CFR policy (V net)
            if action_source == 'cfr':
                sl_memory.push((infoset, chosen_action)) # Store for supervised learning

            # Estimate & Store Regret Samples for V network
            # This is the core Deep CFR step: sample regrets at the current state (I_t)
            # Pass the *current* game state to the estimation function
            instant_regrets = estimate_instantaneous_regrets(
                game_state, pi_network, v_network, environment, all_hands, device
            )

            # If regrets were successfully estimated, store them for V network training
            if instant_regrets is not None:
                # Store (infoset_str, episode_number, regret_dictionary)
                adv_memory.push((infoset, episode, instant_regrets))

            # Execute action and get the next state
            game_state = environment.step(game_state, chosen_action)
        #End Gameplay Loop 


        # Episode finished
        if environment.is_terminal(game_state) and game_state["terminal_utility"] is not None:
            final_utils = game_state["terminal_utility"]
            episode_reward_p0 = final_utils[0]
            episode_reward_avg = sum(final_utils) / NUM_PLAYERS # Should be ~0 for zero-sum
        else:
             # Episode ended unexpectedly (error or edge case)
             print(f"Warning: Episode {episode} ended abnormally. Final state: {game_state}")
             episode_reward_p0 = 0.0
             episode_reward_avg = 0.0


        cumulative_reward_p0 += episode_reward_p0
        cumulative_reward_avg += episode_reward_avg # Track cumulative avg reward

        results['episodes'].append(episode)
        results['p0_rewards'].append(episode_reward_p0)
        results['cumulative_p0_rewards'].append(cumulative_reward_p0)
        results['average_rewards'].append(episode_reward_avg)
        results['cumulative_average_rewards'].append(cumulative_reward_avg)


        # Network Updates
        # Update Pi Network (Supervised Learning on CFR actions)
        if episode % update_pi_every == 0 and len(sl_memory) >= batch_size_sl:
            batch_sl = sl_memory.sample(batch_size_sl)
            if batch_sl: # Ensure sample is not empty
                infoset_strs, target_actions = zip(*batch_sl)

                # Batch conversion to tensors
                try:
                     # Use environment's method for tensor conversion
                     infoset_tensors = torch.cat([environment.get_infoset_tensor(inf_str, device) for inf_str in infoset_strs], dim=0)
                     target_action_indices = torch.tensor(target_actions, dtype=torch.long).to(device) # Target actions are class indices

                     # Forward pass
                     pi_network.train() # Set to train mode
                     action_logits = pi_network(infoset_tensors)

                     # Calculate Cross-Entropy loss
                     loss_pi = F.cross_entropy(action_logits, target_action_indices)

                     # Backward pass and optimization
                     optimizer_pi.zero_grad()
                     loss_pi.backward()
                     torch.nn.utils.clip_grad_norm_(pi_network.parameters(), 1.0) # Gradient clipping
                     optimizer_pi.step()
                     scheduler_pi.step()
                     pi_network.eval() # Set back to eval mode
                except Exception as e:
                     print(f"Error during Pi network update: {e}")
                     # print(f"Batch SL data causing error: {batch_sl[:5]}") # Print first few items
                     pi_network.eval() # Ensure eval mode


        # Update V Network (Regression on Estimated Cumulative Regrets)
        if episode % update_v_every == 0 and len(adv_memory) >= batch_size_adv:
             batch_adv = adv_memory.sample(batch_size_adv)
             if batch_adv:
                 accumulated_regrets_in_batch = defaultdict(lambda: defaultdict(float))
                 infoset_counts_in_batch = defaultdict(int)

                 for infoset_str, _, instant_regret_dict in batch_adv:
                     infoset_counts_in_batch[infoset_str] += 1
                     for action, regret_val in instant_regret_dict.items():
                         if 0 <= action < NUM_ACTIONS: # Check action validity
                             accumulated_regrets_in_batch[infoset_str][action] += regret_val

                 unique_infoset_strs_v = list(accumulated_regrets_in_batch.keys())
                 if not unique_infoset_strs_v: continue # Skip if batch had no processable data

                 try:
                      # Prepare tensors
                      infoset_tensors_v = torch.cat([environment.get_infoset_tensor(inf_str, device) for inf_str in unique_infoset_strs_v], dim=0)

                      # Target: Average instantaneous regret for each (infoset, action) pair in the batch
                      # This acts as a sample of the gradient direction for the cumulative regret.
                      target_avg_inst_regrets_tensor = torch.zeros((len(unique_infoset_strs_v), NUM_ACTIONS), device=device)
                      for i, infoset_str in enumerate(unique_infoset_strs_v):
                           reg_dict = accumulated_regrets_in_batch[infoset_str]
                           count = infoset_counts_in_batch[infoset_str]
                           if count > 0:
                               for action, acc_regret in reg_dict.items():
                                   target_avg_inst_regrets_tensor[i, action] = acc_regret / count

                      # Forward pass - V network predicts *cumulative* regrets
                      v_network.train() # Set to train mode
                      predicted_cumulative_regrets = v_network(infoset_tensors_v)

                      # Loss: MSE between predicted *cumulative* regrets and the *average instantaneous* regrets from batch.
                      # This trains the V network to predict values whose gradient (change over time)
                      # matches the sampled instantaneous regrets.
                      loss_v = F.mse_loss(predicted_cumulative_regrets, target_avg_inst_regrets_tensor)

                      # Backward pass and optimization
                      optimizer_v.zero_grad()
                      loss_v.backward()
                      torch.nn.utils.clip_grad_norm_(v_network.parameters(), 1.0) # Gradient clipping
                      optimizer_v.step()
                      scheduler_v.step()
                      v_network.eval() # Set back to eval mode
                 except Exception as e:
                      print(f"Error during V network update: {e}")
                      # print(f"Batch Adv data causing error: {batch_adv[:5]}")
                      v_network.eval() # Ensure eval mode

        if episode % exploitability_every == 0 or episode == num_episodes:
            exploit_start_time = time.time()
            print(f"Calculating exploitability at episode {episode}...")
            exploit = calculate_exploitability(pi_network, environment, device)
            exploit_end_time = time.time()
            results['exploitability'].append(exploit)
            results['exploitability_episodes'].append(episode)
            elapsed_time = time.time() - start_time
            exploit_calc_time = exploit_end_time - exploit_start_time
            current_lr_pi = optimizer_pi.param_groups[0]['lr']
            current_lr_v = optimizer_v.param_groups[0]['lr']
            print(f"Episode {episode}/{num_episodes} | Avg Exploit: {exploit:.6f} "
                  f"| Exploit Calc Time: {exploit_calc_time:.2f}s "
                  f"| LR Pi: {current_lr_pi:.6f} | LR V: {current_lr_v:.6f} "
                  f"| Total Time: {elapsed_time:.2f}s")


        if episode % save_every == 0 or episode == num_episodes:
             model_save_path_pi = f"{filename_prefix}_pi_network_ep{episode}.pth"
             model_save_path_v = f"{filename_prefix}_v_network_ep{episode}.pth"
             torch.save(pi_network.state_dict(), model_save_path_pi)
             torch.save(v_network.state_dict(), model_save_path_v)
             print(f"Saved networks to {model_save_path_pi} and {model_save_path_v}")

             # Save intermediate results
             results_file = f"{filename_prefix}_results_ep{episode}.json"
             try:
                 with open(results_file, 'w') as f:
                     # Convert numpy arrays if any exist to lists for JSON compatibility
                     serializable_results = {}
                     for key, value in results.items():
                         if isinstance(value, np.ndarray):
                             serializable_results[key] = value.tolist()
                         else:
                             serializable_results[key] = value
                     json.dump(serializable_results, f, indent=4)
                 print(f"Intermediate results saved to {results_file}")
             except Exception as e:
                 print(f"Error saving intermediate results: {e}")

             # Plot intermediate results
             plot_results(results, reward_window_size, f"{filename_prefix}_ep{episode}")


    final_results_file = f"{filename_prefix}_final_results.json"
    try:
         with open(final_results_file, 'w') as f:
            serializable_results = {}
            for key, value in results.items():
                if isinstance(value, np.ndarray):
                    serializable_results[key] = value.tolist()
                else:
                    serializable_results[key] = value
            json.dump(serializable_results, f, indent=4)
         print(f"Final results saved to {final_results_file}")
    except Exception as e:
         print(f"Error saving final results: {e}")

    plot_results(results, reward_window_size, filename_prefix + "_final")

    print("Training finished.")
    return pi_network, v_network # Return trained networks



if __name__ == '__main__':
    train_config_3p = {
        'num_episodes': 200_000,      # More episodes for 3 players
        'sl_buffer_size': 100_000,  # Larger buffers
        'adv_buffer_size': 100_000,
        'batch_size_sl': 128,         # Larger batches
        'batch_size_adv': 128,
        'update_pi_every': 20,        # Update frequency
        'update_v_every': 10,
        'learning_rate_pi': 0.001,    # Starting LR for Pi
        'learning_rate_v': 0.01,     # Starting LR for V (may need tuning)
        'eta': 0.1,                   # Exploration vs Exploitation trade-off
        'exploitability_every': 1_000, # How often to run expensive exploitability calc
        'reward_window_size': 100,   # For smoothing reward plot
        'save_every': 10_000,         # Save checkpoints
        'results_dir': 'kuhn_nfsp_deepcfr_3p_results_A', # Specific folder name
        'hidden_size': 128            # Network hidden layer size
    }

    # Train the agent
    print("Running 3-Player Kuhn Poker Training with NFSP-DeepCFR...")
    final_pi_net, final_v_net = train(train_config_3p)
    print("Training Complete.")
