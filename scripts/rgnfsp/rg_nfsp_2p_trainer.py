# Necessary imports from previous code + new ones
import torch
import torch.nn.functional as F
import torch.optim as optim
from typing import Dict, List, Tuple, Any, Callable, Optional, Set
import itertools
import random
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import itertools
from typing import Dict, List, Tuple, Optional, FrozenSet
from collections import defaultdict
import pandas as pd
import json


random.seed(1337)
np.random.seed(1999)
torch.manual_seed(42)


# --- Constants ---
CARDS = ('J', 'Q', 'K') # Use tuple for immutability
NUM_CARDS = len(CARDS)
CARD_MAP = {card: i for i, card in enumerate(CARDS)}


# Actions - Using named constants for clarity and potential Enum use later
class Actions:
    PASS, BET, RAISE, FOLD, CALL = 0, 1, 2, 3, 4
    LIST = (PASS, BET, RAISE, FOLD, CALL) # Tuple
    MAP_STR = {PASS: 'p', BET: 'b', RAISE: 'r', FOLD: 'f', CALL: 'c'}
    MAP_INT = {v: k for k, v in MAP_STR.items()}
NUM_ACTIONS = len(Actions.LIST)
INFOSER_FEATURE_SIZE = NUM_CARDS + (NUM_ACTIONS * 4) # 3 + 5*4 = 23


# Define sets of terminal histories for efficient lookup
TERMINAL_HISTORIES: FrozenSet[str] = frozenset([
    "pp",   # Check-Check
    "bc",   # Bet-Call
    "pbc",  # Pass-Bet-Call
    "pbrc", # Pass-Bet-Raise-Call
    "brc",  # Bet-Raise-Call
    "bf",   # Bet-Fold
    "pbf",  # Pass-Bet-Fold
    "pbrf", # Pass-Bet-Raise-Fold
    "brf"   # Bet-Raise-Fold
])

class KuhnPokerEnvironment:
    """
    A robust implementation of the Kuhn Poker environment with a raise action.

    Rules:
    - 3 Cards: J, Q, K
    - 2 Players (Player 0, Player 1)
    - Ante: 1 unit per player
    - Actions: Pass/Check(0), Bet(1), Raise(2), Fold(3), Call(4)
             Bet costs 1 unit. Raise costs an additional 1 unit (total bet 2).
    - Max 1 bet/raise per round.
    - History string tracks public actions ('p', 'b', 'r', 'f', 'c').
    """
    def __init__(self):
        self.card_permutations: List[Tuple[str, str]] = list(itertools.permutations(CARDS, 2))
        # Cache for terminal utilities to avoid re-computation: Key=(history, sorted_cards_tuple)
        self._terminal_utilities_cache: Dict[Tuple[str, Tuple[str, str]], Tuple[float, float]] = {}

    def get_legal_actions(self, history: str) -> List[int]:
        """Returns the list of legal action integers for the current history."""
        if self.is_terminal(history):
            return []

        # Determine actions based on history pattern
        # Uses Actions class constants for readability
        if history == "": # Player 0 turn, Round 1
            return [Actions.PASS, Actions.BET]
        elif history == "p": # Player 1 turn, Round 1 (P0 passed)
            return [Actions.PASS, Actions.BET]
        elif history == "b": # Player 1 turn, Round 1 (P0 bet)
            return [Actions.FOLD, Actions.CALL, Actions.RAISE]
        elif history == "pb": # Player 0 turn, Round 2 (P0 passed, P1 bet)
            # Allows check-raise. If check-raise is not allowed, remove Actions.RAISE
            return [Actions.FOLD, Actions.CALL, Actions.RAISE]
        elif history == "pbr": # Player 1 turn, Round 2 (P0 passed, P1 bet, P0 raised)
            return [Actions.FOLD, Actions.CALL]
        elif history == "br": # Player 0 turn, Round 2 (P0 bet, P1 raised)
            return [Actions.FOLD, Actions.CALL]
        else:
            # Should not be reached if is_terminal is correct
            raise ValueError(f"History '{history}' is not terminal but has no defined legal actions.")

    def step(self, history: str, action: int) -> str:
        """
        Applies an action to the history string.
        Raises ValueError if the action is illegal.
        """
        if action not in self.get_legal_actions(history):
            legal_strs = [Actions.MAP_STR.get(a, '?') for a in self.get_legal_actions(history)]
            action_str = Actions.MAP_STR.get(action, '?')
            raise ValueError(f"Illegal action '{action_str}' ({action}) for history '{history}'. Legal: {legal_strs}")
        return history + Actions.MAP_STR[action]

    def get_player_turn(self, history: str) -> int:
        """Returns the player (0 or 1) whose turn it is. Returns -1 if terminal."""
        if self.is_terminal(history):
            return -1

        # Player 0 acts first (len=0), and after P1 completes action after P0 pass ('pb', len=2),
        # or after P1 raises P0's bet ('br', len=2)
        if history == "" or history == "pb" or history == "br":
            return 0
        # Player 1 acts after P0 acts ('p' or 'b', len=1),
        # or after P0 raises P1's bet ('pbr', len=3)
        elif history == "p" or history == "b" or history == "pbr":
            return 1
        else:
             # Should not happen for valid non-terminal histories
             raise ValueError(f"Could not determine player turn for non-terminal history '{history}'")

    def is_terminal(self, history: str) -> bool:
        """Checks if the history represents a terminal state using the predefined set."""
        return history in TERMINAL_HISTORIES

    def _calculate_payoffs(self, history: str, cards: Tuple[str, str]) -> Tuple[float, float]:
        """
        Internal helper to calculate payoffs based on terminal history and cards.
        Assumes history is terminal.
        Returns (payoff_p0, payoff_p1).
        """
        p0_card, p1_card = cards
        # Determine winner: Player 0 wins if their card rank is higher
        winner = 0 if CARD_MAP[p0_card] > CARD_MAP[p1_card] else 1
        p0_payoff = 0.0

        # --- Payoff Logic based on Terminal History ---
        # Pot starts at 2 (Ante 1 each)

        if history == "pp": # Check-Check -> Showdown
            # Pot = 2. Winner gets loser's ante = 1 unit profit.
            p0_payoff = 1.0 if winner == 0 else -1.0
        elif history == "bf": # Bet-Fold
            # P1 folds to P0's bet. P0 wins P1's ante. Profit = 1.
            p0_payoff = 1.0
        elif history == "pbf": # Pass-Bet-Fold
            # P0 folds to P1's bet. P1 wins P0's ante. P0 Profit = -1.
            p0_payoff = -1.0
        elif history == "bc" or history == "pbc": # Bet-Call or Pass-Bet-Call -> Showdown
            # Bet = 1. Pot = 2 (antes) + 1 (bet) + 1 (call) = 4. Winner gets 2 profit.
            p0_payoff = 2.0 if winner == 0 else -2.0
        elif history == "brf": # Bet-Raise-Fold
            # P0 bet 1, P1 raised to 2, P0 folded. P0 loses ante(1) + bet(1) = 2. P0 Profit = -2.
            p0_payoff = -2.0
        elif history == "pbrf": # Pass-Bet-Raise-Fold
            # P0 passed, P1 bet 1, P0 raised to 2, P1 folded. P1 loses ante(1) + bet(1) = 2. P0 wins 2. P0 Profit = 2.
            p0_payoff = 2.0
        elif history == "brc" or history == "pbrc": # Bet-Raise-Call or Pass-Bet-Raise-Call -> Showdown
            # Bet = 2. Pot = 2 (antes) + 2 (P0 total bet) + 2 (P1 total bet) = 6. Winner gets 3 profit.
            p0_payoff = 3.0 if winner == 0 else -3.0
        # No else needed because is_terminal should be checked before calling

        # Return payoffs for Player 0 and Player 1 (zero-sum game)
        return (p0_payoff, -p0_payoff)

    def get_utility(self, history: str, cards: Tuple[str, str], player: int) -> float:
        """
        Returns the utility for the specified player at a terminal state.
        Uses caching for efficiency.
        Raises ValueError if the history is not terminal or player index is invalid.
        """
        if not self.is_terminal(history):
            raise ValueError(f"get_utility called on non-terminal history '{history}'")
        if player not in [0, 1]:
            raise ValueError(f"Invalid player index {player} requested for utility.")

        # Use sorted cards tuple for cache key consistency, regardless of input order
        # This assumes the winner logic in _calculate_payoffs correctly uses the original 'cards' tuple
        cards_tuple_sorted = tuple(sorted(cards))
        cache_key = (history, cards_tuple_sorted)

        if cache_key not in self._terminal_utilities_cache:
            # Calculate payoffs using the original card order (needed for winner determination)
            payoffs = self._calculate_payoffs(history, cards)
            # Store in cache using the canonical sorted key
            self._terminal_utilities_cache[cache_key] = payoffs

        # Return the utility for the requested player from the cached result
        return self._terminal_utilities_cache[cache_key][player]

    def get_infoset(self, history: str, cards: Tuple[str, str], player: int) -> str:
        """
        Returns the information set string for the player.
        Format: PlayerCard + HistoryString
        """
        if player not in [0, 1]:
            raise ValueError(f"Invalid player index {player} requested for infoset.")
        # cards is expected to be (p0_card, p1_card)
        return cards[player] + history

    def get_all_card_deals(self) -> List[Tuple[str, str]]:
        """Returns all possible card dealings (permutations)."""
        return self.card_permutations

    def get_infoset_tensor(self, infoset_str: str, device: torch.device) -> torch.Tensor:
        MAX_HISTORY_LEN = 4
        card = infoset_str[0]
        history = infoset_str[1:]
        card_idx = CARD_MAP[card]
        card_one_hot = F.one_hot(torch.tensor(card_idx), num_classes=NUM_CARDS)
        history_indices = [Actions.MAP_INT.get(action_char) for action_char in history]
        if history_indices:
            history_one_hot = F.one_hot(torch.tensor(history_indices), num_classes=NUM_ACTIONS)
            flat_history = history_one_hot.view(-1)
        else:
            flat_history = torch.empty(0, dtype=torch.float32)

        padding_needed = (NUM_ACTIONS * MAX_HISTORY_LEN) - len(flat_history)
        history_padding = torch.zeros(padding_needed, dtype=torch.float32)
        final_history_tensor = torch.cat((flat_history.float(), history_padding))
        infoset_tensor = torch.cat((card_one_hot.float(), final_history_tensor))
        return infoset_tensor.to(device).unsqueeze(0)

# ==================================
# CFR Policy
# ===================================
def get_cfr_policy(
        env: KuhnPokerEnvironment,
        v_network: torch.nn.Module,
        infoset_str: str,
        legal_actions: List[int],
        device: torch.device
        ) -> Dict[int, float]:
    """
    Get the CFR policy for a given infoset using the V network.
    This function computes the policy based on the advantage values
    predicted by the V network for the legal actions in the infoset.
    The policy is normalized to sum to 1.0.

    Args:
        env: KuhnPokerEnvironment, The Kuhn Poker environment.
        v_network: torch.nn.Module, The V network for advantage estimation.
        infoset_str: str, The infoset string for the current state.
        legal_actions: List[int], List of legal actions for the current state.
        device: torch.device, The device to perform computations on.
    
    Returns:
        policy: Dict[int, float], The computed CFR policy for the given infoset.
    """
    infoset_tensor = env.get_infoset_tensor(infoset_str, device)
    with torch.no_grad():
        advantage_outputs = v_network(infoset_tensor).squeeze(0)
    regrets = {action: advantage_outputs[action].item() for action in legal_actions}
    positive_regrets = {action: max(regret, 0.0) for action, regret in regrets.items()}
    sum_positive_regrets = sum(positive_regrets.values())
    policy = {}
    if sum_positive_regrets > 0:
        for action in legal_actions: policy[action] = positive_regrets[action] / sum_positive_regrets
    else:
        num_legal = len(legal_actions); policy = {action: 1.0 / num_legal if num_legal > 0 else 0.0 for action in legal_actions}
    return policy

# ==================================
# Average Policy
# ===================================
def get_avg_policy(
        env: KuhnPokerEnvironment,
        pi_network: torch.nn.Module,
        infoset_str: str, legal_actions: List[int],
        device: torch.device
        )-> Dict[int, float]:
    """
    Get the average policy for a given infoset using the Pi network.
    This function computes the policy based on the action probabilities
    predicted by the Pi network for the legal actions in the infoset.
    The policy is normalized to sum to 1.0.
    
    Args:
        env: KuhnPokerEnvironment, The Kuhn Poker environment.
        pi_network: torch.nn.Module, The Pi network for action probabilities.
        infoset_str: str, The infoset string for the current state.
        legal_actions: List[int], List of legal actions for the current state.
        device: torch.device, The device to perform computations on.
    
    Returns:
        policy: Dict[int, float], The computed average policy for the given infoset.
    """
    infoset_tensor = env.get_infoset_tensor(infoset_str, device)
    with torch.no_grad():
        action_probs = pi_network(infoset_tensor).squeeze(0)
    legal_probs = {action: action_probs[action].item() for action in legal_actions}
    prob_sum = sum(v for v in legal_probs.values() if v > 0)
    policy = {}
    if prob_sum > 1e-6:
        for action in legal_actions: policy[action] = max(0.0, legal_probs[action]) / prob_sum
        final_sum = sum(policy.values())
        if abs(final_sum - 1.0) > 1e-5 and final_sum > 1e-6:
            for action in legal_actions: policy[action] /= final_sum
    else:
        num_legal = len(legal_actions); policy = {action: 1.0 / num_legal if num_legal > 0 else 0.0 for action in legal_actions}
    return policy

# =======================================
# Regret Estimation & Expected Utility
# =======================================
memo_reach_prob = {}
memo_utility = {} # Now used for exploitability calc

def get_opponent_reach_probability(
        target_history: str,
        target_cards: Tuple[str, str],
        player_p: int, pi_network: torch.nn.Module,
        environment: KuhnPokerEnvironment,
        device: torch.device
        ) -> float:
    """
    Calculate the reach probability of the opponent given the target history and cards.
    This function uses memoization to avoid redundant calculations.
    
    Args:
        target_history: str, The target history string.
        target_cards: Tuple[str, str], The target cards for the players.
        player_p: int, The player index (0 or 1).
        pi_network: torch.nn.Module, The Pi network for action probabilities.
        environment: KuhnPokerEnvironment, The Kuhn Poker environment.
        device: torch.device, The device to perform computations on.
    
    Returns:
        reach_prob: float, The calculated reach probability.
    """
    cards_tuple = tuple(sorted(target_cards))
    memo_key = (target_history, cards_tuple, player_p)
    if memo_key in memo_reach_prob: return memo_reach_prob[memo_key]
    reach_prob = 1.0 / len(environment.card_permutations)
    current_history = ""
    for action_char in target_history:
        action = Actions.MAP_INT.get(action_char)
        node_player = environment.get_player_turn(current_history)
        if node_player != player_p:
            opponent_player = 1 - node_player
            opponent_infoset = environment.get_infoset(current_history, target_cards, opponent_player)
            legal_actions = environment.get_legal_actions(current_history)
            opponent_policy = get_avg_policy(environment, pi_network, opponent_infoset, legal_actions, device)
            action_taken_prob = opponent_policy.get(action, 0.0)
            reach_prob *= action_taken_prob
            if reach_prob <= 1e-9: reach_prob = 0.0; break
        current_history += action_char
    memo_reach_prob[memo_key] = reach_prob
    return reach_prob

# Note: get_expected_utility from previous code is needed here for regret calc
# It's slightly modified to fit the regret calc context below more cleanly
def get_expected_utility_regret_calc(
    node: Tuple[str, Tuple[str, str]], # (history, cards)
    player_p: int,
    v_network: torch.nn.Module, # Player p uses V network (CFR policy)
    pi_network: torch.nn.Module, # Opponent uses Pi network (Avg policy)
    environment: KuhnPokerEnvironment,
    device: torch.device,
    memo: Dict[Tuple[str, Tuple[str, str]], float]
) -> float:
    history, cards = node
    node_tuple = (history, tuple(sorted(cards)))
    if node_tuple in memo: return memo[node_tuple]
    if environment.is_terminal(history):
        utility = environment.get_utility(history, cards, player_p); memo[node_tuple] = utility; return utility

    current_player = environment.get_player_turn(history)
    legal_actions = environment.get_legal_actions(history)
    current_infoset = environment.get_infoset(history, cards, current_player)
    expected_value = 0.0

    if current_player == player_p:
        strategy_dist = get_cfr_policy(environment, v_network, current_infoset, legal_actions, device)
    else:
        strategy_dist = get_avg_policy(environment, pi_network, current_infoset, legal_actions, device)

    for action in legal_actions:
        prob = strategy_dist.get(action, 0.0)
        if prob > 0:
            next_history = environment.step(history, action)
            next_node = (next_history, cards)
            expected_value += prob * get_expected_utility_regret_calc(
                next_node, player_p, v_network, pi_network, environment, device, memo
            )
    memo[node_tuple] = expected_value
    return expected_value

def estimate_instantaneous_regrets(I_t_str: str, pi_network: torch.nn.Module, v_network: torch.nn.Module, environment: KuhnPokerEnvironment, all_possible_hands: List[Tuple[str, str]], device: torch.device) -> Optional[Dict[int, float]]:
    player_p_card = I_t_str[0]
    history_at_I_t = I_t_str[1:]
    player_p = environment.get_player_turn(history_at_I_t)
    if player_p == -1: return None
    legal_actions_at_I_t = environment.get_legal_actions(history_at_I_t)
    if not legal_actions_at_I_t: return None

    total_opp_reach_prob_sum = 0.0
    cumulative_regrets: Dict[int, float] = {action: 0.0 for action in legal_actions_at_I_t}
    valid_hand_count = 0
    memo_reach_prob.clear() # Clear global reach prob memo

    for hand in all_possible_hands:
        if hand[player_p] != player_p_card: continue
        cards_for_hand = hand
        opp_reach_prob = get_opponent_reach_probability(history_at_I_t, cards_for_hand, player_p, pi_network, environment, device)
        if opp_reach_prob <= 1e-9: continue
        valid_hand_count += 1
        total_opp_reach_prob_sum += opp_reach_prob

        cf_values_per_action_hand: Dict[int, float] = {}
        memo_utility_for_hand = {} # Fresh memo for this hand

        for action_prime in legal_actions_at_I_t:
            next_history = environment.step(history_at_I_t, action_prime)
            next_node = (next_history, cards_for_hand)
            cf_values_per_action_hand[action_prime] = get_expected_utility_regret_calc(
                next_node, player_p, v_network, pi_network, environment, device, memo_utility_for_hand
            )

        current_cfr_policy_at_I_t = get_cfr_policy(environment, v_network, I_t_str, legal_actions_at_I_t, device)
        v_I_t_hand = 0.0
        for action_double_prime in legal_actions_at_I_t:
            prob = current_cfr_policy_at_I_t.get(action_double_prime, 0.0)
            v_I_t_hand += prob * cf_values_per_action_hand[action_double_prime]

        for action_prime in legal_actions_at_I_t:
            regret_contribution = opp_reach_prob * (cf_values_per_action_hand[action_prime] - v_I_t_hand)
            cumulative_regrets[action_prime] += regret_contribution

    if valid_hand_count == 0 or total_opp_reach_prob_sum <= 1e-9: return None
    final_regrets = {action: cum_regret / total_opp_reach_prob_sum for action, cum_regret in cumulative_regrets.items()}
    return final_regrets


# ================================
# Replay Buffers
# ================================

class ReservoirBuffer:
    """Reservoir buffer for SL Memory (MSL) or Advantage Memory (MV)."""
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, item):
        if len(self.memory) < self.capacity:
            self.memory.append(item)
        else:
            # Probability of replacing is capacity / (position + 1)
            idx = random.randint(0, self.position)
            if idx < self.capacity:
                self.memory[idx] = item
        self.position += 1

    def sample(self, batch_size):
        return random.sample(self.memory, min(batch_size, len(self.memory)))

    def __len__(self):
        return len(self.memory)


# ===============================
# Neural Network
# ===============================
class KuhnNetwork(torch.nn.Module):
    """ Simple MLP for Kuhn Poker """
    def __init__(self, input_size, output_size, hidden_size=64):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, output_size)
        )
        self.output_size = output_size

    def forward(self, x):
        out = self.net(x)
        # Apply softmax only if used as Pi network (average policy)
        # V network (advantage) returns raw scores
        # Returning raw scores here; softmax applied in get_avg_policy
        return out


# ================================
# Exploitability Calculation
# ================================
memo_br = {}

def get_best_response_value_recursive(
    node: Tuple[str, Tuple[str, str]], # (history, cards)
    br_player: int, # The player playing the best response
    fixed_policy_net: torch.nn.Module, # The network of the player playing the fixed policy (Π)
    environment: KuhnPokerEnvironment,
    device: torch.device
) -> float:
    """ Calculates the value for br_player assuming they play optimally against fixed_policy_net """
    history, cards = node
    node_tuple = (history, tuple(cards), br_player) # Include br_player in memo key

    if node_tuple in memo_br:
        return memo_br[node_tuple]

    if environment.is_terminal(history):
        utility = environment.get_utility(history, cards, br_player)
        memo_br[node_tuple] = utility
        return utility

    current_player = environment.get_player_turn(history)
    legal_actions = environment.get_legal_actions(history)

    if current_player == br_player:
        # Best response player maximizes their value
        best_value = -float('inf')
        if not legal_actions: return 0.0 # Should not happen in non-terminal Kuhn

        for action in legal_actions:
            next_history = environment.step(history, action)
            next_node = (next_history, cards)
            value = get_best_response_value_recursive(
                next_node, br_player, fixed_policy_net, environment, device
            )
            best_value = max(best_value, value)
        memo_br[node_tuple] = best_value
        return best_value

    else: # Fixed policy player's turn
        fixed_player = 1 - br_player
        current_infoset = environment.get_infoset(history, cards, fixed_player)
        strategy_dist = get_avg_policy(environment, fixed_policy_net, current_infoset, legal_actions, device)
        expected_value = 0.0
        if not legal_actions: return 0.0 # Should not happen

        for action in legal_actions:
            prob = strategy_dist.get(action, 0.0)
            if prob > 0:
                next_history = environment.step(history, action)
                next_node = (next_history, cards)
                expected_value += prob * get_best_response_value_recursive(
                     next_node, br_player, fixed_policy_net, environment, device
                 )
        memo_br[node_tuple] = expected_value
        return expected_value


def calculate_exploitability(
    pi_network: torch.nn.Module,
    environment: KuhnPokerEnvironment,
    device: torch.device
) -> float:
    """ Calculates the exploitability of the pi_network. """
    total_br_value = 0.0
    all_hands = environment.card_permutations

    # Value for Player 0 playing BR against Player 1 playing Pi
    memo_br.clear() # Clear memoization before BR calculation
    br_value_p0 = 0.0
    for cards in all_hands:
        initial_node = ("", cards) # Start after deal
        br_value_p0 += get_best_response_value_recursive(initial_node, 0, pi_network, environment, device)

    # Value for Player 1 playing BR against Player 0 playing Pi
    memo_br.clear() # Clear memoization
    br_value_p1 = 0.0
    for cards in all_hands:
         initial_node = ("", cards)
         # Need value for P1, but our function returns value for br_player.
         # Since it's zero-sum, v(pi, BR(pi)) for P1 = -v(BR(pi), pi) for P0
         # Let's calculate v(BR(pi), pi) for P1 directly
         br_value_p1 += get_best_response_value_recursive(initial_node, 1, pi_network, environment, device)

    # Average value achieved by the best responder over all deals
    # Exploitability = (Value P0 gets as BR + Value P1 gets as BR) / 2 / num_deals
    # Because value is from BR player's perspective
    exploitability = (br_value_p0 + br_value_p1) / 2.0 / len(all_hands)
    return exploitability


# ================================
# Plotting Function
# ================================
def plot_results(results, window_size, filename_prefix):
    """ Plots rewards and exploitability. """
    if not results['episodes']:
        print("No results to plot.")
        return

    episodes = results['episodes']
    raw_rewards = results['raw_rewards']
    cum_rewards = results['cumulative_rewards']
    exploitability = results['exploitability']
    exploit_episodes = results['exploitability_episodes']

    plt.style.use('seaborn-v0_8-darkgrid') # Nicer style
    fig, axes = plt.subplots(3, 1, figsize=(10, 15), sharex=True) # 3 rows, 1 column

    # 1. Raw Rewards (with rolling average)
    axes[0].plot(episodes, raw_rewards, label='Raw Reward per Episode', alpha=0.3)
    if len(raw_rewards) >= window_size:
        rolling_avg = np.convolve(raw_rewards, np.ones(window_size)/window_size, mode='valid')
        axes[0].plot(episodes[window_size-1:], rolling_avg, label=f'Smoothed Reward ({window_size}-ep avg)', color='red')
    axes[0].set_ylabel("Reward")
    axes[0].set_title("Episode Rewards")
    axes[0].legend()
    axes[0].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False) # Hide x-axis labels except last

    # 2. Cumulative Rewards
    axes[1].plot(episodes, cum_rewards, label='Cumulative Reward', color='green')
    axes[1].set_ylabel("Total Reward")
    axes[1].set_title("Cumulative Reward Over Training")
    axes[1].legend()
    axes[1].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False) # Hide x-axis labels except last


    # 3. Exploitability
    if exploit_episodes:
        axes[2].plot(exploit_episodes, exploitability, label='Exploitability (NashConv/2)', marker='o', linestyle='-', color='purple')
        axes[2].set_xlabel("Episode")
        axes[2].set_ylabel("Exploitability (mbb/hand)")
        axes[2].set_title("Policy Exploitability Over Training")
        axes[2].legend()
        axes[2].set_yscale('log') # Often helpful for exploitability
        axes[2].grid(True, which="both", ls="--")
    else:
        axes[2].text(0.5, 0.5, 'No exploitability data recorded', horizontalalignment='center', verticalalignment='center', transform=axes[2].transAxes)
        axes[2].set_xlabel("Episode")


    plt.tight_layout()
    plot_filename = f"{filename_prefix}_training_plots.png"
    plt.savefig(plot_filename)
    print(f"Saved plots to {plot_filename}")
    # plt.show() # Optionally display plots immediately


# ================================
# Training Function
# ================================
def train(config):
    """ NFSP-DeepCFR Training Loop """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    num_episodes = config.get('num_episodes', 200_000)
    sl_buffer_size = config.get('sl_buffer_size', 100_000)
    adv_buffer_size = config.get('adv_buffer_size', 100_000)
    batch_size_sl = config.get('batch_size_sl', 128)
    batch_size_adv = config.get('batch_size_adv', 128)
    update_pi_every = config.get('update_pi_every', 20) # Episodes
    update_v_every = config.get('update_v_every', 20) # Episodes
    learning_rate_pi = config.get('learning_rate_pi', 0.001)
    learning_rate_v = config.get('learning_rate_v', 0.001)
    eta = config.get('eta', 0.08) # Anticipatory parameter
    exploitability_every = config.get('exploitability_every', 1000) # Episodes
    reward_window_size = config.get('reward_window_size', 100)
    save_every = config.get('save_every', 5000) # Episodes
    results_dir = config.get('results_dir', 'kuhn_nfsp_deepcfr_results')
    hidden_size = config.get('hidden_size', 64)

    os.makedirs(results_dir, exist_ok=True)
    filename_prefix = os.path.join(results_dir, "kuhn_nfsp_deepcfr")

    # Initialize components
    environment = KuhnPokerEnvironment()
    all_hands = environment.card_permutations

    pi_network = KuhnNetwork(INFOSER_FEATURE_SIZE, NUM_ACTIONS, hidden_size).to(device) # Π network (Average Policy)
    v_network = KuhnNetwork(INFOSER_FEATURE_SIZE, NUM_ACTIONS, hidden_size).to(device) # V network (Advantage/Regret)

    optimizer_pi = optim.Adam(pi_network.parameters(), lr=learning_rate_pi)
    optimizer_v = optim.Adam(v_network.parameters(), lr=learning_rate_v)

    # Wrap optimizers with learning rate scheduler's according to total number of episode and update frequency
    pi_step_size = 10_000
    v_step_size = 10_000
    scheduler_pi = optim.lr_scheduler.StepLR(optimizer_pi, step_size=pi_step_size, gamma=0.5)
    scheduler_v = optim.lr_scheduler.StepLR(optimizer_v, step_size=v_step_size, gamma=0.95)


    sl_memory = ReservoirBuffer(sl_buffer_size) # M_SL: Stores (infoset_str, action) from CFR policy
    adv_memory = ReservoirBuffer(adv_buffer_size) # M_V: Stores (infoset_str, iteration, regret_dict)

    # Training results storage
    results = {
        'episodes': [], 'raw_rewards': [], 'cumulative_rewards': [],
        'exploitability': [], 'exploitability_episodes': []
    }
    cumulative_reward_total = 0.0
    start_time = time.time()

    print("Starting Training...")
    for episode in range(1, num_episodes + 1):
        # Sample initial state (deal cards)
        cards = random.choice(all_hands)
        history = ""
        episode_reward_p0 = 0.0 # Track reward for plotting (e.g., from P0's perspective)

        # --- Gameplay Loop ---
        while not environment.is_terminal(history):
            current_player = environment.get_player_turn(history)
            infoset = environment.get_infoset(history, cards, current_player)
            legal_actions = environment.get_legal_actions(history)
            if not legal_actions: break # Should not happen if is_terminal is correct

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
            try:
                 chosen_action = random.choices(action_list, weights=prob_list, k=1)[0]
            except ValueError as e:
                 print(f"Warning: ValueError during action sampling. {e}. Policy: {action_policy}. Falling back to uniform.")
                 chosen_action = random.choice(action_list)


            # Store (infoset, action) pair if action came from CFR policy
            if action_source == 'cfr':
                sl_memory.push((infoset, chosen_action))

            # Estimate & Store Regret Samples for V 
            instant_regrets = estimate_instantaneous_regrets(
                infoset, pi_network, v_network, environment, all_hands, device
            )
            if instant_regrets is not None:
                # Store (infoset, iteration_counter, regret_dict)
                adv_memory.push((infoset, episode, instant_regrets))

            # Execute action
            history = environment.step(history, chosen_action)

        # Episode finished, record reward (e.g., P0's utility)
        final_utility_p0 = environment.get_utility(history, cards, 0)
        episode_reward_p0 = final_utility_p0
        cumulative_reward_total += episode_reward_p0

        results['episodes'].append(episode)
        results['raw_rewards'].append(episode_reward_p0)
        results['cumulative_rewards'].append(cumulative_reward_total)


        # Update Pi Network (Supervised Learning)
        if episode % update_pi_every == 0 and len(sl_memory) >= batch_size_sl:
            batch_sl = sl_memory.sample(batch_size_sl)
            infoset_strs, target_actions = zip(*batch_sl)

            # Batch conversion to tensors
            infoset_tensors = torch.cat([environment.get_infoset_tensor(inf_str, device) for inf_str in infoset_strs], dim=0)
            target_action_indices = torch.tensor(target_actions, dtype=torch.long).to(device) # Target actions are class indices

            # Forward pass
            pi_network.train()
            action_logits = pi_network(infoset_tensors)

            # Calculate Cross-Entropy loss
            loss_pi = F.cross_entropy(action_logits, target_action_indices)

            # Backward pass and optimization
            optimizer_pi.zero_grad()
            loss_pi.backward()
            optimizer_pi.step()
            scheduler_pi.step()
            pi_network.eval()


        # Update V Network (Regression on Cumulative Regrets)
        if episode % update_v_every == 0 and len(adv_memory) >= batch_size_adv:
             batch_adv = adv_memory.sample(batch_size_adv)
             # Accumulate instantaneous regrets for unique infosets within this batch
             # This approximates the true cumulative regret R_T(I, a) = sum_{t=1..T} r_t(I, a)
             accumulated_regrets_in_batch = defaultdict(lambda: defaultdict(float)) # {infoset_str: {action: total_instant_regret}}

             for infoset_str, _, instant_regret_dict in batch_adv:
                 for action, regret_val in instant_regret_dict.items():
                     if 0 <= action < NUM_ACTIONS: # Ensure action index is valid
                         accumulated_regrets_in_batch[infoset_str][action] += regret_val

             # Prepare tensors based on the unique infosets found in the batch
             unique_infoset_strs_v = list(accumulated_regrets_in_batch.keys())

             if not unique_infoset_strs_v:
                 continue # Skip update if batch had no processable regret data

             infoset_tensors_v = torch.cat([environment.get_infoset_tensor(inf_str, device) for inf_str in unique_infoset_strs_v], dim=0)

             # Create target tensor using the batch-accumulated regrets
             # Target shape: [num_unique_infosets, NUM_ACTIONS]
             target_accumulated_regrets_tensor = torch.zeros((len(unique_infoset_strs_v), NUM_ACTIONS), device=device)
             for i, infoset_str in enumerate(unique_infoset_strs_v):
                  reg_dict = accumulated_regrets_in_batch[infoset_str]
                  for action, acc_regret in reg_dict.items():
                       target_accumulated_regrets_tensor[i, action] = acc_regret

             # Forward pass
             v_network.train()
             predicted_regrets = v_network(infoset_tensors_v)

             # Calculate MSE loss
             loss_v = F.mse_loss(predicted_regrets, target_accumulated_regrets_tensor)

             # Backward pass and optimization
             optimizer_v.zero_grad()
             loss_v.backward()
             optimizer_v.step()
             scheduler_v.step()
             v_network.eval()

        # Calculate and Record Exploitability
        if episode % exploitability_every == 0:
            pi_network.eval() # Ensure Pi net is in eval mode
            exploit = calculate_exploitability(pi_network, environment, device)
            results['exploitability'].append(exploit)
            results['exploitability_episodes'].append(episode)
            elapsed_time = time.time() - start_time
            print(f"Episode {episode}/{num_episodes} | Exploitability: {exploit:.6f} | Time: {elapsed_time:.2f}s")

        if episode % save_every == 0 or episode == num_episodes:
             torch.save(pi_network.state_dict(), f"{filename_prefix}_pi_network_ep{episode}.pth")
             torch.save(v_network.state_dict(), f"{filename_prefix}_v_network_ep{episode}.pth")
             print(f"Saved networks at episode {episode}")

    plot_results(results, reward_window_size, filename_prefix)
    # Arrays are different length due to exploitability calculation frequency
    # so save everything to a JSON file
    results_file = os.path.join(results_dir, f"kuhn_nfsp_deepcfr_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {results_file}")

    print("Training finished.")
    return pi_network, v_network

if __name__ == '__main__':
    train_config = {
        'num_episodes': 200_000,  
        'sl_buffer_size': 100_000,
        'adv_buffer_size': 100_000,
        'batch_size_sl': 128,
        'batch_size_adv': 128,
        'update_pi_every': 20,
        'update_v_every': 10,
        'learning_rate_pi': 0.005, 
        'learning_rate_v': 0.1,
        'eta': 0.1,
        'exploitability_every': 1_000, 
        'reward_window_size': 100,
        'save_every': 10_000,
        'results_dir': 'kuhn_rg_nfsp_2p', 
        'hidden_size': 128
    }

    # Train the agent
    print("Running Training...")
    final_pi_net, final_v_net = train(train_config)
    print("Training Complete.")

    # Find the last saved Pi network to simulate
    results_dir = train_config['results_dir']
    last_episode = train_config['num_episodes']
    final_pi_path = os.path.join(results_dir, f"kuhn_nfsp_deepcfr_pi_network_ep{last_episode}.pth")

