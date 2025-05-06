"""
2 & 3 Player version of Kuhn Poker RG-NFSP agent.

This script contains inference only code. The training code will be uploaded in a separete directly soon.
"""

import torch
import torch.nn.functional as F
from typing import List
import random


# =============================
# Neural Network for 3P Kuhn Poker
# =============================
class KuhnNetwork(torch.nn.Module):
    def __init__(
            self,
            input_size,
            output_size,
            hidden_size=128
            ):
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
        return self.net(x)


# ============================
# RGNFSP Agent for 2P Kuhn Poker
# ============================
class RGNFSP2PPlayer:
    """
    RG-NFSP agent for 2-player Kuhn Poker.
    """
    def __init__(
            self,
            hidden_size: int = 128,
            device: torch.device = torch.device("cpu")
            ):
        self.num_players = 2
        self.cards = ['J', 'Q', 'K'] # 4 cards for 3 players
        self.num_cards = len(self.cards)
        self.card_map = {card: i for i, card in enumerate(self.cards)}
        self.card_rank = {card: rank for rank, card in enumerate(self.cards)} # J=0, Q=1, K=2
        self.CHECK, self.BET, self.CALL, self.FOLD, self.RAISE = 0, 1, 2, 3, 4
        self.actions = [self.CHECK, self.BET, self.CALL, self.FOLD, self.RAISE]
        self.num_actions = len(self.actions)
        self.action_map_str = {self.CHECK: 'k', self.BET: 'b', self.CALL: 'c', self.FOLD: 'f', self.RAISE: 'r'}
        self.action_map_int = {v: k for k, v in self.action_map_str.items()}
        self.actions_codes_engine = {0: "check", 1: "bet", 2: "call", 3: "fold", 4: "raise"}
        self.actions_codes_to_int_engine = {v: k for k, v in self.actions_codes_engine.items()}
        INFOSER_FEATURE_SIZE_2P = self.num_cards + (self.num_actions * 4) # Card + History
        self.pi_network = KuhnNetwork(
            input_size=INFOSER_FEATURE_SIZE_2P,
            output_size=self.num_actions,
            hidden_size=hidden_size
        )
        self.max_history_len = 4
        self.hidden_size = hidden_size

        PI_NETWORK_PATH = "models/rg_nfsp_2p_pi_net.pth"

        # Load weights
        try:
            self.pi_network.load_state_dict(torch.load(PI_NETWORK_PATH, map_location=device))
            self.pi_network_weights_path = PI_NETWORK_PATH
            print(f"Loaded Pi network weights from {PI_NETWORK_PATH}")
        except FileNotFoundError:
            print(f"Error: Pi network weights file not found at {PI_NETWORK_PATH}")
        self.device = device
        self.pi_network.to(device)
        self.pi_network.eval()
    
    def _get_infoset_tensor(
            self,
            infoset_str: str,
            device: torch.device
            ) -> torch.Tensor:
        """
        Converts infoset string ('CardHistory') to a one-hot tensor.
        Example: 'J0kb' -> [1,0,0,0]  + [1,0,0,0,0] + [0,1,0,0,0] + padding

        Args:
            infoset_str: str, infoset string in the format 'CardHistory'.
            device: torch.device, device to which the tensor should be moved.
        
        Returns:
            infoset_tensor: torch.Tensor, one-hot encoded tensor representing the infoset.
        """
        card = infoset_str[0]
        history = infoset_str[1:]
        card_idx = self.card_map.get(card, -1) # Handle potential errors
        if card_idx == -1: raise ValueError(f"Invalid card in infoset: {infoset_str}")
        card_one_hot = F.one_hot(torch.tensor(card_idx), num_classes=self.num_cards)

        history_indices = [self.action_map_int.get(action_char, -1) for action_char in history]
        if any(idx == -1 for idx in history_indices): raise ValueError(f"Invalid action char in infoset: {infoset_str}")

        if history_indices:
            history_one_hot = F.one_hot(torch.tensor(history_indices), num_classes=self.num_actions)
            flat_history = history_one_hot.view(-1)
        else:
            flat_history = torch.empty(0, dtype=torch.float32)

        padding_needed = (self.num_actions * self.max_history_len) - len(flat_history)
        if padding_needed < 0:
            # Truncate if history somehow exceeds max length assumption
            flat_history = flat_history[:(self.num_actions * self.max_history_len)]
            padding_needed = 0

        history_padding = torch.zeros(padding_needed, dtype=torch.float32)
        final_history_tensor = torch.cat((flat_history.float(), history_padding))

        # Concatenate card, position, and history
        infoset_tensor = torch.cat((card_one_hot.float(), final_history_tensor))

        return infoset_tensor.to(device).unsqueeze(0) # Add batch dimension

    def _get_avg_policy(
        self,
        pi_network: torch.nn.Module,
        infoset_str: str,
        legal_actions: List[int],
        ):
        """
        Gets average policy from Pi net based on infoset string.

        Args:
            pi_network: torch.nn.Module, Pi network.
            infoset_str: str, infoset string in the format 'CardPosHistory'.
            legal_actions: List[int], list of legal actions.
        
        Returns:
            policy: Dict[int, float], dictionary mapping action to probability.
        """
        if not legal_actions: return {}
        infoset_tensor = self._get_infoset_tensor(infoset_str, self.device)
        with torch.no_grad():
            action_logits = pi_network(infoset_tensor).squeeze(0)
            # Apply softmax to get probabilities
            action_probs = F.softmax(action_logits, dim=-1)

        legal_probs = {action: action_probs[action].item() for action in legal_actions}
        prob_sum = sum(v for v in legal_probs.values() if v > 0)
        policy = {}
        if prob_sum > 1e-6:
            for action in legal_actions: policy[action] = max(0.0, legal_probs[action]) / prob_sum
            final_sum = sum(policy.values())
            if abs(final_sum - 1.0) > 1e-5 and final_sum > 1e-6:
                for action in legal_actions: policy[action] /= final_sum
        else:
            num_legal = len(legal_actions); policy = {action: 1.0 / num_legal for action in legal_actions}
        return policy

    def _convert_engine_history_to_my_history(self, history: List[str]):
        """
        Engine history actions are the format 'check', 'bet', 'call', 'fold', 'raise'.
        Convert to my history format 'k', 'b', 'c', 'f', 'r'.
        There are relevant dicts available in the file.
        """
        # First convert engine strs to ints
        my_history = [self.actions_codes_to_int_engine[a] for a in history]
        # then map ints to my str format
        my_history = [self.action_map_str[a] for a in my_history]
        # then convert to a string
        my_history = ''.join(my_history)
        return my_history

    def get_action(self, card, available_actions, round_num, chips, public_state):
        """
        Args:
            card: Your current card (J, Q, K, or A)
            available_actions: Dict of {action_idx: description} for legal actions
            round_num: Current betting round (1 or 2)
            chips: Your remaining chip count
            public_state: Dict containing game state information
                pot_size: Total chips in the pot
                current_bets: List of each player's current bet
                chip_counts: List of each player's chips
                betting_history: List of previous actions
                folded_players: List of boolean values indicating folded status
                highest_bet: Current highest bet amount
                last_bettor: Player ID who last bet/raised (-1 if none)
                current_player: ID of player making the decision
                player_id: Your player ID (same as current_player)
                min_raise: Minimum raise amount

        Returns:
            action_idx: Integer representing the action
                0: check, 1: bet, 2: call, 3: fold, 4: raise
            raise_amount: Integer amount for raise (only used if action_idx is 4)
        """
        # Get policy from the Pi network
        legal_actions = list(available_actions.keys())
        infoset = f"{card}"
        history = self._convert_engine_history_to_my_history(public_state['betting_history'])
        infoset += history
        action_policy = self._get_avg_policy(self.pi_network, infoset, legal_actions)

        # Sample action
        action_list = list(action_policy.keys())
        prob_list = list(action_policy.values())
        prob_sum = sum(prob_list)
        if prob_sum < 1e-6:
            chosen_action = random.choice(action_list)
        else:
            norm_prob_list = [p / prob_sum for p in prob_list]
            chosen_action = random.choices(action_list, weights=norm_prob_list, k=1)[0]
        
        # Handle raise. Only raise the minimum. (As our agent hasn't yet learned to raise. #TODO soon)
        if chosen_action == self.RAISE:
            raise_amount = min(chips, public_state["min_raise"])
            # Ensure raise amount isn't negative
            raise_amount = max(raise_amount, 0)
            return chosen_action, raise_amount

        return chosen_action, None

# ============================
# RGNFSP Agent for 3P Kuhn Poker
# ============================
class RGNFSP3PPlayer:
    """
    RG-NFSP agent for 3-player Kuhn Poker.
    """
    def __init__(
            self,
            hidden_size: int = 256,
            device: torch.device = torch.device("cpu")
            ):
        self.num_players = 3
        self.cards = ['J', 'Q', 'K', 'A'] # 4 cards for 3 players
        self.num_cards = len(self.cards)
        self.card_map = {card: i for i, card in enumerate(self.cards)}
        self.card_rank = {card: rank for rank, card in enumerate(self.cards)} # J=0, Q=1, K=2, A=3
        self.CHECK, self.BET, self.CALL, self.FOLD, self.RAISE = 0, 1, 2, 3, 4
        self.actions = [self.CHECK, self.BET, self.CALL, self.FOLD, self.RAISE]
        self.num_actions = len(self.actions)
        self.action_map_str = {self.CHECK: 'k', self.BET: 'b', self.CALL: 'c', self.FOLD: 'f', self.RAISE: 'r'}
        self.action_map_int = {v: k for k, v in self.action_map_str.items()}
        self.actions_codes_engine = {0: "check", 1: "bet", 2: "call", 3: "fold", 4: "raise"}
        self.actions_codes_to_int_engine = {v: k for k, v in self.actions_codes_engine.items()}

        PI_NETWORK_PATH = "models/rg_nfsp_3p_pi_net.pth"

        # ============================
        # Infoset Tensor Conversion
        # ============================
        MAX_HISTORY_LEN_3P = 6
        INFOSER_FEATURE_SIZE_3P = self.num_cards + self.num_players + (self.num_actions * MAX_HISTORY_LEN_3P) # Card + Pos + History

        self.pi_network = KuhnNetwork(
            input_size=INFOSER_FEATURE_SIZE_3P,
            output_size=self.num_actions,
            hidden_size=hidden_size
        )
        self.hidden_size = hidden_size
        # Load weights
        try:
            self.pi_network.load_state_dict(torch.load(PI_NETWORK_PATH, map_location=device))
            print(f"Loaded Pi network weights from {PI_NETWORK_PATH}")
        except FileNotFoundError:
            print(f"Error: Pi network weights file not found at {PI_NETWORK_PATH}")
        self.pi_network_weights_path = PI_NETWORK_PATH
        self.max_history_len = MAX_HISTORY_LEN_3P
        self.device = device
        self.pi_network.to(device)
        self.pi_network.eval()
    
    def _get_infoset_tensor(
            self,
            infoset_str: str,
            device: torch.device
            ) -> torch.Tensor:
        """
        Converts infoset string ('CardPosHistory') to a one-hot tensor.
        Example: 'J0kb' -> [1,0,0,0] + [1,0,0] + [1,0,0,0,0] + [0,1,0,0,0] + padding

        Args:
            infoset_str: str, infoset string in the format 'CardPosHistory'.
            device: torch.device, device to which the tensor should be moved.
        
        Returns:
            infoset_tensor: torch.Tensor, one-hot encoded tensor representing the infoset.
        """
        card = infoset_str[0]
        position = int(infoset_str[1])
        history = infoset_str[2:]

        card_idx = self.card_map.get(card, -1) # Handle potential errors
        if card_idx == -1: raise ValueError(f"Invalid card in infoset: {infoset_str}")
        card_one_hot = F.one_hot(torch.tensor(card_idx), num_classes=self.num_cards)

        pos_one_hot = F.one_hot(torch.tensor(position), num_classes=self.num_players)

        history_indices = [self.action_map_int.get(action_char, -1) for action_char in history]
        if any(idx == -1 for idx in history_indices): raise ValueError(f"Invalid action char in infoset: {infoset_str}")

        if history_indices:
            history_one_hot = F.one_hot(torch.tensor(history_indices), num_classes=self.num_actions)
            flat_history = history_one_hot.view(-1)
        else:
            flat_history = torch.empty(0, dtype=torch.float32)

        padding_needed = (self.num_actions * self.max_history_len) - len(flat_history)
        if padding_needed < 0:
            # Truncate if history somehow exceeds max length assumption
            flat_history = flat_history[:(self.num_actions * self.max_history_len)]
            padding_needed = 0

        history_padding = torch.zeros(padding_needed, dtype=torch.float32)
        final_history_tensor = torch.cat((flat_history.float(), history_padding))

        # Concatenate card, position, and history
        infoset_tensor = torch.cat((card_one_hot.float(), pos_one_hot.float(), final_history_tensor))

        return infoset_tensor.to(device).unsqueeze(0) # Add batch dimension

    def _get_avg_policy(
        self,
        pi_network: torch.nn.Module,
        infoset_str: str,
        legal_actions: List[int],
        ):
        """
        Gets average policy from Pi net based on infoset string.

        Args:
            pi_network: torch.nn.Module, Pi network.
            infoset_str: str, infoset string in the format 'CardPosHistory'.
            legal_actions: List[int], list of legal actions.
        
        Returns:
            policy: Dict[int, float], dictionary mapping action to probability.
        """
        if not legal_actions: return {}
        infoset_tensor = self._get_infoset_tensor(infoset_str, self.device)
        with torch.no_grad():
            action_logits = pi_network(infoset_tensor).squeeze(0)
            # Apply softmax to get probabilities
            action_probs = F.softmax(action_logits, dim=-1)

        legal_probs = {action: action_probs[action].item() for action in legal_actions}
        prob_sum = sum(v for v in legal_probs.values() if v > 0)
        policy = {}
        if prob_sum > 1e-6:
            for action in legal_actions: policy[action] = max(0.0, legal_probs[action]) / prob_sum
            final_sum = sum(policy.values())
            if abs(final_sum - 1.0) > 1e-5 and final_sum > 1e-6:
                for action in legal_actions: policy[action] /= final_sum
        else:
            num_legal = len(legal_actions); policy = {action: 1.0 / num_legal for action in legal_actions}
        return policy

    def _convert_engine_history_to_my_history(self, history: List[str]):
        """
        Engine history actions are the format 'check', 'bet', 'call', 'fold', 'raise'.
        Convert to my history format 'k', 'b', 'c', 'f', 'r'.
        There are relevant dicts available in the file.
        """
        # First convert engine strs to ints
        my_history = [self.actions_codes_to_int_engine[a] for a in history]
        # then map ints to my str format
        my_history = [self.action_map_str[a] for a in my_history]
        # then convert to a string
        my_history = ''.join(my_history)
        return my_history

    def get_action(self, card, available_actions, round_num, chips, public_state):
        """
        Args:
            card: Your current card (J, Q, K, or A)
            available_actions: Dict of {action_idx: description} for legal actions
            round_num: Current betting round (1 or 2)
            chips: Your remaining chip count
            public_state: Dict containing game state information
                pot_size: Total chips in the pot
                current_bets: List of each player's current bet
                chip_counts: List of each player's chips
                betting_history: List of previous actions
                folded_players: List of boolean values indicating folded status
                highest_bet: Current highest bet amount
                last_bettor: Player ID who last bet/raised (-1 if none)
                current_player: ID of player making the decision
                player_id: Your player ID (same as current_player)
                min_raise: Minimum raise amount

        Returns:
            action_idx: Integer representing the action
                0: check, 1: bet, 2: call, 3: fold, 4: raise
            raise_amount: Integer amount for raise (only used if action_idx is 4)
        """
        # Get policy from the Pi network
        legal_actions = list(available_actions.keys())
        infoset = f"{card}{public_state['current_player']}"
        history = self._convert_engine_history_to_my_history(public_state['betting_history'])
        infoset += history
        action_policy = self._get_avg_policy(self.pi_network, infoset, legal_actions)

        # Sample action
        action_list = list(action_policy.keys())
        prob_list = list(action_policy.values())
        prob_sum = sum(prob_list)
        if prob_sum < 1e-6:
            chosen_action = random.choice(action_list)
        else:
            norm_prob_list = [p / prob_sum for p in prob_list]
            chosen_action = random.choices(action_list, weights=norm_prob_list, k=1)[0]
        
        # Handle raise. Only raise the minimum. (As our agent hasn't yet learned to raise. #TODO soon)
        if chosen_action == self.RAISE:
            raise_amount = min(chips, public_state["min_raise"])
            # Ensure raise amount isn't negative
            raise_amount = max(raise_amount, 0)
            return chosen_action, raise_amount

        return chosen_action, None