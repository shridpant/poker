import random
import torch
import numpy as np
from players.base import Player
from models.frl_actor_critic import FRLActorCritic

class FRLAgent(Player):
    """
    Federated Reinforcement Learning Agent for Kuhn Poker
    
    This agent uses an actor-critic model and can participate in federated learning
    by sharing and receiving model updates.
    """
    def __init__(self, player_id=None, state_dim=20, action_dim=5, hidden_dim=128, 
                 exploration_rate=0.1, learning_rate=0.001, variant="kuhn_3p",
                 raise_exploration_rate=0.5):
        """
        Initialize the FRL Agent
        
        Args:
            player_id: ID of the player (used for tracking)
            state_dim: Dimension of the state representation
            action_dim: Number of possible actions
            hidden_dim: Hidden layer dimension
            exploration_rate: Epsilon for epsilon-greedy exploration
            learning_rate: Learning rate for optimizer
            variant: Game variant (for reward scaling)
            raise_exploration_rate: Epsilon for raise exploration strategy
        """
        self.player_id = player_id
        self.model = FRLActorCritic(state_dim, action_dim, learning_rate, hidden_dim)
        self.epsilon = exploration_rate
        self.local_transitions = []
        self.current_card = None
        self.variant = variant
        self.betting_history = ""
        self.raise_exploration_rate = raise_exploration_rate
        
        # Removed normalization: using raw scores so no scaling is applied.
        # if variant == "kuhn_3p":
        #     self.model.set_variant_scale(5.0) 
        # elif variant == "kuhn_2p":
        #     self.model.set_variant_scale(3.0)
        
        # Reputation tracking
        self.reputation = {
            "bluff_count": 0,
            "fold_count": 0,
            "aggressive_count": 0,
            "passive_count": 0,
            "hands_played": 0,
            "opponent_folds": {0: 0, 1: 0, 2: 0}  # Track how often each opponent folds to us
        }
        
        # Game history tracking
        self.hand_history = []  # Store outcomes of recent hands
        
    def get_action(self, card, available_actions, round_num, chips_remaining, public_state=None):
        self.current_card = card
        state = self._preprocess_state(card, available_actions, round_num, chips_remaining, public_state)
        
        # Get the device that the model is on
        # # Check if MPS is available
        # if torch.backends.mps.is_available():
        #     device = torch.device("mps")
        #     print(f"Using MPS device")
        # else:
        #     print(f"MPS requested but not available, falling back to CPU")
        #     device = torch.device("cpu")
        
        # print(f"frl_agent.py: Using device: {device}")
        # Move tensors to the same device as the model
        # state_tensor = torch.tensor(state, dtype=torch.float32).to(device)
        
        # Epsilon-greedy exploration
        if random.random() < self.epsilon:
            action_idx = random.choice(list(available_actions))
        else:
            # Use model for exploitation
            action_idx = self.model.select_action(state, available_actions)
        
        raise_amount = 0
        if action_idx == 4:  # raise
            min_r = 1
            if public_state and 'min_raise' in public_state:
                min_r = public_state['min_raise']
                
            if random.random() < self.raise_exploration_rate:
                raise_amount = random.randint(min_r, chips_remaining)
            else:
                raise_amount = max(min_r, chips_remaining // 2)

        return action_idx, raise_amount
   
    def _preprocess_state(self, card, available_actions, round_num, chips_remaining, public_state=None):
        """
        Convert the poker state into a feature vector for the model using raw values without normalization
        """
        # Card one-hot encoding (J=0, Q=1, K=2, A=3)
        if isinstance(card, str):
            card_map = {'J': 0, 'Q': 1, 'K': 2, 'A': 3}
            card = card_map.get(card, None)

        card_one_hot = [0, 0, 0, 0]
        if card is not None and 0 <= card <= 3:
            card_one_hot[card] = 1
            
        # Available actions one-hot encoding
        actions_one_hot = [0, 0, 0, 0, 0]  # check, bet, call, fold, raise
        for action in available_actions:
            if action < len(actions_one_hot):
                actions_one_hot[action] = 1
                
        # Use raw round number without normalization
        round_raw = round_num
        
        # Raw betting history counts without normalization
        history_features = [
            self.betting_history.count('check'),
            self.betting_history.count('bet'),
            self.betting_history.count('call'),
            self.betting_history.count('fold'),
        ]
        
        # Process chips remaining - use raw values
        if isinstance(chips_remaining, dict):
            chips_list = list(chips_remaining.values())
        else:
            # Default equal distribution if no chip info
            chips_list = [100, 100, 100]  # Use raw default value
            
        # Combine all features
        state_vector = card_one_hot + actions_one_hot + [round_raw] + history_features + chips_list
        
        # Incorporate public state features (raw values)
        if public_state:
            pot_size = public_state.get("pot_size", 0)
            highest_bet = public_state.get("highest_bet", 0)
            min_raise = public_state.get("min_raise", 1)
            state_vector += [pot_size, highest_bet, min_raise]

        # Enhanced state with reputation and game history features
        if hasattr(self, 'reputation') and self.reputation["hands_played"] > 0:
            # Use raw counts instead of rates
            bluff_count = self.reputation["bluff_count"]
            fold_count = self.reputation["fold_count"]
            aggressive_count = self.reputation["aggressive_count"]
            passive_count = self.reputation["passive_count"]
            hands_played = self.reputation["hands_played"]
            
            # Raw count of winning hands
            wins_count = sum(1 for r in self.hand_history if r > 0)
            
            # Add to state vector
            reputation_features = [
                bluff_count,
                fold_count,
                aggressive_count,
                passive_count,
                hands_played,
                wins_count,
                len(self.hand_history)
            ]
            
            # Add opponent fold raw counts
            if public_state and 'player_id' in public_state:
                my_id = public_state['player_id']
                opponent_fold_counts = []
                for i in range(3):  # Assuming 3 player game
                    if i != my_id:
                        fold_count_to_me = self.reputation["opponent_folds"][i]
                        opponent_fold_counts.append(fold_count_to_me)
                reputation_features.extend(opponent_fold_counts)
            
            # Add to state vector
            state_vector.extend(reputation_features)
        
        return state_vector
    
    def update_reputation(self, action, result, high_card, bet_size=0, opponents_folded=None):
        """
        Update agent's reputation metrics based on actions and outcomes
        
        Args:
            action: The action taken (0-4)
            result: Win/loss result
            high_card: Whether agent had the high card
            bet_size: Size of bet/raise if applicable
            opponents_folded: List of opponents who folded
        """
        self.reputation["hands_played"] += 1
        
        # Track bluffing (betting/raising with low card and winning)
        if (action in [1, 4]) and result > 0 and not high_card:
            self.reputation["bluff_count"] += 1
            
        # Track folding
        if action == 3:
            self.reputation["fold_count"] += 1
            
        # Track aggression
        if action in [1, 4]:
            self.reputation["aggressive_count"] += 1
            
            # Track if bet size is large relative to pot
            if bet_size > 2:  # Arbitrary threshold for "large" bet
                self.reputation["aggressive_count"] += 0.5
        else:
            self.reputation["passive_count"] += 1
            
        # Track opponents folding to this agent
        if opponents_folded:
            for opp_id in opponents_folded:
                if opp_id != self.player_id:
                    self.reputation["opponent_folds"][opp_id] += 1
                    
        # Store hand history (last 20 hands)
        self.hand_history.append(result)
        if len(self.hand_history) > 20:
            self.hand_history.pop(0)
    
    def record_local_transition(self, transition):
        """
        Record a transition for local training
        
        This is called by the KuhnPokerEngine automatically
        """
        # Update betting history
        action_str = ''
        if transition['chosen_action'] == 0:
            action_str = 'check'
        elif transition['chosen_action'] == 1:
            action_str = 'bet'
        elif transition['chosen_action'] == 2:
            action_str = 'call'
        elif transition['chosen_action'] == 3:
            action_str = 'fold'
        elif transition['chosen_action'] == 4:
            action_str = 'raise'
            
        self.betting_history += action_str + ','
        
        # Extract state from transition
        my_player_id = transition['current_player']
        my_card_key = f'player{my_player_id}_card'
        
        # Extract chips info
        chips_str = transition['state'].get('chips', '')
        chips_dict = {}
        if chips_str:
            for i, c in enumerate(chips_str.split(';')):
                if c:
                    chips_dict[i] = float(c)
        
        state = self._preprocess_state(
            transition['state'].get(my_card_key),
            transition['legal_actions'],
            1 if transition['stage'] == 'first' else 2,
            chips_dict
        )
        
        action = transition['chosen_action']
        reward = transition['reward']
        done = transition['done']
        
        # Store in model memory for training
        if hasattr(self, 'prev_state') and hasattr(self, 'prev_action'):
            self.model.remember(self.prev_state, self.prev_action, reward, state, done)
        
        # Store current state and action for next transition
        if not done:
            self.prev_state = state
            self.prev_action = action
        else:
            # Reset for new episode
            delattr(self, 'prev_state') if hasattr(self, 'prev_state') else None
            delattr(self, 'prev_action') if hasattr(self, 'prev_action') else None
            self.betting_history = ""
        
        # Store in local transitions for analysis
        self.local_transitions.append({
            'state': state,
            'action': action,
            'reward': reward,
            'done': done,
            'variant': self.variant
        })
    
    def train_local(self, learning_rate=None):
        """
        Perform local training on the agent's model and return metrics (as a dict).
        
        Args:
            learning_rate: Optional learning rate override for this training step
        """
        # Update learning rate if provided
        if learning_rate is not None and hasattr(self.model, 'optimizer'):
            for param_group in self.model.optimizer.param_groups:
                param_group['lr'] = learning_rate
        
        result = self.model.train()  # returns (grads, metrics) or None
        if result:
            _, metrics = result
            return metrics
        return {}
    
    def get_model_params(self):
        """Get the current model parameters"""
        return self.model.get_model_params()
    
    def set_model_params(self, params):
        """Update model with new parameters (from server or other agents)"""
        self.model.set_model_params(params)
    
    def save_model(self, path):
        """Save the model to disk"""
        torch.save(self.model.model.state_dict(), path)
    
    def load_model(self, path):
        """Load the model from disk"""
        self.model.model.load_state_dict(torch.load(path))
    
    def get_gradients(self):
        """
        Compute gradients from the model without applying them.
        Returns a list of gradient tensors if available, else an empty list.
        """
        result = self.model.compute_gradients()
        if result is not None:
            grads, metrics = result
            return grads
        return []