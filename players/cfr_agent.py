import random
import json
import sys
import os
import time
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class RaiseValueNetwork(nn.Module):
    """Neural network to predict optimal raise amounts based on game state."""
    def __init__(self, input_size=14, hidden_size=32):
        super(RaiseValueNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()  # Output between 0 and 1 to scale by available chips
        )
    
    def forward(self, x):
        return self.model(x)


class RaiseExperienceDataset(Dataset):
    """Dataset for training the raise amount network."""
    def __init__(self, experiences):
        self.states = [exp['state'] for exp in experiences]
        self.amounts = [exp['amount'] for exp in experiences]
        self.rewards = [exp['reward'] for exp in experiences]
    
    def __len__(self):
        return len(self.states)
    
    def __getitem__(self, idx):
        return {
            'state': torch.FloatTensor(self.states[idx]),
            'amount': torch.FloatTensor([self.amounts[idx]]),
            'reward': torch.FloatTensor([self.rewards[idx]])
        }


class CFRNode:
    """Node in the counterfactual regret minimization tree."""
    def __init__(self, infoset, num_actions=5):
        self.infoset = infoset
        self.num_actions = num_actions
        self.regret_sum = [0.0] * num_actions
        self.strategy = [0.0] * num_actions
        self.strategy_sum = [0.0] * num_actions
        self.visits = 0

    def get_strategy(self, realization_weight):
        """Compute current strategy via regret-matching and accumulate for averaging."""
        normalizing_sum = 0.0
        for a in range(self.num_actions):
            self.strategy[a] = max(0, self.regret_sum[a])
            normalizing_sum += self.strategy[a]
        
        for a in range(self.num_actions):
            if normalizing_sum > 0:
                self.strategy[a] /= normalizing_sum
            else:
                self.strategy[a] = 1.0 / self.num_actions
            
            self.strategy_sum[a] += realization_weight * self.strategy[a]
        
        self.visits += 1
        return self.strategy

    def get_average_strategy(self):
        """Returns the average strategy across all training iterations."""
        avg_strategy = [0.0] * self.num_actions
        normalizing_sum = sum(self.strategy_sum)
        
        if normalizing_sum > 0:
            for a in range(self.num_actions):
                avg_strategy[a] = self.strategy_sum[a] / normalizing_sum
        else:
            for a in range(self.num_actions):
                avg_strategy[a] = 1.0 / self.num_actions
                
        return avg_strategy


class EnhancedCFRTrainer:
    """Trains CFR strategy for both 2 and 3-player Kuhn poker with all actions."""
    def __init__(self, iterations=1000000, num_players=2, out_dir=None):
        self.node_map = {}  # maps infoset -> CFRNode
        self.iterations = iterations
        self.num_players = num_players
        self.num_actions = 5  # check, bet, call, fold, raise
        
        # Determine output directory for strategy file
        base = out_dir or os.path.join(os.path.dirname(__file__), '..', 'logs', 'game_data')
        self.out_dir = os.path.normpath(base)
        os.makedirs(self.out_dir, exist_ok=True)
        
        # Card mapping
        self.card_values = {"J": 0, "Q": 1, "K": 2, "A": 3}
        self.action_mapping = {0: "check", 1: "bet", 2: "call", 3: "fold", 4: "raise"}
        self.reverse_action_mapping = {"check": 0, "bet": 1, "call": 2, "fold": 3, "raise": 4}

    def train(self):
        """Train CFR for the specified number of iterations and save strategy to JSON."""
        util = 0.0
        if self.num_players == 2:
            cards = [0, 1, 2]  # J, Q, K for 2-player
        else:
            cards = [0, 1, 2, 3]  # J, Q, K, A for 3-player
        
        # Values to track for plotting
        iterations_to_plot = []
        avg_game_values = []
        exploitabilities = []
        
        # Track values every N iterations
        plot_interval = max(1, self.iterations // 100)
        
        print(f"Training CFR for {self.num_players}-player Kuhn poker...")
        for i in range(self.iterations):
            if i % 10000 == 0 and i > 0:
                print(f"Iteration {i}/{self.iterations}")
                
            random.shuffle(cards)
            player_cards = cards[:self.num_players]
            iter_util = self._cfr(player_cards, "", [1.0] * self.num_players, 0)
            util += iter_util
            
            # Record values for plotting at specified intervals
            if i % plot_interval == 0 or i == self.iterations - 1:
                avg_value = util / (i + 1)
                iterations_to_plot.append(i + 1)
                avg_game_values.append(avg_value)
                
                # Calculate current exploitability
                # exploitability = self._calculate_exploitability()
                # exploitabilities.append(exploitability)
            
        avg_game_value = util / self.iterations
        print(f"Average game value: {avg_game_value:.3f}")

        # Create and save the plot
        self._create_convergence_plot(iterations_to_plot, avg_game_values)
        
        # Save average strategy
        strategy = {infoset: node.get_average_strategy()
                for infoset, node in self.node_map.items()}
        
        filename = f"cfr_strategy_{self.num_players}p.json"
        filepath = os.path.join(self.out_dir, filename)
        with open(filepath, "w") as f:
            json.dump(strategy, f, indent=2)
        print(f"[CFRTrainer] Strategy saved to {filepath}")

    def _create_convergence_plot(self, iterations, game_values):
        """Creates and saves a plot of game value vs iterations"""
        plt.figure(figsize=(10, 6))
        plt.plot(iterations, game_values, color='#ff6900', linewidth=2)
        plt.xlabel('Iterations', fontsize=14)
        plt.ylabel('Average Game Value', fontsize=14)
        plt.title(f'External Sampled CFR Convergence in {self.num_players}-player Kuhn Poker', fontsize=16)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Save the plot
        plot_path = os.path.join(self.out_dir, f'cfr_convergence_{self.num_players}p.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Convergence plot saved to {plot_path}")
        plt.close()


    def _cfr(self, cards, history, reach_probs, current_player):
        """Main CFR recursive function."""
        # Terminal state: calculate utility
        if self._is_terminal(history):
            return self._utility(history, cards, current_player)
        
        # Check if current player's turn is over
        next_player = self._get_next_player(history, current_player)
        
        # Create infoset key: card + history
        infoset = str(cards[current_player]) + history
        
        # Get or create node
        if infoset not in self.node_map:
            self.node_map[infoset] = CFRNode(infoset, self.num_actions)
        
        node = self.node_map[infoset]
        
        # Get current strategy for this infoset
        strategy = node.get_strategy(reach_probs[current_player])
        
        # Available actions based on history
        available_actions = self._get_available_actions(history)
        
        # Initialize utility and action utilities
        action_utils = np.zeros(self.num_actions)
        node_util = 0
        
        # Recursively call for each action
        for action_idx in available_actions:

            # Create new history after this action
            new_history = self._append_action_to_history(history, action_idx)
            
            # Calculate new reach probabilities
            new_reach_probs = reach_probs.copy()
            new_reach_probs[current_player] *= strategy[action_idx]
            
            action_utils[action_idx] = -self._cfr(cards, new_history, new_reach_probs, next_player)
            node_util += strategy[action_idx] * action_utils[action_idx]
        
        # Update regrets
        for action_idx in available_actions:
            regret = action_utils[action_idx] - node_util
            node.regret_sum[action_idx] += (
                np.prod([reach_probs[p] for p in range(self.num_players) if p != current_player])
            ) * regret
            
        return node_util

    def _get_next_player(self, history, current_player):
        """Determine who acts next based on the current history."""
        return (current_player + 1) % self.num_players
    
    def _get_available_actions(self, history):
        """Determine available actions based on game history."""
        # Default available actions: check and bet at the start
        if not history:
            return [0, 1]  # check, bet
                
        last_action = int(history[-1])
        
        if last_action == 0:  # After check
            return [0, 1]  # can check, bet
        elif last_action == 1:  # After bet
            return [2, 3, 4]  # can call, fold, or raise
        elif last_action == 2:  # After call
            return [0, 1, 2, 3, 4]  # can check, bet,  call, fold, or raise
        elif last_action == 3:  # After fold
            # Game would end after a fold
            return [0]
        elif last_action == 4:  # After raise
            return [2, 3, 4]  # can call the raise, fold, or re-raise
                
        return [0]


    def _append_action_to_history(self, history, action):
        """Add an action to the history."""
        return history + str(action)

    def _is_terminal(self, history):
        """Check if we've reached a terminal state."""
        if not history:
            return False
                
        if len(history) >= 10:  
            return True
                
        # Fold always ends
        if "3" in history:
            return True
                
        # Check for all players checking sequence
        if len(history) >= self.num_players:
            all_checks = True
            for i in range(1, self.num_players + 1):
                if history[-i] != "0":
                    all_checks = False
                    break
            if all_checks:
                return True
                
        # After a call, check if it was in response to a bet or raise
        if history and history[-1] == "2":
            for action in reversed(history[:-1]):
                if action in ["1", "4"]: 
                    return True
                        
        return False


        
    def _utility(self, history, cards, current_player):
        """Calculate the utility at a terminal state."""
        if "3" in history:
            for i in range(len(history)):
                if history[i] == "3":
                    folded_player = i % self.num_players
                    return 1 if folded_player != current_player else -1
        
        card_values = [self._card_value(c) for c in cards]
        winners = []
        max_card = max(card_values)
        
        for p in range(self.num_players):
            if card_values[p] == max_card:
                winners.append(p)
                
        pot_contribution = 1 
        
        # Add bets/calls to pot 
        for i, action in enumerate(history):
            player = i % self.num_players
            if action == "1":  # bet
                pot_contribution += 1
            elif action == "2":  # call
                pot_contribution += 1
            elif action == "4":  # raise 
                pot_contribution += 2
        
        # If current player won
        if current_player in winners:
            # Split pot among winners
            return pot_contribution / len(winners)
        else:
            return -pot_contribution / (self.num_players - len(winners))
    
    def _card_value(self, card):
        """Return the numeric value of a card."""
        if isinstance(card, str):
            return self.card_values.get(card, -1)
        return card 


class AdvancedCFRAgent:
    """
    Advanced agent that uses CFR for action selection and neural network for raise decisions.
    Supports both 2-player and 3-player Kuhn poker with all actions.
    """
    def __init__(self, player_id=0, num_players=2, strategy_file=None):
        self.player_id = player_id
        self.num_players = num_players
        
        # Load appropriate strategy file based on num_players
        if strategy_file:
            strategy_path = strategy_file
        else:
            base_dir = os.path.join(os.path.dirname(__file__), '..', 'logs', 'game_data')
            strategy_path = os.path.join(base_dir, f"cfr_strategy_{num_players}p.json")
            
        # Fallback to 2-player if specific file not found
        if not os.path.exists(strategy_path):
            fallback_path = os.path.join(os.path.dirname(__file__), '..', 'logs', 'game_data', "cfr_strategy_2p.json")
            if os.path.exists(fallback_path):
                strategy_path = fallback_path
                print(f"Using fallback strategy: {fallback_path}")
            else:
                # Create default strategy (uniform distribution)
                self.strategy = defaultdict(lambda: [0.2, 0.2, 0.2, 0.2, 0.2])
                print("No strategy file found. Using uniform strategy.")
                strategy_path = None
                
        if strategy_path:
            try:
                with open(strategy_path, "r") as f:
                    self.strategy = defaultdict(lambda: [0.2, 0.2, 0.2, 0.2, 0.2], json.load(f))
                print(f"Loaded strategy from {strategy_path}")
            except Exception as e:
                print(f"Error loading strategy: {e}")
                self.strategy = defaultdict(lambda: [0.2, 0.2, 0.2, 0.2, 0.2])
        
        # Neural network for raise decisions
        self.input_size = 14  # Features representing game state
        self.raise_model = RaiseValueNetwork(self.input_size)
        self.optimizer = optim.Adam(self.raise_model.parameters(), lr=0.001)
        self.loss_fn = nn.MSELoss()
        
        self.experiences = []
        self.min_experiences_to_train = 50
        self.batch_size = 32
        self.exploration_rate = 0.2 
        
        # Card mapping
        self.card_values = {"J": 0, "Q": 1, "K": 2, "A": 3}
        
        # Track state for reward assignment
        self.current_round_states = []
        self.cumulative_reward = 0
    
    def get_action(self, card, available_actions, round_num, chips, public_state):
        """Interface for game engine to get the next action from this agent."""
        # Extract betting history from public state
        history = ""
        for action in public_state.get('betting_history', []):
            if action == "check":
                history += "0"
            elif action == "bet":
                history += "1"
            elif action == "call":
                history += "2"
            elif action == "fold":
                history += "3"
            elif action == "raise":
                history += "4"
        
        # Create infoset key using player's card and history
        card_idx = self._card_to_index(card)
        infoset = str(card_idx) + history
        
        # Get strategy probabilities from CFR
        strategy_probs = self.strategy.get(infoset, None)
        
        # If infoset not found, return reasonable default action
        if strategy_probs is None:
            return self._default_action(available_actions)
        
        # Filter strategy by available actions
        available_indices = list(available_actions.keys())
        filtered_probs = []
        filtered_actions = []
        
        for idx in available_indices:
            if idx < len(strategy_probs):
                filtered_probs.append(strategy_probs[idx])
                filtered_actions.append(idx)
        
        # Normalize probabilities
        if sum(filtered_probs) > 0:
            filtered_probs = [p / sum(filtered_probs) for p in filtered_probs]
        else:
            filtered_probs = [1.0 / len(filtered_actions)] * len(filtered_actions)
        
        # Select action based on probabilities
        action_idx = random.choices(filtered_actions, weights=filtered_probs, k=1)[0]
        
        # For raise actions, use neural network to determine amount
        raise_amount = None
        if action_idx == 4:  # If raising
            # Get state representation
            state_vector = self._extract_state_features(card, available_actions, round_num, chips, public_state)
            
            # Store the state for later training
            self.current_round_states.append({
                'state': state_vector.copy(),
                'action': action_idx
            })
            
            # Get raise amount from neural network
            raise_amount = self._get_raise_amount(state_vector, chips)
        
        # Store state information for all actions 
        if action_idx != 4:  
            state_vector = self._extract_state_features(card, available_actions, round_num, chips, public_state)
            self.current_round_states.append({
                'state': state_vector.copy(),
                'action': action_idx
            })
        
        return (action_idx, raise_amount)
    
    def _default_action(self, available_actions):
        """Returns a reasonable default action when strategy is unknown."""
        # Prefer call over fold, check over bet, etc.
        priority_order = [2, 0, 1, 4, 3]  # call, check, bet, raise, fold
        
        for action in priority_order:
            if action in available_actions:
                if action == 4:  
                    return (action, 1) 
                return (action, None)
        
        # Fallback to first available action
        first_action = list(available_actions.keys())[0]
        return (first_action, None)
    
    def _extract_state_features(self, card, available_actions, round_num, chips, public_state):
        """Extract features from the game state for neural network input."""
        # Card value (one-hot encoded)
        card_value = self._card_to_index(card)
        card_features = [0, 0, 0, 0]  # J, Q, K, A
        card_features[card_value] = 1
        
        # Round information
        round_feature = [round_num / 10.0]  # Normalize round number
        
        # Betting stage
        stage_feature = [1.0 if public_state.get('stage') == 'first' else 0.0]
        
        # Position features
        position_features = [0, 0, 0]  # For 3 possible positions
        position_features[min(public_state.get('current_player', 0), 2)] = 1
        
        # Pot and chips ratio
        pot_feature = [public_state.get('pot_size', 0) / 20.0]  # Normalize pot
        chips_feature = [chips / 100.0]  # Normalize chips
        
        # Betting history counts
        history = public_state.get('betting_history', [])
        bet_count = sum(1 for a in history if a == 'bet' or a == 'raise')
        fold_count = sum(1 for a in history if a == 'fold')
        betting_features = [bet_count / 5.0, fold_count / 3.0]
        
        # Highest bet in the current round
        highest_bet_feature = [public_state.get('highest_bet', 0) / 10.0]
        
        # Combine all features
        return card_features + round_feature + stage_feature + position_features + \
               pot_feature + chips_feature + betting_features + highest_bet_feature
    
    def _get_raise_amount(self, state_vector, chips):
        """Use neural network to determine raise amount."""
        # Convert state to tensor
        state_tensor = torch.FloatTensor(state_vector).unsqueeze(0)
        
        # Exploration vs Exploitation
        if random.random() < self.exploration_rate:
            # Explore: random bet between min and max
            min_raise = 1
            max_raise = chips
            return random.randint(min_raise, max(min_raise, max_raise))
        
        # Exploit: use neural network
        with torch.no_grad():
            relative_bet = self.raise_model(state_tensor).item()
        
        # Scale bet between min_raise and available chips
        min_raise = 1
        bet_amount = int(min_raise + relative_bet * (chips - min_raise))
        return max(min_raise, min(bet_amount, chips))
    
    def update_reward(self, reward):
        """Update the agent with the final reward from the round."""
        self.cumulative_reward += reward
        
        # Associate reward with all states from this round
        if self.current_round_states:
            # Create experience entries for all raise actions in this round
            for state_info in self.current_round_states:
                if state_info['action'] == 4:  # Only raises get added to experience
                    self.experiences.append({
                        'state': state_info['state'],
                        'amount': state_info.get('raise_amount', 1) / 100.0,  # Normalize amount
                        'reward': reward
                    })
            
            # Clear round states
            self.current_round_states = []
            
            # Train if we have enough experiences
            if len(self.experiences) >= self.min_experiences_to_train:
                self._train_raise_model()
    
    def _train_raise_model(self):
        """Train the neural network on collected experiences."""
        if len(self.experiences) < self.batch_size:
            return
            
        # Create dataset and dataloader
        dataset = RaiseExperienceDataset(self.experiences)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Training loop
        self.raise_model.train()
        total_loss = 0
        
        for batch in dataloader:
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            states = batch['state']
            predicted_amounts = self.raise_model(states)
            
            # Calculate loss
            # Positive rewards reinforce the bet amount, negative rewards push away
            target_amounts = batch['amount']
            rewards = batch['reward']
            
            # Use rewards as weights for the loss
            loss = 0
            for i in range(len(rewards)):
                if rewards[i] > 0:
                    # Positive reward - encourage similar bets
                    loss += self.loss_fn(predicted_amounts[i], target_amounts[i]) * rewards[i]
                else:
                    # Negative reward - discourage these bet amounts
                    if predicted_amounts[i] < target_amounts[i]:
                        ideal = target_amounts[i] + 0.2  # Push higher
                    else:
                        ideal = target_amounts[i] - 0.2  # Push lower
                    ideal = torch.clamp(ideal, 0, 1)  # Keep within valid range
                    loss += self.loss_fn(predicted_amounts[i], ideal) * abs(rewards[i])
            
            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        # Print training statistics
        print(f"Trained raise model on {len(self.experiences)} experiences. Loss: {total_loss:.4f}")
        
        # Reset experiences
        if len(self.experiences) > 1000:
            self.experiences = self.experiences[-500:]  # Keep last 500
            
        # Reduce exploration rate over time
        self.exploration_rate = max(0.05, self.exploration_rate * 0.95)
    
    def _card_to_index(self, card):
        """Convert card to numeric index."""
        return self.card_values.get(card, 0)
    
    def save_model(self, path=None):
        """Save the neural network model."""
        if path is None:
            base_dir = os.path.join(os.path.dirname(__file__), '..', 'logs', 'game_data')
            path = os.path.join(base_dir, f"raise_model_player{self.player_id}.pt")
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model_state_dict': self.raise_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'experiences': self.experiences[:100],  # Save some experiences for warm start
            'exploration_rate': self.exploration_rate
        }, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path=None):
        """Load the neural network model."""
        if path is None:
            base_dir = os.path.join(os.path.dirname(__file__), '..', 'logs', 'game_data')
            path = os.path.join(base_dir, f"raise_model_player{self.player_id}.pt")
        
        if os.path.exists(path):
            checkpoint = torch.load(path)
            self.raise_model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.experiences = checkpoint.get('experiences', [])
            self.exploration_rate = checkpoint.get('exploration_rate', 0.2)
            print(f"Model loaded from {path}")
            return True
        return False


class CFRPlayerWrapper:
    def __init__(self, player_id=0, num_players=2):
        self.player_id = player_id
        self.agent = AdvancedCFRAgent(player_id, num_players)
        self.supports_federated = True
        self.local_transitions = []
    
    def get_action(self, card, available_actions, round_num, chips, public_state):
        return self.agent.get_action(card, available_actions, round_num, chips, public_state)
    
    def record_local_transition(self, transition_data):
        self.local_transitions.append(transition_data)
        
    def update_reward(self, reward):
        self.agent.update_reward(reward)
    
    def save_model(self):
        self.agent.save_model()
    
    def load_model(self):
        return self.agent.load_model()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train CFR or run with neural network agent.")
    parser.add_argument("--train", action="store_true", help="Train CFR and save strategy.")
    parser.add_argument("--play", action="store_true", help="Play using the agent.")
    parser.add_argument("--players", type=int, default=2, choices=[2, 3],
                        help="Number of players (2 or 3)")
    parser.add_argument("--iterations", type=int, default=100000,
                        help="Number of CFR training iterations.")
    args = parser.parse_args()

    if args.train:
        trainer = EnhancedCFRTrainer(iterations=args.iterations, num_players=args.players)
        start = time.time()
        trainer.train()
        print(f"Training took {time.time() - start:.2f} seconds.")
    
    elif args.play:
        try:
            from engine.KuhnPokerEngine import KuhnPokerEngine
            
            # Create agents
            agents = [CFRPlayerWrapper(i, args.players) for i in range(args.players)]
            
            # Try to load saved models
            for agent in agents:
                agent.load_model()
            
            # Create game
            if args.players == 2:
                game = KuhnPokerEngine(agents[0], agents[1], auto_rounds=10)
            else:  # 3 players
                game = KuhnPokerEngine(agents[0], agents[1], agents[2], num_players=3, auto_rounds=10)
            
            # Run game
            game.run_game()
            
            # Save models after playing
            for agent in agents:
                agent.save_model()
                
        except ImportError:
            print("Could not import KuhnPokerEngine. Make sure the engine module is available.")
    
    else:
        print("Specify --train to train CFR or --play to play a match.")
