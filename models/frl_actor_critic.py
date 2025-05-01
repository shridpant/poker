import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ActorCriticNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(ActorCriticNetwork, self).__init__()
        # Embedding layer for processing state information
        self.embedding = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor (policy) head
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # Critic (value) head
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x):
        embedding = self.embedding(x)
        action_logits = self.actor(embedding)
        value = self.critic(embedding)
        return action_logits, value
    
    def get_action_probs(self, x):
        embedding = self.embedding(x)
        action_logits = self.actor(embedding)
        return F.softmax(action_logits, dim=-1)
    
    def get_value(self, x):
        embedding = self.embedding(x)
        value = self.critic(embedding)
        return value

class FRLActorCritic:
    def __init__(self, state_dim, action_dim, learning_rate=0.001, hidden_dim=128):
        # ------------------------------------------------------------------
        # Device selection: prefer Apple‑Silicon MPS, else CPU
        # ------------------------------------------------------------------
        self.device = torch.device(
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
        self.model = ActorCriticNetwork(state_dim, action_dim, hidden_dim)
        self.model.to(self.device)
        self.state_dim = state_dim  # fixed feature length
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.gamma = 0.99  # Discount factor
        self.gae_lambda = 0.95  # GAE smoothing parameter
        self.entropy_coef = 0.01        # NEW – encourages exploration
        self.memory = []
        self.last_entropy = None  # initialise entropy tracker
        self.variant_scale = 1.0  # Default scaling factor
        
        # Dynamically track reward statistics for adaptive normalization
        self.reward_stats = {
            "count": 0,
            "mean": 0,
            "min": float('inf'),
            "max": float('-inf'),
            "std": 1.0,
            "recent_rewards": []
        }
        
    def set_variant_scale(self, scale):
        """Set the scaling constant for reward normalization based on game variant"""
        self.variant_scale = scale
    
    def update_reward_statistics(self, reward):
        """Update running statistics for adaptive normalization"""
        self.reward_stats["count"] += 1
        self.reward_stats["min"] = min(self.reward_stats["min"], reward)
        self.reward_stats["max"] = max(self.reward_stats["max"], reward)
        
        # Update running mean
        delta = reward - self.reward_stats["mean"]
        self.reward_stats["mean"] += delta / self.reward_stats["count"]
        
        # Keep a window of recent rewards for standard deviation
        self.reward_stats["recent_rewards"].append(reward)
        if len(self.reward_stats["recent_rewards"]) > 100:  # Keep only last 100 rewards
            self.reward_stats["recent_rewards"].pop(0)
            
        # Recalculate standard deviation if we have enough samples
        if len(self.reward_stats["recent_rewards"]) > 1:
            self.reward_stats["std"] = max(1.0, np.std(self.reward_stats["recent_rewards"]))
    
    def adaptive_normalize(self, reward):
        """
        Chip‑ratio normalization.

        Rewards are scaled by ``self.variant_scale`` (usually the starting stack
        or blind amount) so that they stay in a consistent numeric range
        across different Kuhn‑poker variants.  For standard two‑chip ante
        games, ``variant_scale`` should be set to 2.0, yielding rewards in
        approximately ``[-1, 1]``.
        """
        return reward / self.variant_scale
    
    def dual_scale_normalize(self, reward, lambda_param=0.2):
        """Implement dual-scale normalization from the methodology"""
        # Primary scaling by variant constant
        primary_scale = reward / self.variant_scale
        
        # Secondary standardization component (use running stats in practice)
        reward_std = max(1.0, abs(reward))  # Simple approximation
        secondary_scale = reward / reward_std
        
        # Combine using lambda parameter
        return primary_scale + lambda_param * secondary_scale

    def _ensure_vector(self, state):
        """
        Convert an incoming state representation to a fixed‑length vector of
        length ``self.state_dim``.  If the input is shorter, pad with zeros;
        if longer, truncate.  Accepts list, tuple, NumPy array, or scalar.
        """
        if isinstance(state, np.ndarray):
            state = state.tolist()
        elif not isinstance(state, (list, tuple)):
            state = [state]

        if len(state) < self.state_dim:
            state = list(state) + [0.0] * (self.state_dim - len(state))
        elif len(state) > self.state_dim:
            state = list(state)[: self.state_dim]

        return state

    def select_action(self, state, available_actions):
        """Select an action using the model's policy with masking for unavailable actions"""
        # Ensure fixed-length state vector
        state = self._ensure_vector(state)
        state_tensor = torch.tensor(
            state, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        action_logits, _ = self.model(state_tensor)
        action_probs = F.softmax(action_logits, dim=-1)
        
        # Mask unavailable actions
        mask = torch.zeros(action_probs.size(-1), device=self.device)
        for action in available_actions:
            mask[action] = 1
        masked_probs = action_probs * mask
        if masked_probs.sum() == 0:
            raise ValueError(f"No valid actions for mask {mask} at state {state}")
        
        # Normalize probabilities after masking
        if masked_probs.sum() > 0:
            masked_probs = masked_probs / masked_probs.sum()
        else:
            # Fallback to uniform distribution over available actions
            masked_probs = mask / mask.sum()
        
        # Sample action based on probabilities
        action_idx = torch.multinomial(masked_probs, 1).item()

        # Calculate policy entropy for diagnostics
        dist = torch.distributions.Categorical(masked_probs.squeeze(0))
        entropy = dist.entropy()
        self.last_entropy = entropy.item()

        return action_idx
    
    def get_last_entropy(self):
        """Return entropy recorded during the last action selection."""
        return getattr(self, "last_entropy", None)
    
    def remember(self, state, action, reward, next_state, done):
        """Store transition in memory using adaptive normalization"""
        # Ensure fixed-length state vectors
        state = self._ensure_vector(state)
        next_state = self._ensure_vector(next_state)
        # Apply adaptive normalization
        norm_reward = self.adaptive_normalize(reward)
        self.memory.append((state, action, norm_reward, next_state, done))
    
    def get_model_params(self):
        """Get the current model parameters"""
        return [param.data.clone() for param in self.model.parameters()]
    
    def set_model_params(self, params):
        """Update model parameters"""
        for target_param, param in zip(self.model.parameters(), params):
            target_param.data.copy_(param)
    
    def compute_gradients(self):
        """Compute gradients without applying them"""
        if len(self.memory) < 1:
            return None
            
        # Process memories
        states = torch.tensor([mem[0] for mem in self.memory],
                              dtype=torch.float32, device=self.device)
        actions = torch.tensor([mem[1] for mem in self.memory],
                               dtype=torch.long, device=self.device)
        rewards = torch.tensor([mem[2] for mem in self.memory],
                               dtype=torch.float32, device=self.device)
        next_states = torch.tensor([mem[3] for mem in self.memory],
                                   dtype=torch.float32, device=self.device)
        dones = torch.tensor([mem[4] for mem in self.memory],
                             dtype=torch.float32, device=self.device)
        
        # Get current policy logits and values
        action_logits, values = self.model(states)
        action_probs = F.softmax(action_logits, dim=-1)
        dist = torch.distributions.Categorical(action_probs)
        log_probs = dist.log_prob(actions)
        
        # ------------------------------------------------------------------
        # Generalized Advantage Estimation (GAE-λ)
        # ------------------------------------------------------------------
        with torch.no_grad():
            next_values = self.model.get_value(next_states).squeeze()

        values = values.squeeze()
        advantages = torch.zeros_like(rewards)

        gae = 0.0
        for t in reversed(range(len(rewards))):
            mask = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * next_values[t] * mask - values[t]
            gae = delta + self.gamma * self.gae_lambda * mask * gae
            advantages[t] = gae

        returns = advantages + values  # target for value function

        # ------------------------------------------------------------------
        # Losses
        # ------------------------------------------------------------------
        entropy = dist.entropy().mean()
        actor_loss = -(log_probs * advantages.detach()).mean() - self.entropy_coef * entropy
        critic_loss = F.mse_loss(values, returns.detach())
        loss = actor_loss + 0.5 * critic_loss  # 0.5 scales value loss
        
        # Compute gradients
        self.optimizer.zero_grad()
        loss.backward()
        
        # Get gradients
        grads = [param.grad.data.clone() for param in self.model.parameters() if param.grad is not None]
        
        return grads, {"actor_loss": actor_loss.item(), "critic_loss": critic_loss.item()}
    
    def apply_gradients(self, grads):
        """Apply provided gradients to model"""
        for param, grad in zip([p for p in self.model.parameters() if p.requires_grad], grads):
            if param.grad is None:
                param.grad = grad.clone()
            else:
                param.grad.data.copy_(grad)
        self.optimizer.step()
    
    def train(self):
        """Train the model on collected experiences"""
        result = self.compute_gradients()
        if result:
            grads, metrics = result
            self.apply_gradients(grads)
            self.memory = []  # Clear memory after training
            return grads, metrics
        return None, {}
    
    def to(self, device):
        self.model.to(device)
