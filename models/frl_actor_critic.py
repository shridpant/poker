import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

class ActorCriticNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(ActorCriticNetwork, self).__init__()
        # Embedding layer with residual connections
        self.input_layer = nn.Linear(state_dim, hidden_dim)
        self.embedding = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Actor (policy) head
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # Value head with extra layer for stability
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, 1)
        )
        
        # Layernorm for stabilizing training
        self.layer_norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, x):
        base = self.input_layer(x)
        embedding = self.embedding(base) + base  # Residual connection
        embedding = self.layer_norm(embedding)  # Normalization for stability
        
        action_logits = self.actor(embedding)
        value = self.critic(embedding)
        return action_logits, value
    
    def get_action_probs(self, x):
        # Applying input layer first, just as in the forward method
        base = self.input_layer(x)
        # Adding the residual connection like in forward
        embedding = self.embedding(base) + base
        # Applying layer normalization for consistency
        embedding = self.layer_norm(embedding)
        # Then getting action logits and applying softmax
        action_logits = self.actor(embedding)
        return F.softmax(action_logits, dim=-1)
    
    def get_value(self, x):
        # Applying input layer first, just like in the forward method
        base = self.input_layer(x)
        # Adding the residual connection like in forward
        embedding = self.embedding(base) + base
        # Applying layer normalization for consistency
        embedding = self.layer_norm(embedding)
        # Then getting the value estimate
        value = self.critic(embedding)
        return value

class FRLActorCritic:
    def __init__(self, state_dim, action_dim, learning_rate=0.0003, hidden_dim=256, batch_size=64):
        self.device = torch.device(
            "mps" if torch.backends.mps.is_available() else "cpu"
        )

        self.model = ActorCriticNetwork(state_dim, action_dim, hidden_dim)
        self.model.to(self.device)
        self.state_dim = state_dim  # fixed feature length
        self.optimizer = torch.optim.Adam(self.model.parameters(), 
                                          lr=learning_rate,
                                          eps=1e-5)
        self.gamma = 0.95
        self.gae_lambda = 0.92
        self.entropy_coef = 0.05
        self.memory = []
        self.last_entropy = None 
        self.variant_scale = 1.0
        self.batch_size = batch_size
        
        # Dynamically tracking reward statistics for adaptive normalization
        self.reward_stats = {
            "count": 0,
            "mean": 0,
            "min": float('inf'),
            "max": float('-inf'),
            "std": 1.0,
            "recent_rewards": []
        }
        
        # Adding target network
        self.target_model = ActorCriticNetwork(state_dim, action_dim, hidden_dim)
        self.target_model.to(self.device)
        self.target_update_freq = 5  # Updating target network every 5 training steps
        self.train_step_counter = 0
        self.target_update_tau = 0.01  # Soft update parameter
        
        # Copying weights to target network
        self.update_target_network(tau=1.0)  # Hard update at initialization

        # Separating optimizers for actor and critic
        self.actor_optimizer = torch.optim.Adam(
            [p for name, p in self.model.named_parameters() if 'actor' in name],
            lr=learning_rate, eps=1e-5
        )
        self.critic_optimizer = torch.optim.Adam(
            [p for name, p in self.model.named_parameters() if 'critic' in name],
            lr=learning_rate * 0.5,  # Lower learning rate for critic (half)
            eps=1e-5
        )

    def set_variant_scale(self, scale):
        """Set the scaling constant for reward normalization based on game variant"""
        self.variant_scale = scale
    
    def update_reward_statistics(self, reward):
        """Update running statistics for adaptive normalization"""
        self.reward_stats["count"] += 1
        self.reward_stats["min"] = min(self.reward_stats["min"], reward)
        self.reward_stats["max"] = max(self.reward_stats["max"], reward)
        
        # Updating running mean
        delta = reward - self.reward_stats["mean"]
        self.reward_stats["mean"] += delta / self.reward_stats["count"]
        
        # Keeping a window of recent rewards for standard deviation
        self.reward_stats["recent_rewards"].append(reward)
        if len(self.reward_stats["recent_rewards"]) > 100:  # Keep only last 100 rewards
            self.reward_stats["recent_rewards"].pop(0)
            
        # Recalculating standard deviation if we have enough samples
        if len(self.reward_stats["recent_rewards"]) > 1:
            self.reward_stats["std"] = max(1.0, np.std(self.reward_stats["recent_rewards"]))
    
    def adaptive_normalize(self, reward):
        """
        Normalize rewards using running statistics for more stable learning
        """
        # Updating statistics
        self.update_reward_statistics(reward)
        
        # Applying clipping to prevent extremely large rewards/penalties
        normalized = reward / max(1.0, self.reward_stats["std"])
        return torch.clamp(torch.tensor(normalized, dtype=torch.float32), -10, 10).item()
    
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
        # Ensuring fixed-length state vector
        state = self._ensure_vector(state)
        state_tensor = torch.tensor(
            state, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        action_logits, _ = self.model(state_tensor)
        action_probs = F.softmax(action_logits, dim=-1)
        
        # Masking unavailable actions
        mask = torch.zeros(action_probs.size(-1), device=self.device)
        for action in available_actions:
            mask[action] = 1
        masked_probs = action_probs * mask
        if masked_probs.sum() == 0:
            raise ValueError(f"No valid actions for mask {mask} at state {state}")
        
        # Normalizing probabilities after masking
        if masked_probs.sum() > 0:
            masked_probs = masked_probs / masked_probs.sum()
        else:
            # Fallback to uniform distribution over available actions
            masked_probs = mask / mask.sum()
        
        # Sampling action based on probabilities
        action_idx = torch.multinomial(masked_probs, 1).item()

        # Calculating policy entropy for diagnostics
        dist = torch.distributions.Categorical(masked_probs.squeeze(0))
        entropy = dist.entropy()
        self.last_entropy = entropy.item()

        return action_idx
    
    def get_last_entropy(self):
        """Return entropy recorded during the last action selection."""
        return getattr(self, "last_entropy", None)
    
    def remember(self, state, action, reward, next_state, done):
        """Store transition in memory with priority based on reward magnitude"""
        # Ensuring fixed-length state vectors
        state = self._ensure_vector(state)
        next_state = self._ensure_vector(next_state)
        # Applying adaptive normalization
        norm_reward = self.adaptive_normalize(reward)
        
        # Calculating priority (larger absolute reward = higher priority)
        priority = abs(norm_reward) + 0.01  # Small constant to ensure non-zero priority
        
        # Adding to memory and printing debugging info
        self.memory.append((state, action, norm_reward, next_state, done, priority))
        if abs(reward) > 0:
            print(f"[MEMORY] Added transition with reward {reward} (normalized: {norm_reward:.4f}, priority: {priority:.4f})")
    
    def get_model_params(self):
        """Get the current model parameters"""
        return [param.data.clone() for param in self.model.parameters()]
    
    def set_model_params(self, params):
        """Update model parameters"""
        for target_param, param in zip(self.model.parameters(), params):
            target_param.data.copy_(param)
    
    def compute_gradients(self):
        """Compute gradients without applying them"""
        if not self.memory:
            print("[DEBUG] No memory to train on")
            return {}
        
        effective_batch = min(self.batch_size, len(self.memory))
        print(f"[DEBUG] Training on {effective_batch} samples from memory buffer of size {len(self.memory)}")
        
        # Calculating total priority
        total_priority = sum(mem[5] for mem in self.memory)
        # Sampling based on priority
        if total_priority > 0:
            probs = [mem[5]/total_priority for mem in self.memory]
            batch_indices = np.random.choice(len(self.memory), effective_batch, p=probs, replace=False)
            batch = [self.memory[idx] for idx in batch_indices]
        else:
            batch = random.sample(self.memory, effective_batch)

        # Extracting batch components
        states      = torch.tensor([b[0] for b in batch], dtype=torch.float32, device=self.device)
        actions     = torch.tensor([b[1] for b in batch], dtype=torch.long, device=self.device)
        rewards     = torch.tensor([b[2] for b in batch], dtype=torch.float32, device=self.device)
        next_states = torch.tensor([b[3] for b in batch], dtype=torch.float32, device=self.device)
        dones       = torch.tensor([b[4] for b in batch], dtype=torch.float32, device=self.device)
        
        # Getting current policy logits and values
        action_logits, values = self.model(states)
        action_probs = F.softmax(action_logits, dim=-1)
        dist = torch.distributions.Categorical(action_probs)
        log_probs = dist.log_prob(actions)
        
        # TD(λ) returns calculation for more stable critic learning
        with torch.no_grad():
            next_values = self.target_model.get_value(next_states)
            if next_values.dim() == 0:
                next_values = next_values.unsqueeze(0)
            else:
                next_values = next_values.squeeze()

        # Ensuring values has the right shape
        if values.dim() == 0:
            values = values.unsqueeze(0)
        else:
            values = values.squeeze()
            
        # TD(λ) returns
        returns = torch.zeros_like(rewards)
        lambda_param = 0.8  # Slightly lower than GAE lambda
        future_return = 0.0

        for t in reversed(range(len(rewards))):
            mask = 1.0 - dones[t]
            future_return = rewards[t] + self.gamma * ((1 - lambda_param) * next_values[t] + 
                                                      lambda_param * future_return) * mask
            returns[t] = future_return

        advantages = returns - values  # Standard advantage calculation

        # Losses with PPO-style clipping for stability
        # Actor loss calculation
        entropy = dist.entropy().mean()
        old_log_probs = log_probs.detach()
        ratio = torch.exp(log_probs - old_log_probs)
        surr1 = ratio * advantages.detach()  # Note the detach here
        surr2 = torch.clamp(ratio, 0.8, 1.2) * advantages.detach()
        actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy

        # Critic loss (pure Huber loss for stability)
        critic_loss = F.smooth_l1_loss(values, returns.detach())

        # Applying gradients separately
        self.actor_optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        # Getting gradients directly from actor parameters
        actor_params = list(self.model.actor.parameters())
        actor_grads = [param.grad.data.clone() for param in actor_params if param.grad is not None]

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # Getting gradients directly from critic parameters
        critic_params = list(self.model.critic.parameters())
        critic_grads = [param.grad.data.clone() for param in critic_params if param.grad is not None]

        # Combining for returning
        all_grads = actor_grads + critic_grads
        
        return all_grads, {"actor_loss": actor_loss.item(), "critic_loss": critic_loss.item()}
    
    def apply_gradients(self, grads):
        """Apply provided gradients to model parameters"""
        # Getting actor and critic parameters
        actor_params = list(self.model.actor.parameters())
        critic_params = list(self.model.critic.parameters())
        
        # Splitting grads list into actor and critic parts
        actor_count = len(actor_params)
        actor_grads = grads[:actor_count]
        critic_grads = grads[actor_count:]
        
        # Zero gradients
        self.optimizer.zero_grad()
        
        # Applying actor gradients
        for param, grad in zip(actor_params, actor_grads):
            param.grad = grad.clone()
        
        # Applying critic gradients
        for param, grad in zip(critic_params, critic_grads):
            param.grad = grad.clone()
        
        # Step the main optimizer
        self.optimizer.step()
        
        # Updating target network if needed
        self.train_step_counter += 1
        if self.train_step_counter % self.target_update_freq == 0:
            self.update_target_network()
    
    def store_transition(self, state, action, reward, next_state, done):
        """
        Put one (s, a, r, s', done) tuple into replay memory.
        """
        self.memory.append((state, action, reward, next_state, done))
    
    def train(self):
        """Train the model on collected experiences"""
        result = self.compute_gradients()
        if result:
            grads, metrics = result
            self.apply_gradients(grads)
            
            if len(self.memory) > self.batch_size * 3:
                # Keeping highest priority experiences plus some random ones for diversity
                # Sorting by priority (last element in each tuple)
                sorted_memory = sorted(self.memory, key=lambda x: x[5], reverse=True)
                
                # Keeping top 50% of high-priority experiences
                top_experiences = sorted_memory[:self.batch_size]
                
                # And random 50% for exploration/diversity
                remaining = random.sample(sorted_memory[self.batch_size:], 
                                         min(self.batch_size, len(sorted_memory)-self.batch_size))
                
                self.memory = top_experiences + remaining
            
            return grads, metrics
        return None, {}
    
    def to(self, device):
        self.model.to(device)

    def update_target_network(self, tau=None):
        """Update target network by polyak averaging or hard update"""
        tau = tau if tau is not None else self.target_update_tau
        
        with torch.no_grad():
            for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
                target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
