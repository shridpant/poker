import wandb
import sys, os
import matplotlib.pyplot as plt
import numpy as np
import argparse
import datetime
import math
import torch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from engine.KuhnPokerEngine import KuhnPokerEngine
from players.frl_agent import FRLAgent
from models.frl_aggregator import FRLAggregator

def reward_shaping(raw_reward, chips_before, chips_after, starting_chips, training_mode="selfish"):
    """
    Uses final_chips - initial_chips as the reward without any scaling or normalization.
    IMO: Raw chip differences provide more transparent learning signals.
    """
    SCALE_FACTOR = 1.0
    return (chips_after - chips_before) * SCALE_FACTOR

def validate(agents, engine, num_hands=0):
    """
    Evaluate agents against fixed opponents or each other.
    
    Args:
        agents: List of agent objects to evaluate
        engine: Poker engine instance
        num_hands: Number of hands to play for evaluation
    
    Returns:
        validation_loss and detailed metrics dictionary
    """
    # Save original chips and exploration rates
    original_chips = engine.chips.copy()
    original_exploration = [agent.epsilon for agent in agents]
    
    # Set evaluation mode (reduce exploration)
    for agent in agents:
        agent.epsilon = 0.05  # Minimal exploration during evaluation
    
    # Reset chips for fair evaluation
    engine.chips = [100] * engine.num_players
    
    # Track metrics for each agent
    metrics = {i: {"reward": 0, "wins": 0, "losses": 0, "actions": {}} for i in range(len(agents))}
    
    # Run validation hands
    for _ in range(num_hands):
        # Run a single hand
        reward_info = engine.run_round()

        if reward_info:
            for pid, rew in reward_info.items():
                stub_state = [0.0] * agents[pid].state_dim
                agents[pid].model.remember(stub_state, 0, rew, stub_state, True)
        
        if not reward_info:
            continue
            
        # Record metrics
        for i, reward in reward_info.items():
            metrics[i]["reward"] += reward
            if reward > 0:
                metrics[i]["wins"] += 1
            elif reward < 0:
                metrics[i]["losses"] += 1
    
    # Calculate win rates and average rewards
    for i in metrics:
        metrics[i]["avg_reward"] = metrics[i]["reward"] / num_hands if num_hands > 0 else 0
        metrics[i]["win_rate"] = metrics[i]["wins"] / num_hands if num_hands > 0 else 0
    
    # Restore original chips and exploration rates
    engine.chips = original_chips
    for i, agent in enumerate(agents):
        agent.epsilon = original_exploration[i]
    
    # Calculate a single validation loss value
    avg_win_rate = sum(m["win_rate"] for m in metrics.values()) / len(metrics) if metrics else 0
    validation_loss = 1.0 - avg_win_rate
    
    return validation_loss, metrics

def calculate_poker_metrics(agent_transitions, num_recent=50):
    """Calculate poker-specific metrics from agent transitions"""
    
    if not agent_transitions or len(agent_transitions) < 5:
        return {}  # Not enough data
        
    # Get most recent transitions
    recent = agent_transitions[-min(num_recent, len(agent_transitions)):]
    
    # Calculate metrics
    total_actions = len(recent)
    actions = [t['action'] for t in recent]
    rewards = [t['reward'] for t in recent]
    
    # Basic metrics
    metrics = {
        "check_rate": actions.count(0) / max(1, total_actions),
        "bet_rate": actions.count(1) / max(1, total_actions),
        "call_rate": actions.count(2) / max(1, total_actions),
        "fold_rate": actions.count(3) / max(1, total_actions),
        "raise_rate": actions.count(4) / max(1, total_actions),
        "avg_reward": sum(rewards) / max(1, len(rewards)),
        "positive_reward_rate": sum(1 for r in rewards if r > 0) / max(1, len(rewards))
    }
    
    # Calculate aggression factor (bet+raise / check+fold)
    aggressive = actions.count(1) + actions.count(4)
    passive = actions.count(0) + actions.count(3)
    metrics["aggression_factor"] = aggressive / max(1, passive)
    
    return metrics

def plot_rewards(avg_reward_agent0, avg_reward_agent1, avg_reward_agent2, avg_reward_central):
    """
    Plot separate reward graphs for each agent and the central model using raw values.
    """
    plt.figure()
    plt.plot(avg_reward_agent0, label="Agent 0")
    plt.title("Avg Reward: Agent 0")
    plt.xlabel("Hand")
    plt.ylabel("Raw Chip Difference")
    plt.legend()

    plt.figure()
    plt.plot(avg_reward_agent1, label="Agent 1")
    plt.title("Avg Reward: Agent 1")
    plt.xlabel("Hand")
    plt.ylabel("Raw Chip Difference")
    plt.legend()

    plt.figure()
    plt.plot(avg_reward_agent2, label="Agent 2")
    plt.title("Avg Reward: Agent 2")
    plt.xlabel("Hand")
    plt.ylabel("Raw Chip Difference")
    plt.legend()

    plt.figure()
    plt.plot(avg_reward_central, label="Central Aggregator")
    plt.title("Avg Reward: Central Aggregator")
    plt.xlabel("Hand")
    plt.ylabel("Raw Chip Difference")
    plt.legend()

    plt.show()

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train FRL agents for Kuhn Poker")
    parser.add_argument("--rounds", type=int, default=1, help="Number of training rounds")
    parser.add_argument("--hands_per_round", type=int, default=1, help="Number of hands to play per round")
    parser.add_argument("--aggregation", type=str, default="median", 
                       choices=["fedavg", "median", "trimmed_mean"],
                       help="Aggregation method for federated learning")
    parser.add_argument("--reset_chips", action="store_true", 
                       help="Reset chips at the beginning of each round")
    parser.add_argument("--training_mode", type=str, default="selfish",
                       choices=["selfish", "cooperative"],
                       help="Training optimization mode: individual or team performance")
    parser.add_argument("--reward_scaling", type=float, default=1.0,
                       help="Scaling factor for rewards (higher values emphasize wins/losses)")
    parser.add_argument("--initial_exploration", type=float, default=0.3,
                       help="Initial exploration rate (epsilon)")
    parser.add_argument("--min_exploration", type=float, default=0.05,
                       help="Minimum exploration rate")
    parser.add_argument("--exploration_decay", type=float, default=0.8,
                       help="Decay factor for exploration rate per round")
    parser.add_argument("--learning_rate", type=float, default=0.0005,  # Reduced from 0.001
                       help="Learning rate for model training")
    parser.add_argument("--experiment_name", type=str, default=None,
                       help="Name for this experiment run (for wandb logging)")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run the training on")
    parser.add_argument("--num_players", type=int, choices=[2, 3], default=3,
                       help="Number of players in the game (2 or 3)")
    args = parser.parse_args()
    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Add a timestamp to distinguish training sessions
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    timestamp_header = f"\n{'='*80}\n{'='*25} TRAINING SESSION STARTED AT {timestamp} {'='*25}\n{'='*80}\n"
    print(timestamp_header)
    
    # Initializing W&B for logging
    run_name = args.experiment_name or f"frl-kuhn-{timestamp.replace(' ', '_')}"
    wandb.init(project="kuhn-poker-frl", name=run_name, config={
        "rounds": args.rounds,
        "hands_per_round": args.hands_per_round,
        "aggregation_method": args.aggregation,
        "reset_chips": args.reset_chips,
        "training_mode": args.training_mode,
        "reward_scaling": args.reward_scaling,
        "learning_rate": args.learning_rate,
        "initial_exploration": args.initial_exploration,
        "min_exploration": args.min_exploration,
        "exploration_decay": args.exploration_decay,
        "num_players": args.num_players  # Add this line
    })
    wandb.define_metric("hand")
    wandb.define_metric("avg_reward", step_metric="hand")
    wandb.define_metric("actor_loss", step_metric="hand")
    wandb.define_metric("critic_loss", step_metric="hand")
    config = wandb.config

    # Creating aggregator with the specified method
    aggregator = FRLAggregator(robust=(args.aggregation != "fedavg"), agg_method=config.aggregation_method)

    # Creating FRL Agents based on number of players
    agents = []
    for i in range(args.num_players):
        agent = FRLAgent(player_id=i, variant=f"kuhn_{args.num_players}p")
        agent.model.to(device)
        agents.append(agent)
    
    # For backward compatibility with earlier code
    agent0 = agents[0]
    agent1 = agents[1]
    agent2 = agents[2] if args.num_players == 3 else None

    # Initial chip values to restore when resetting
    INITIAL_CHIPS = 100
    
    # Setting up KuhnPokerEngine with the FRL Agents
    log_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "game_logs.txt")
    
    if args.num_players == 2:
        engine = KuhnPokerEngine(
            player0=agent0,
            player1=agent1,
            delay=0.0,
            num_players=2,
            auto_rounds=0,
        )
    else:  # 3 players
        engine = KuhnPokerEngine(
            player0=agent0,
            player1=agent1,
            player2=agent2,
            delay=0.0,
            num_players=3,
            auto_rounds=0,
        )

    best_reward = float("-inf")
    best_params = None
    
    # Track best parameters for each individual agent
    best_agent_params = {0: None, 1: None, 2: None}
    best_agent_rewards = {0: float("-inf"), 1: float("-inf"), 2: float("-inf")}
    
    # Lists to store metrics for plotting
    all_hands = []
    all_rewards = []
    all_actor_losses = []
    all_critic_losses = []
    avg_reward_agent0 = []
    avg_reward_agent1 = []
    avg_reward_agent2 = []
    avg_reward_central = []

    # To track cumulative rewards
    cum_reward_agent0 = [0]
    cum_reward_agent1 = [0]
    cum_reward_agent2 = [0]
    cum_reward_central = [0]

    # To track total hands played across all rounds
    global_hand_count = 0

    # Saving the original log method
    original_log_method = engine.log
    
    # Logging the training session start timestamp to the game log file
    original_log_method(timestamp_header)
    
    # To track game statistics for dynamic normalization
    game_stats = {
        "max_reward": 0,
        "min_reward": 0,
        "reward_history": [],
        "player_stats": {i: {"bluff_frequency": 0, "fold_frequency": 0, "win_frequency": 0} for i in range(3)},
        "initial_chips": INITIAL_CHIPS
    }

    # Setting up exploration rate decay
    initial_epsilon = 0.3  # Initial exploration rate
    min_epsilon = 0.05     # Minimum exploration rate
    epsilon_decay = 0.8    # Decay factor per round

    # Training Loop
    for rd in range(config.rounds):
        round_num = rd + 1
        
        original_log_method(f"\n{'='*80}")
        original_log_method(f"{'='*30} TRAINING ROUND {round_num}/{config.rounds} {'='*30}")
        original_log_method(f"{'='*80}")
        
        # Reset chips if specified
        if args.reset_chips:
            engine.chips = [INITIAL_CHIPS] * engine.num_players
            original_log_method(f"Chips reset to {INITIAL_CHIPS} for all players at the start of round {round_num}")
        
        # Display current chip counts at the start of the round
        chips_str = ", ".join([f"Player {i}: {c}" for i, c in enumerate(engine.chips)])
        original_log_method(f"Starting round {round_num} with chip counts: {chips_str}")
        
        round_rewards = []
        # one empty dict per agent so indexing is safe
        metrics_list = [{} for _ in agents]
        updated_params = None # Placeholder for updated parameters
        
        # Adjust exploration rate for this round
        current_epsilon = max(
            args.min_exploration, 
            args.initial_exploration * (0.999 ** (global_hand_count))  # Slower decay
        )
        print(f"Setting exploration rate to {current_epsilon:.4f} for round {round_num}")
        for agent in agents:
            agent.epsilon = current_epsilon
        wandb.log({"exploration_rate": current_epsilon})
        
        # Multiple hands within this round
        for hand in range(config.hands_per_round):
            hand_num = hand + 1
            global_hand_count += 1
            
            original_log_method(f"\n{'-'*70}")
            original_log_method(f"{'-'*25} Round {round_num} / Hand {hand_num} {'-'*25}")
            original_log_method(f"{'-'*70}")
            
            # Setting the hand number explicitly to ensure correct tracking
            engine.current_hand = global_hand_count
            
            # Creating a custom logging function that adds prefixes
            # (still uses original_log_method)
            def custom_log(msg):
                prefix = f"[Round {round_num}] [Hand {hand_num}] "
                original_log_method(f"{prefix}{msg}")
            
            engine.log = custom_log
            
            # Before running the hand, storing chip counts
            chips_before = engine.chips.copy()
            
            # Running a single hand
            reward_info = engine.run_round()

            # Adding this block to guarantee agents have experiences to learn from
            if reward_info:  # If the hand completed with rewards
                for pid, rew in reward_info.items():
                    # Creating minimal state representation for learning
                    stub_state = [0.0] * agents[pid].state_dim
                    # Storing terminal transition with the hand's reward
                    agents[pid].model.remember(stub_state, 0, rew, stub_state, True)
            
            if reward_info is None:
                reward_info = {}
            
            # Computing rewards based on chip differences
            shaped_rewards = {}
            for player_id in range(len(engine.chips)):
                chips_after = engine.chips[player_id]
                chips_before_player = chips_before[player_id]
                raw_reward = chips_after - chips_before_player
                shaped_reward = reward_shaping(
                    raw_reward, 
                    chips_before_player, 
                    chips_after, 
                    INITIAL_CHIPS,
                    config.training_mode
                )
                shaped_rewards[player_id] = shaped_reward

                # Updating game statistics
                game_stats["reward_history"].append(shaped_reward)
                game_stats["max_reward"] = max(game_stats["max_reward"], shaped_reward)
                game_stats["min_reward"] = min(game_stats["min_reward"], shaped_reward)

                # Tracking player statistics for reputation system
                if raw_reward > 0:
                    game_stats["player_stats"][player_id]["win_frequency"] += 1

                # Logging the raw vs. shaped reward for transparency
                original_log_method(f"Player {player_id}: Raw reward {raw_reward} → Shaped reward {shaped_reward:.2f}")
            
            # To guarantee memory population after each hand with the hand's reward
            for pid, agent in enumerate(agents):
                if pid in shaped_rewards:
                    # Creating minimal state representation
                    stub_state = [0.0] * agent.model.state_dim
                    
                    # Storing a transition with the hand's reward
                    # Using a dummy action (0) because currently interested in the reward signal
                    agent.model.remember(stub_state, 0, shaped_rewards[pid], stub_state, True)
                    
                    print(f"[TRAIN] Added reward transition {shaped_rewards[pid]} to agent {pid}'s memory")
            
            # Using shaped rewards for learning
            hand_reward = sum(shaped_rewards.values())
            round_rewards.append(hand_reward)
            
            # Creating a round-only logger for between hands
            def round_logger(msg):
                prefix = f"[Round {round_num}] "
                original_log_method(f"{prefix}{msg}")
            
            # Switching to round-only logger
            engine.log = round_logger
            
            original_log_method(f"Round {round_num} / Hand {hand_num} completed with reward: {hand_reward}")
            
            # Printing current chip counts for debugging
            chips_str = ", ".join([f"Player {i}: {c}" for i, c in enumerate(engine.chips)])
            original_log_method(f"Chip counts after Round {round_num} / Hand {hand_num}: {chips_str}")

        # Restoring the original log method
        engine.log = original_log_method
        
        # After all hands in the round:
        original_log_method(f"\n{'-'*70}")
        original_log_method(f"{'-'*25} Round {round_num} Summary {'-'*25}")
        original_log_method(f"{'-'*70}")

        # Calculating adjusted learning rate based on round
        adjusted_lr = config.learning_rate * (0.99 ** rd)  # Learning rate decay
                
        # Gathering training metrics from each agent with adjusted learning rate
        wandb.log({"learning_rate": adjusted_lr})

        # Each agent performs local training and obtains gradients/metrics
        metrics_list = []
        for idx, agent in enumerate(agents):
            m = agent.train_local()
            metrics_list.append(m)
            if m:
                wandb.log({f"agent{idx}_{k}": v for k, v in m.items()},
                        step=global_hand_count)

        grad_list = []
        for ag in agents:
            grads = ag.get_gradients()
            if grads:  # If there was experience to learn from
                grad_list.append(grads)

        # Aggregating updates if any grads exist
        if grad_list:
            print(f"Aggregating gradients from {len(grad_list)} agents")
            agg_grads = aggregator.aggregate_gradients(grad_list)
            if agg_grads:
                print(f"Successfully aggregated gradients with method: {config.aggregation_method}")
                # Apply to all agents
                for ag in agents:
                    ag.model.apply_gradients(agg_grads)
                updated_params = agent0.get_model_params()
                print(f"Updated parameters distributed to all agents")
                
                # Only clear memories AFTER training is complete and gradients are applied
                for ag in agents:
                    ag.model.memory = []
        else:
            print(f"No gradients available for aggregation in round {round_num}")
            # If no gradients, but we have agents with params, still use them
            if updated_params is None and best_params is not None:
                updated_params = best_params
                print(f"Using previous best parameters instead")

        # After all hands in the round:
        # Calculating average reward across all hands in this round
        avg_reward = sum(round_rewards) / len(round_rewards) if round_rewards else 0
        original_log_method(f"Round {round_num} completed with average reward: {avg_reward}")
        
        # Final chip counts for the round
        chips_str = ", ".join([f"Player {i}: {c}" for i, c in enumerate(engine.chips)])
        original_log_method(f"Final chip counts after round {round_num}: {chips_str}")
        
        # Storing metrics for plotting using hands instead of rounds
        all_hands.append(global_hand_count)
        all_rewards.append(avg_reward)
        avg_reward_agent0.append(metrics_list[0].get("avg_reward", 0))
        avg_reward_agent1.append(metrics_list[1].get("avg_reward", 0))
        # Only append agent2 data if we have 3 players
        if args.num_players > 2:
            avg_reward_agent2.append(metrics_list[2].get("avg_reward", 0))
        else:
            avg_reward_agent2.append(0)  # Add placeholder value for 2-player games
        avg_reward_central.append(avg_reward)

        # Calculating and tracking cumulative rewards
        cum_reward_agent0.append(cum_reward_agent0[-1] + shaped_rewards.get(0, 0))
        cum_reward_agent1.append(cum_reward_agent1[-1] + shaped_rewards.get(1, 0))
        # Only update agent2 cumulative rewards if we have 3 players
        if args.num_players > 2:
            cum_reward_agent2.append(cum_reward_agent2[-1] + shaped_rewards.get(2, 0))
        else:
            cum_reward_agent2.append(cum_reward_agent2[-1])  # Just carry over previous value
        cum_reward_central.append(cum_reward_central[-1] + avg_reward)

        # Creating a wandb plot for cumulative rewards
        if args.num_players > 2:
            # 3-player version
            cum_reward_data = [[x, cum_reward_agent0[i+1], cum_reward_agent1[i+1], 
                               cum_reward_agent2[i+1], cum_reward_central[i+1]] 
                              for i, x in enumerate(all_hands)]
            cum_reward_columns = ["hand", "agent0_cum", "agent1_cum", "agent2_cum", "central_cum"]
            cum_reward_series = ["agent0_cum", "agent1_cum", "agent2_cum", "central_cum"]
        else:
            # 2-player version
            cum_reward_data = [[x, cum_reward_agent0[i+1], cum_reward_agent1[i+1], 
                               cum_reward_central[i+1]] 
                              for i, x in enumerate(all_hands)]
            cum_reward_columns = ["hand", "agent0_cum", "agent1_cum", "central_cum"]
            cum_reward_series = ["agent0_cum", "agent1_cum", "central_cum"]
            
        cum_reward_table = wandb.Table(data=cum_reward_data, columns=cum_reward_columns)
        
        wandb_log_data = {
            "hand": global_hand_count,
            "cumulative_reward_plot": wandb.plot.line(
                cum_reward_table,
                "hand",
                cum_reward_series,
                title="Cumulative Rewards Over Time"
            ),
            "cum_reward_agent0": cum_reward_agent0[-1],
            "cum_reward_agent1": cum_reward_agent1[-1],
            "cum_reward_central": cum_reward_central[-1]
        }
        
        if args.num_players > 2:
            wandb_log_data["cum_reward_agent2"] = cum_reward_agent2[-1]
            
        wandb.log(wandb_log_data)

        # Creating another wandb plot for rewards using Table format
        reward_data = [[x, y] for (x, y) in zip(all_hands, all_rewards)]
        reward_table = wandb.Table(data=reward_data, columns=["hand", "reward"])
        wandb.log({
            "hand": global_hand_count,
            "avg_reward": avg_reward,
            "reward_plot": wandb.plot.line(
                reward_table,
                "hand", 
                "reward",
                title="Average Reward per Hand"
            )
        })

        # Logging the raw rewards for transparency and learning
        raw_reward_data = {
            "raw_avg_reward_agent0": avg_reward_agent0[-1],
            "raw_avg_reward_agent1": avg_reward_agent1[-1],
            "raw_avg_reward_central": avg_reward_central[-1]
        }
        
        if args.num_players > 2:
            raw_reward_data["raw_avg_reward_agent2"] = avg_reward_agent2[-1]
            
        wandb.log(raw_reward_data)

        # More W&B logging
        # Collecting average losses across all agents
        actor_losses = [m.get("actor_loss", 0.0) for m in metrics_list if m]
        critic_losses = [m.get("critic_loss", 0.0) for m in metrics_list if m]
        if actor_losses:
            avg_actor_loss = sum(actor_losses)/len(actor_losses)
            avg_critic_loss = sum(critic_losses)/len(critic_losses)
            
            all_actor_losses.append(avg_actor_loss)
            all_critic_losses.append(avg_critic_loss)
            
            # Creating a wandb plot for losses using Table format
            loss_data = [[x, y1, y2] for (x, y1, y2) in zip(all_hands, all_actor_losses, all_critic_losses)]
            loss_table = wandb.Table(data=loss_data, columns=["hand", "actor_loss", "critic_loss"])
            wandb.log({
                "hand": global_hand_count,
                "actor_loss": avg_actor_loss,
                "critic_loss": avg_critic_loss,
                "loss_plot": wandb.plot.line(
                    loss_table,
                    "hand",
                    ["actor_loss", "critic_loss"],
                    title="Training Losses"
                )
            })

        # Validation pass
        val_loss, val_metrics = validate(agents, engine)
        wandb.log({"validation_loss": val_loss})
        for i, metrics in val_metrics.items():
            wandb.log({f"val_win_rate_{i}": metrics["win_rate"], 
                       f"val_avg_reward_{i}": metrics["avg_reward"]})

        # Tracking individual agent rewards (using shaped rewards)
        for i, agent in enumerate(agents):
            agent_reward = shaped_rewards.get(i, 0)
            wandb.log({f"agent{i}_reward": agent_reward})
            wandb.log({f"agent{i}_raw_reward": reward_info.get(i, 0)})
            
            # Saving best model for each individual agent
            if agent_reward > best_agent_rewards[i]:
                best_agent_rewards[i] = agent_reward
                best_agent_params[i] = agent.get_model_params()
                
                try:
                    # Creating directory if it doesn't exist
                    models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
                    os.makedirs(models_dir, exist_ok=True)
                    
                    save_path = os.path.join(models_dir, f"best_frl_agent{i}.pt")
                    agent.save_model(save_path)
                    print(f"New best model for agent{i} saved with reward {agent_reward} at round {round_num}")
                    
                    # Logging model artifact to wandb
                    wandb.save(save_path)
                    wandb.log({f"best_reward_agent{i}": agent_reward})
                except Exception as e:
                    print(f"Error saving agent{i} model: {e}")

        # Tracking action distributions for each agent
        action_distributions = {}
        aggression_ratios = {}
        for i, agent in enumerate(agents):  # Using agents list instead of hard-coded indices
            # Extracting action counts from agent's memory if available
            actions_taken = [t['action'] for t in agent.local_transitions[-config.hands_per_round:] if 'action' in t]
            
            if actions_taken:
                action_counts = {
                    'check': actions_taken.count(0),
                    'bet': actions_taken.count(1),
                    'call': actions_taken.count(2),
                    'fold': actions_taken.count(3),
                    'raise': actions_taken.count(4)
                }
                
                total_actions = sum(action_counts.values())
                if total_actions > 0:
                    action_distribution = {k: v/total_actions for k, v in action_counts.items()}
                    action_distributions[i] = action_distribution
                    
                    # Calculating overall aggression ratio (bet+raise vs check+fold)
                    aggressive_actions = action_counts['bet'] + action_counts['raise']
                    passive_actions = action_counts['check'] + action_counts['fold']
                    aggression_ratios[i] = aggressive_actions / max(1, aggressive_actions + passive_actions)

        # Average actor/critic losses across all agents
        actor_losses  = [m["actor_loss"]  for m in metrics_list if "actor_loss"  in m]
        critic_losses = [m["critic_loss"] for m in metrics_list if "critic_loss" in m]
        avg_actor_loss  = sum(actor_losses)  / len(actor_losses)  if actor_losses  else 0.0
        avg_critic_loss = sum(critic_losses) / len(critic_losses) if critic_losses else 0.0

        # Collecting everything in one dict
        metrics = {
            # rewards
            "avg_reward": avg_reward,
        }
        
        # Adding individual agent metrics
        agent_reward_lists = {
            0: avg_reward_agent0,
            1: avg_reward_agent1,
            2: avg_reward_agent2 if args.num_players == 3 else []
        }
        
        for i, agent in enumerate(agents):
            if i in agent_reward_lists and len(agent_reward_lists[i]) > 0:
                metrics[f"raw_avg_reward_agent{i}"] = agent_reward_lists[i][-1]
            else:
                metrics[f"raw_avg_reward_agent{i}"] = 0.0
        
        metrics["raw_avg_reward_central"] = avg_reward_central[-1]
        
        # Losses
        metrics["actor_loss"] = avg_actor_loss
        metrics["critic_loss"] = avg_critic_loss
        
        # Per-agent losses
        for i, m in enumerate(metrics_list):
            if i < len(agents):
                metrics[f"agent{i}_actor_loss"] = m.get("actor_loss", 0.0)
                metrics[f"agent{i}_critic_loss"] = m.get("critic_loss", 0.0)
        
        # validation
        metrics["validation_loss"] = val_loss

        # Adding hand field for correct step metric logging
        metrics["hand"] = global_hand_count
        # Finally logging once using hand counter as step metric
        wandb.log(metrics)

        # Saving best global model
        if avg_reward > best_reward and updated_params is not None:
            best_reward = avg_reward
            best_params = updated_params
            
            # Immediately saving the current best model after each improvement
            try:
                # Creating directory if it doesn't exist
                models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
                os.makedirs(models_dir, exist_ok=True)
                
                # Applying best parameters and save global model
                agent0.set_model_params(best_params)
                save_path = os.path.join(models_dir, "best_frl_global.pt")
                agent0.save_model(save_path)
                print(f"New best global model saved with reward {best_reward} at round {round_num}")
                
                # Logging model artifact to wandb
                wandb.save(save_path)
                wandb.log({"best_reward": best_reward})
            except Exception as e:
                print(f"Error saving global model: {e}")
    
    # Final model saves
    models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
    os.makedirs(models_dir, exist_ok=True)
    
    # Saving final versions of all agents
    for i, agent in enumerate(agents):
        try:
            final_save_path = os.path.join(models_dir, f"final_frl_agent{i}.pt")
            agent.save_model(final_save_path)
            print(f"Final model for agent{i} saved")
            wandb.save(final_save_path)
        except Exception as e:
            print(f"Error saving final model for agent{i}: {e}")
    
    # Saving final global model if we have best parameters
    if best_params is not None:
        try:
            # Applying the best parameters we found to agent0
            agent0.set_model_params(best_params)
            save_path = os.path.join(models_dir, "best_frl_global.pt")
            agent0.save_model(save_path)
            print(f"Best global model saved with reward {best_reward}")
            
            # Saving the final model as a wandb artifact
            wandb.save(save_path)
        except Exception as e:
            print(f"Error saving final global model: {e}")
    else:
        # If somehow we never found best params, just saving the current agent0
        try:
            save_path = os.path.join(models_dir, "final_frl_global.pt")
            agent0.save_model(save_path)
            print(f"Saved current agent0 model as final since no best parameters were found")
            wandb.save(save_path)
        except Exception as e:
            print(f"Error saving fallback global model: {e}")
    
    completion_timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    timestamp_footer = f"\n{'='*80}\n{'='*25} TRAINING SESSION COMPLETED AT {completion_timestamp} {'='*25}\n{'='*80}"
    original_log_method(timestamp_footer)
    
    print("Training finished.")
    wandb.finish()

if __name__ == "__main__":
    main()