# Federated Reinforcement Learning (FRL) for Kuhn Poker

This document explains how to train the FRL agents and gives a brief overview of how they work.

## How It Works

FRL agents each maintain local models that learn from self-play data. After a set of training rounds, an aggregator combines the locally trained parameters into a shared global model using methods like median or FedAvg (see frl_aggregator.py).

## Training the FRL Agents

To train the FRL agents and collect data, you can run:

```bash
python3 scripts/train.py --rounds 100 --hands_per_round 10 --aggregation median --num_players 3 --reset_chips
```

This example:
- Creates a Kuhn Poker environment with three FRL agents.
- Plays multiple rounds of self-play, each containing a configurable number of hands.
- Performs local updates on each agent’s data.
- Aggregates parameters using the chosen aggregation method (in this case, “median”).

Adjust the arguments as needed to change the aggregation strategy, number of players, or training rounds.

### Command-Line Arguments

Below are some key options recognized by train.py:

• --rounds (int): Number of training rounds.  
• --hands_per_round (int): Number of hands to play in each round.  
• --aggregation (str): Aggregation method for parameters (fedavg, median, or trimmed_mean).  
• --reset_chips (flag): Resets each player's chips to 100 at the start of every round if specified.  
• --training_mode (str): Defines if agents train selfishly or cooperatively.  
• --reward_scaling (float): Scales the final chip difference for each hand.  
• --initial_exploration (float): Initial epsilon used for agent exploration.  
• --min_exploration (float): Minimum epsilon value that exploration decays toward.  
• --exploration_decay (float): Decay factor that reduces epsilon each round.  
• --learning_rate (float): Learning rate for agent model updates.  
• --experiment_name (str): Label for logging runs in Weights & Biases.  
• --device (str): Device to use (e.g., cpu, mps).  
• --num_players (2 or 3): Number of players in the Kuhn Poker game.  

Use them like:
```bash
python3 scripts/train.py --rounds 5 --hands_per_round 10 --aggregation fedavg --reset_chips
```
