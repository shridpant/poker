# Federated Reinforcement Learning for 3-Player Kuhn Poker
#### aka जुवाडे

This document provides a detailed overview of how to implement and train a Federated Reinforcement Learning (FRL) agent (alongside other agents) in a 3-player Kuhn Poker environment. The goal is to develop a **plug-and-play** framework where any RL model can be inserted as a player, and specifically highlight an FRL-based model that can be configured to train in a distributed manner.

## Table of Contents
1. [Overview](#overview)  
2. [Prerequisites](#prerequisites)  
3. [Repository Structure](#repository-structure)  
4. [Core Components](#core-components)  
5. [Federated RL Pipeline](#federated-rl-pipeline)  
6. [Setting Up the Environment](#setting-up-the-environment)  
7. [Logging with Weights & Biases](#logging-with-weights--biases)  
8. [Training Steps](#training-steps)  
9. [Federated Learning Details](#federated-learning-details)  
10. [Evaluation and Testing](#evaluation-and-testing)  
11. [Troubleshooting and Tips](#troubleshooting-and-tips)  
12. [References](#references)

---

## 1. Overview
Our aim is twofold:
1. **RL Framework (Plug-and-Play)**: Allow any RL model to serve as a player in our 3-player Kuhn Poker engine without major code refactoring.  
2. **Dedicated FRL Model**: Train a strong FRL-based agent that can later be tested against other strong players.

You will find guidelines for:
- Integrating your own PyTorch-based model
- Using an FRL aggregator (e.g., [FedAvg](https://arxiv.org/abs/1602.05629), robust aggregators, etc.)
- Logging everything (high-level metrics + detailed actions) with [Weights & Biases](https://wandb.ai/)

---

## 2. Prerequisites
- **Python 3.9** (Mac/Linux)
- **PyTorch** (v1.9+ recommended for GPU support)
- **Weights & Biases (wandb)** for logging
- [Optional but recommended] Familiarity with RL libraries or frameworks that handle distributed/federated setups, such as [Flower](https://flower.dev/), [Ray RLlib](https://www.ray.io/rllib), or custom solutions.

**Required Python Packages** (minimally):
```bash
pip install torch wandb numpy
```

If you plan to use a more advanced FRL library (like Flower or Ray), install those as well:
```bash
pip install flwr  # or ray[rllib]
```

---

## 3. Repository Structure
Below is the recommended folder layout. Some files might already exist in your repository.

```
poker
├── engine
│   ├── KuhnPokerEngine.py
│   ├── utilities.py
│   └── ...
├── models
│   ├── README.md              ← (this file)
│   ├── frl_base.py            ← base FRL agent class
│   ├── aggregator.py          ← aggregator/federated logic
│   ├── model_arch.py          ← neural network definitions
│   └── ...
├── scripts
│   ├── train.py               ← launching distributed (federated) training
│   ├── evaluate.py            ← script for evaluating trained agents
│   └── ...
├── data
│   └── (optional) store any pre-collected transitions or logs here
└── requirements.txt
```

You can adapt this structure as needed, but keep the **models** folder for your FRL code, the **engine** folder for poker logic, and **scripts** for running training/evaluation.

---

## 4. Core Components
1. **Kuhn Poker Engine** (`engine/KuhnPokerEngine.py`):  
   - A 3-player version that manages the deck, betting, and pot calculations.
   - Exposes a method like `play_hand()` or `step()` for multi-agent turns.

2. **FRL Base Class** (`models/frl_base.py`):  
   - Defines a common interface for RL/federated agents (initialization, action selection, local updates).
   - Handles saving/retrieving model parameters for aggregation.

3. **Aggregator** (`models/aggregator.py`):  
   - Defines how local models or gradients are combined (e.g., FedAvg, median).
   - Optionally contains logic for robust or Byzantine-tolerant updates.

4. **Model Architecture** (`models/model_arch.py`):  
   - One or more neural network definitions (actor-critic, Q-learning, or your own design).
   - Written in PyTorch to easily integrate with GPU backends on cloud VMs.

5. **Scripts**  
   - **train.py**: Orchestrates the entire training pipeline (from environment setup to federated rounds).  
   - **evaluate.py**: Loads a trained agent and tests it against other strong agents or built-in bots.

---

## 5. Federated RL Pipeline
At a high level, the pipeline for FRL in a 3-player Kuhn Poker scenario involves:

1. **Initialize Agents**  
   - Each of the 3 players is an RL-based agent (potentially the same underlying model) with local parameters.

2. **Self-Play and Data Collection**  
   - Agents interact with the environment by playing multiple hands.
   - Each agent captures transitions: (state, action, reward, next_state, done).
   - Rewards may be normalized based on variant or pot-size (as suggested in the paper).

3. **Local Updates**  
   - Each agent locally updates its parameters using its collected transitions (policy gradient, Q-learning, etc.).
   - No raw transitions are shared among agents, preserving privacy.

4. **Parameter Aggregation**  
   - Agents send model parameters or gradients to an aggregator.
   - The aggregator combines them (e.g., with FedAvg) to produce a global model.
   - The global model is then broadcast back to the agents.

5. **Repeat**  
   - The updated global model is used for subsequent self-play rounds.
   - Over multiple rounds, the agents converge to a stronger, more generalized policy.

---

## 6. Setting Up the Environment
1. **Install Dependencies**  
   - Ensure PyTorch and wandb are installed, as well as any FRL library if desired.

2. **Configure W&B**  
   - Run `wandb login` from your terminal (or set the `WANDB_API_KEY`) to enable logging.

3. **Adjust KuhnPokerEngine**  
   - Make sure it can handle 3-player action loops.
   - Expose a function or interface for retrieving transitions so each RL agent can store them.

4. **Refactoring**  
   - You may need to adapt the engine so it can accept a “player” with a method like `get_action(state)`.

---

## 7. Logging with Weights & Biases
1. **Initialization**  
   - In your `train.py`, call:
     ```python
     import wandb
     wandb.init(project="kuhn-poker-frl", config={...})
     ```

2. **Recording Metrics**  
   - After each training step or episode, log relevant data:
     ```python
     wandb.log({
       "episode_reward": episode_reward,
       "win_rate": current_win_rate,
       "loss": current_loss
       # Add more stats as needed
     })
     ```

3. **Detailed Logs**  
   - For debugging or offline analysis, you can store step-by-step transitions in a separate structure and optionally log them to W&B as tables.

---

## 8. Training Steps
Below is an example workflow; modify as needed:

1. **Launch Training**  
   - `python3 scripts/train.py --rounds 5 --hands_per_round 3 --aggregation fedavg --reset_chips`
   - This script might:
     1. Create 3 RL agents.  
     2. Loop for `rounds`:  
        - Each agent self-plays for `hands_per_round` in the environment.  
        - The agent collects data and does local training.  
        - Once all agents finish, aggregator merges parameters into a global model.  
        - Global model is distributed back.  
     3. Log each agent’s performance to W&B.

2. **Local Training**  
   - Each agent calls something like `agent.update_local_model()` after gathering a batch of transitions.
   - If you want to incorporate advanced FRL frameworks, the aggregator might run on a separate server process.

3. **Saving Models**  
   - Periodically save checkpoint files for each agent or the aggregated global model:
     ```python
     torch.save(agent.model.state_dict(), f"agent_{agent_id}_round_{round_idx}.pt")
     ```

---

## 9. Federated Learning Details
- **Aggregator** (`models/aggregator.py`):  
  Implement your aggregator methods. For example:
  - `FedAvg`: Average the parameters across agents  
  - `Median`: Coordinate-wise median to defend against outliers  
  - `Trimmed Mean`: Drop highest and lowest parameter values, then average  

- **Communication**  
  - If you want a realistic setup, each agent runs in a separate process and only shares parameters with the aggregator. For local simulations, you can run them sequentially.

- **Privacy Enhancements**  
  - You can incorporate secure aggregation or differentially private updates as described in the paper.

---

## 10. Evaluation and Testing
After training completes, you can use `scripts/evaluate.py` to:
1. Load the final global model.  
2. Spawn the FRL agent in the environment with two other strong or random agents.  
3. Measure average reward, win rate, or exploitability over many hands.  
4. Log results to W&B or a CSV file.

Example CLI usage:
```bash
python scripts/evaluate.py --agent_checkpoint agent_global.pt --num_hands 10000
```

---

## 11. Troubleshooting and Tips
1. **Convergence Issues**  
   - Try smaller learning rates or different optimizers (Adam vs. RMSProp).
   - Increase the number of hands or rounds for more experience.
2. **Partial Observability**  
   - Ensure your policy accounts for hidden cards and uses actor-critic methods suited for imperfect information.
3. **Scaling Up**  
   - If you move beyond Kuhn Poker to HUNL or Omaha, watch out for larger state spaces and needed compute resources.
4. **Logging Overhead**  
   - If logging every step is slow, reduce logging frequency or log only high-level metrics.

---

## 12. References
- **Paper Citations**:  
  - [AlphaHoldem: End-to-end Reinforcement Learning for Poker](#)  
  - [Byzantine-Robust Aggregation in Federated Learning](#)  
  - [Ray RLlib for distributed RL](https://docs.ray.io/en/latest/rllib.html)  
  - [Flower for Federated Learning](https://flower.dev/)

- **Implementation Examples**:  
  - [PyTorch Official Tutorials](https://pytorch.org/tutorials/)  
  - [W&B Documentation](https://docs.wandb.ai/)

---

## Next Steps
1. **Implement** the FRL base class and aggregator in `models/`.  
2. **Refactor** the KuhnPokerEngine (if needed) to seamlessly collect transitions.  
3. **Create** your training script (`scripts/train.py`) that orchestrates local training, aggregation, and logging.  
4. **Run** a few local tests to confirm everything.  
5. **Deploy** on a GPU VM for extended training rounds, leveraging Weights & Biases for monitoring.

## Running the Code
Below are some quick tips on how to run training and evaluation:

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Train the FRL agents (example command):
   ```bash
   python scripts/train.py --rounds 10 --hands_per_round 500 --aggregation fedavg
   ```
   This will:
   - Initialize a Kuhn Poker environment with 3 FRL agents.
   - Run multiple rounds of self-play, each with a configurable number of hands.
   - Perform local training on each agent’s collected data.
   - Aggregate model parameters using your chosen aggregator.

3. Evaluate or test agents:
   - Adapt or create an `evaluate.py` script to load trained models and measure win rates over many hands.
   - You might run:
     ```bash
     python scripts/evaluate.py --agent_checkpoint agent_global.pt
     ```
   - This can match the trained FRL agent against other bots or strategies.