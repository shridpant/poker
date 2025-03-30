# A Reinforcement Learning Framework for Poker

This repository provides a flexible implementation of Kuhn Poker (extendable to other poker variations) with support for different player agent types, including human players, random agents, and federated learning agents. The project is designed as a plug-and-play system to easily implement and test various ML algorithms in a simplified poker environment.

## Game Rules

### Standard Kuhn Poker (2 Players)
Kuhn poker is a simple poker variant played with just 3 cards: Jack (J), Queen (Q), and King (K).

1. **Setup**: 
   - Each player antes 1 chip to the pot and receives 1 card
   - Cards are ranked from lowest to highest: J < Q < K

2. **Gameplay**:
   - First betting round: Players can check or bet 1 chip
   - If there's a bet, the other player can call or fold
   - If both players check, cards are revealed (showdown)
   - If a player bets and the other calls, there's another betting round

3. **Winner**: The highest card wins the pot. If a player folds, the other player wins automatically.

### Modified Kuhn Poker (3 Players)
Our 3-player variant uses a modified deck to create more strategic depth:

1. **Setup**:
   - Uses 4 cards: Jack (J), Queen (Q), King (K), and Ace (A)
   - 3 cards are dealt to players, and 1 card remains hidden
   - This hidden card creates uncertainty - even the player with the King doesn't know if the Ace is in play

2. **Gameplay**:
   - Same basic mechanics as 2-player, but with 3 players taking turns
   - More complex betting dynamics due to multiple players

3. **Winner**: Highest card wins, or last remaining player if others fold.

## File Structure

```
FRL_Poker/
├── game_engine.py       # Core game mechanics
├── utilities.py         # Logging and utility functions 
├── players/
│   ├── base.py          # Abstract base Player class
│   ├── human_agent.py   # Human player implementation
│   ├── random_agent.py  # Random action agent 
│   └── federated_agent.py # Basic federated learning agent
├── logs/
│   ├── game_log.txt     # Game logs
│   └── game_data/       # Data files for ML training
│       ├── rl_data.csv  # Reinforcement learning data
│       └── federated_player_*_data.csv # Player-specific data
```

## Installation & Requirements

1. Clone the repository:
   ```
   git clone https://github.com/shridpant/poker
   cd Poker
   ```

2. Install dependencies:
   ```
   pip install pyspiel
   ```

## Running the Game

There's an ```example.ipynb``` for you. I tried to make it super intuitive, but a call would probably be easier to get it all sorted out!

### 2-Player Game (Human vs Random)

```python
from game_engine import KuhnPokerEngine
from players.human_agent import HumanPlayer
from players.random_agent import RandomPlayer

player0 = HumanPlayer()
player1 = RandomPlayer()

engine = KuhnPokerEngine(
    player0=player0,
    player1=player1,
    delay=0.0,  # Set delay to 0 when human players are involved
    num_players=2,
    auto_rounds=None  # None to ask for next round after each hand
)

engine.run_game()
```

### 3-Player Game (Human vs Random vs Federated)

```python
from game_engine import KuhnPokerEngine
from players.human_agent import HumanPlayer
from players.random_agent import RandomPlayer
from players.federated_agent import FederatedPlayer

player0 = HumanPlayer()
player1 = RandomPlayer()
player2 = FederatedPlayer(player_id=1)

engine = KuhnPokerEngine(
    player0=player0,
    player1=player1,
    player2=player2,
    delay=0.0,
    num_players=3,
    auto_rounds=None
)

engine.run_game()
```

## Adding Custom Agents

The framework allows easy implementation of custom agents, including reinforcement learning and federated learning approaches.

### Basic Agent Structure

Create a new file in the `players/` directory and implement the required interface:

```python
from players.base import Player

class MyCustomAgent(Player):
    def __init__(self):
        # Initialize your agent
        pass
        
    def get_action(self, card, available_actions, round_num, chips_remaining):
        # Your decision-making logic here
        # Return action_idx or (action_idx, raise_amount) tuple
        return chosen_action_idx
```

### Adding an RL Agent

```python
class MyRLAgent(Player):
    def __init__(self):
        # Define model, optimizer, etc.
        self.model = self.create_model()  # Your model definition
        self.memory = []  # Experience replay buffer
        
    def create_model(self):
        # Define your neural network or other ML model
        pass
        
    def get_action(self, card, available_actions, round_num, chips_remaining):
        # Convert state to your model's input format
        state_tensor = self.preprocess_state(card, available_actions, round_num)
        # Use model for prediction
        q_values = self.model(state_tensor)
        # Select best valid action
        return self.select_action(q_values, available_actions)
        
    def train(self):
        # Training loop using collected experiences
        pass
```

## Data Collection for Machine Learning

The system automatically collects game data suitable for machine learning:

1. **RL Data**: Transitions for reinforcement learning are stored in `logs/game_data/rl_data.csv` with state, action, reward, and next state information.

2. **Federated Data**: Each federated agent stores its own local data in `logs/game_data/federated_player_{id}_data.csv`.

The state representation includes:
- Round number
- Game stage (first/second betting round)
- Current player
- Pot size
- Player's card
- Available actions
- Betting history
- Chip counts

This framework collects gameplay data in CSV format to support training reinforcement learning (RL) or federated reinforcement learning (FRL) agents. The data is stored in files such as `rl_data.csv` and `federated_player_X_data.csv`.

### RL Data Format (`rl_data.csv`)

The `rl_data.csv` file contains detailed information about each decision made during the game. Below is a description of each column:

| Column Name      | Description                                                                 |
|------------------|-----------------------------------------------------------------------------|
| `session_id`     | Unique identifier for the game session (e.g., timestamp).                  |
| `round`          | The current round number within the session.                               |
| `decision_index` | The index of the decision within the round (e.g., 0 for the first decision).|
| `stage`          | The stage of the round (`first` or `second` betting round).                |
| `current_player` | The player making the decision at this step (e.g., 0, 1, or 2).            |
| `state`          | Serialized representation of the game state at the time of the decision.   |
| `legal_actions`  | List of valid actions available to the player at this step.                |
| `chosen_action`  | The action chosen by the player (encoded as an integer).                   |
| `reward`         | The reward received for this action (e.g., chips won or lost).             |
| `done`           | Boolean indicating whether the game or round has ended.                   |

#### Example Row
```csv
session_id,round,decision_index,stage,current_player,state,legal_actions,chosen_action,reward,done
20250329173312,1,0,first,0,"{'round': 1, 'stage': 'first', 'current_player': 0, 'player0_card': 1, 'player1_card': 2, 'player2_card': -1, 'pot': 2, 'chips': '9;9', 'betting_history': ''}","[0, 1]",0,0,False
```

## Known Issues & Troubleshooting

- **Action Display Issue**: Sometimes, for the human player, the available actions are not immediately shown. This can be remedied by pressing the "ESC" key and then entering the actual action.
  
- **Module Import Errors**: If you encounter import errors, make sure your working directory is set correctly.
