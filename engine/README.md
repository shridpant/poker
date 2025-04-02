# KuhnPokerEngine Documentation

## Overview

`KuhnPokerEngine` is a flexible implementation of both traditional 2-player Kuhn Poker and a modified 3-player variant. This engine manages the entire game flow, from dealing cards to determining winners, while supporting various player agent types (human, random, RL-based, etc.).

## Table of Contents
- [Game Rules](#game-rules)
  - [Standard Kuhn Poker (2 Players)](#standard-kuhn-poker-2-players)
  - [Modified Kuhn Poker (3 Players)](#modified-kuhn-poker-3-players)
- [Creating Custom Agents](#creating-custom-agents)
  - [Reinforcement Learning Agents](#reinforcement-learning-agents)
- [Running the Game](#running-the-game)
- [Engine Architecture](#engine-architecture)
- [Betting Structure](#betting-structure)
- [Game Flow](#game-flow)
- [Code Deep Dive](#code-deep-dive)
  - [Execution Flow](#execution-flow)
  - [Betting Logic](#betting-logic)
  - [Showdown Implementation](#showdown-implementation)
  - [Edge Cases and Error Handling](#edge-cases-and-error-handling)
- [Data Collection for RL](#data-collection-for-rl)
  - [RL Specific](#rl-specific)

## Game Rules

### Standard Kuhn Poker (2 Players)
- **Deck**: Three cards: Jack (J), Queen (Q), and King (K)
- **Setup**: Each player antes 1 chip and receives 1 card
- **Card Ranking**: J < Q < K
- **Betting Rounds**:
  - First player can check or bet (1 chip)
  - If first player checks, second player can check or bet
  - If second player bets after a check, first player can call or fold
  - If first player bets, second player can call or fold
  - If both players check, game goes to showdown
- **Winning**: Highest card wins the pot or last remaining player if others fold

### Modified Kuhn Poker (3 Players)
- **Deck**: Four cards: Jack (J), Queen (Q), King (K), and Ace (A)
- **Setup**: Each player antes 1 chip and receives 1 card
- **Card Ranking**: J < Q < K < A
- **Key Difference**: One card remains hidden (face-down), adding uncertainty
  - Even players with K cannot be sure they have the highest card
- **Betting Rounds**:
  - Players act in clockwise order (starting with player 0)
  - First round: Players can check or bet (1 chip)
  - If everyone checks, game goes to showdown
  - If any player bets/raises, all active players get chance to respond
  - Second round: Only if first round had betting and at least 2 players remain
- **Winning**: Highest card wins among active players or last remaining player if others fold

## Creating Custom Agents

To create a custom agent, you need to implement a class with a `get_action()` method:

```python
class CustomPlayerAgent:
    def get_action(self, card, available_actions, round_num, chips, public_state):
        """
        Parameters:
        - card: Your current card (J, Q, K, or A)
        - available_actions: Dict of {action_idx: description} for legal actions
        - round_num: Current betting round (1 or 2)
        - chips: Your remaining chip count
        - public_state: Dict containing game state information
          - pot_size: Total chips in the pot
          - current_bets: List of each player's current bet
          - chip_counts: List of each player's chips
          - betting_history: List of previous actions
          - folded_players: List of boolean values indicating folded status
          - highest_bet: Current highest bet amount
          - last_bettor: Player ID who last bet/raised (-1 if none)
          - current_player: ID of player making the decision
          - player_id: Your player ID (same as current_player)
          - min_raise: Minimum raise amount

        Returns:
        - action_idx: Integer representing the action
          0: check, 1: bet, 2: call, 3: fold, 4: raise
        - raise_amount: Integer amount for raise (only used if action_idx is 4)
        """
        # Your decision logic here
        return action_idx, raise_amount  # (e.g., 0, None) for check
```

### Reinforcement Learning Agents

For specifically implementing RL-based agents, you can use state information and reward signals:

#### State Representation
Your agent can use these key elements from the environment (example):

```python
class RLAgent:
    def get_action(self, card, available_actions, round_num, chips, public_state):
        # Card value - one of J, Q, K, A (in 3-player game)
        card_value = card
        
        # Convert to numeric for easier processing (1=J, 2=Q, 3=K, 4=A)
        card_values = {"J": 1, "Q": 2, "K": 3, "A": 4}
        numeric_card = card_values[card]
        
        # Extract features for your RL state space
        features = [
            numeric_card / 4.0,  # Normalized card rank
            public_state['pot_size'] / 20.0,  # Normalized pot size
            chips / 10.0,  # Normalized chips remaining
            round_num / 2.0,  # Betting round (1 or 2)
            len(available_actions) / 5.0,  # Number of available actions (normalized)
            public_state['highest_bet'] / 10.0,  # Current highest bet
            public_state['player_id'] / 3.0,  # Player position (normalized)
            
            # Betting history features
            public_state['betting_history'].count('bet') / 5.0,
            public_state['betting_history'].count('raise') / 5.0,
            public_state['betting_history'].count('fold') / 5.0
        ]
        
        # Use your trained model to predict best action
        # action_idx = self.model.predict(features)
        
        # (example) For a basic RL agent, you might choose actions with epsilon-greedy:
        if random.random() < self.epsilon:  # Exploration
            action_idx = random.choice(list(available_actions.keys()))
        else:  # Exploitation
            action_idx = self.trained_model.predict(features)
            
        # For raising, determine a raise amount
        raise_amount = 1
        if action_idx == 4:  # raise action
            raise_amount = max(1, min(chips, self.determine_raise_amount(features)))
            
        return action_idx, raise_amount
        
    def update_model(self, transition):
        # Implement your learning algorithm (e.g., Q-learning, DQN)
        # The engine automatically collects transitions with:
        #   - state: Game state before action
        #   - action: Chosen action
        #   - reward: Chip difference at hand conclusion
        #   - next_state: Next game state
        #   - done: Whether the episode ended
        pass
```

#### Reward Structure
- Rewards in Kuhn Poker are based on chip differences after each hand
- Positive reward: chips gained (winning the pot)
- Negative reward: chips lost (folding or losing at showdown)
- Value range: typically between -3 and +5 in simple games
- The engine automatically calculates and assigns rewards to each transition

#### Transition Collection
The engine automatically records:
- State information at decision time
- The action chosen
- Resulting reward
- Whether the episode ended

For custom RL collection, implement:
```python
def record_local_transition(self, transition):
    # Store or process the transition data for your agent
    pass
```

## Running the Game

To run a game with the engine, create a script (e.g., `main.py`/`example.ipynb`) like the following:

```python
from engine.KuhnPokerEngine import KuhnPokerEngine
from players.human_agent import HumanPlayer
from players.random_agent import RandomPlayer
# from players.your_custom_agent import YourCustomAgent

# Create player instances
player0 = HumanPlayer()
player1 = RandomPlayer()
player2 = RandomPlayer()  # For 3-player games

# Initialize the game engine
engine = KuhnPokerEngine(
    player0=player0,
    player1=player1,
    player2=player2,  # include for 3-player games
    delay=0.0,  # Delay between actions (0.0 for human players is best)
    num_players=3,  # 2 or 3 players
    auto_rounds=None  # None for interactive mode, or N for fixed number of rounds
)

# Run the game
engine.run_game()
```

Run from the root folder with:
```
python main.py
# or 
jupyter notebook  # then open and run example.ipynb
```

## Engine Architecture

The `KuhnPokerEngine` has the following main methods:

- `__init__()`: Initializes the game with players and settings
- `run_game()`: Main loop that runs multiple rounds
- `run_round()`: Manages a single round (dealing, betting, showdown)
- `betting_round()`: Handles player actions during a betting round
- `showdown()`: Determines the winner and distributes the pot
- `_deal_cards_2player()/_deal_cards_3player()`: Deals cards based on player count

## Betting Structure

Actions are represented by integers:
- **0: Check/Pass** - Skip betting when there's no bet to call
- **1: Bet** - Place the initial bet (1 chip in basic Kuhn)
- **2: Call** - Match the current highest bet
- **3: Fold** - Forfeit the hand when there's a bet to call
- **4: Raise** - Increase an existing bet (3-player mode supports this)

The engine enforces game rules by only offering valid actions to each player.

## Game Flow

1. **Initialization**: Each player starts with 10 chips
2. **Round Start**: Players ante 1 chip each
3. **Dealing**:
   - 2-player: Each player gets one card from J, Q, K
   - 3-player: Each player gets one card from J, Q, K, A (one card hidden)
4. **First Betting Round**:
   - Players act in turn, starting with player 0
   - Can check, bet, call, fold, or raise (based on context)
5. **Second Betting Round** (if applicable):
   - Only happens if first round had betting and multiple players remain
   - Starts with the next active player after the last player who acted in round 1
6. **Showdown**:
   - Compare cards of non-folded players
   - Highest card wins the pot
7. **Next Round**:
   - If auto_rounds is None, asks player whether to continue
   - If auto_rounds is set, continues until that number is reached

## Code Deep Dive

### Execution Flow

The engine's execution flows through several key methods:

1. **Game Initialization**:
   ```python
   def __init__(self, player0, player1, player2=None, delay=0.5, num_players=2, auto_rounds=None):
       # Setup players, chips (CHIP_TOTAL), RLDataLogger for transitions
       # Initialize game variables
   ```

2. **Main Game Loop** (`run_game`):
   ```python
   def run_game(self):
       # Log game start
       while True:
           self.current_hand += 1
           self.run_round()  # Run a single hand
           
           # Check whether to continue based on auto_rounds setting
           if self.auto_rounds and reached limit:
               break
           elif user chooses not to continue:
               break
       
       # Game end: Log final chip counts and write transitions
   ```

3. **Round Structure** (`run_round`):
   ```python
   def run_round(self):
       # Initialize round variables (pot, bets, folded status)
       # Collect ante from each player (1 chip)
       
       # Deal cards based on player count
       deal_cards_function()
       
       # First betting round
       first_round_actions, last_acted_player = self.betting_round(1)
       
       # Check if game should proceed to showdown:
       # Note: There are no additional rounds in 2-player Kuhn Poker
       if all players checked OR only 2 players and betting happened OR 1 player remaining:
           self.showdown()
           return
       
       # Second betting round if needed (3-player with active betting)
       second_round_actions = self.betting_round(2, starting_player=next_active_after_last_acted)
       
       # Final showdown
       self.showdown()
   ```

### Betting Logic

The `betting_round` method handles all player actions with sophisticated logic:

```python
def betting_round(self, round_num, starting_player=0):
    # Initialize round variables
    # - Track actions list, current player, highest bet
    # - Track players_acted counter, last_bettor
    
    while not all_players_acted:
        # Skip folded players
        if player folded:
            continue
            
        # Determine available actions (check/bet or call/fold/raise)
        available_actions = {}
        if no_bet_to_call:
            available_actions[0] = "check"
            if has_chips:
                available_actions[1] = "bet"
        else:
            if has_enough_chips:
                available_actions[2] = "call"
            available_actions[3] = "fold"
            if has_extra_chips:
                available_actions[4] = "raise"
        
        # Get player's action
        action_idx, raise_amount = player.get_action(...)
        
        # Process action based on type:
        if action is check:
            # Record check action
        elif action is bet:
            # Deduct chips, add to pot, set highest bet
            # Mark this player as last bettor
            # Reset players_acted counter
        elif action is call:
            # Deduct call amount, add to pot
            # Check if all players have acted with equal bets
        elif action is fold:
            # Mark player as folded, reduce active player count
            # Check if only one player remains
        elif action is raise:
            # Process call portion first
            # Then process raise portion
            # Update highest bet, last bettor
            # Reset players_acted counter
            # Update minimum raise amount for future raises
        
        # Record transition for RL
        record_transition(...)
        
        # Move to next player
        players_acted += 1
        current_player = next player
        
        # Check completion conditions:
        # 1. Only one player remains active
        # 2. All players checked (no bets)
        # 3. After betting: everyone acted and bets are equal
    
    return actions, last_acted_player
```

Key aspects of the betting logic:

- **Available Actions**: Dynamically computed based on game state
- **Bet Processing**: Enforces game rules about valid bet amounts
- **Raise Handling**: 
  - First processes the call amount
  - Then adds the raise amount
  - Updates minimum raise size for future raises (important)
- **Completion Checks**: Three ways a betting round can end:
  1. Only one player remains (others folded)
  2. All players check (no betting)
  3. After betting: all players had a chance to act and all bets are equal

### Showdown Implementation

The `showdown` method handles winner determination and chip distribution:

```python
def showdown(self, chips_before_round):
    # Log showdown phase
    # Show cards of non-folded players
    
    if only_one_player_remains:
        # Award pot to last remaining player
    else:
        # Compare card ranks among active players
        # Highest card wins
    
    # Award pot to winner
    self.chips[winner] += self.pot
    
    # Calculate rewards (chip differences)
    # Update transitions with rewards and terminal states
    
    # Log results and chip counts
```

Key Points:
- **Card Ranking**: Simple comparison using a rank dictionary (J=1, Q=2, K=3, A=4)
- **Reward Calculation**: Compares final chips to starting chips for each player
- **Terminal States**: Only the last action for each player in the round is marked as terminal

### Edge Cases and Error Handling

The engine handles below edge cases:

1. **Insufficient Chips**:
   - If a player doesn't have enough to call, they're forced to fold or go all-in
   - If a player can't meet the minimum raise, they can only call or fold
   - Players with zero chips can still check but not bet (standard poker rules)

2. **Raise Amounts**:
   - Minimum raise tracking ensures proper poker raise rules
   - If the raise amount provided is invalid, it's adjusted to the valid range

3. **Completion Conditions**:
   - Special handling for when all players check
   - Special handling for 2-player vs 3-player rules
   - Detection when only one player remains (others folded)

4. **Showdown Logic**:
   - Handles default winner when others fold
   - Properly compares cards for multiple active players

5. **Round Transitions**:
   - In 3-player games, second round starts with the next active player after the last to act
   - Second round skipped in 2-player games after betting (follows Kuhn rules)
   - Second round skipped when all players check in first round

## Data Collection

The engine records transitions for reinforcement learning:
- State information before each action
- Actions taken
- Rewards (chip differences)
- Terminal states

This data can be accessed through the engine's data logger for training machine learning models.

The engine exports this data in two formats:
1. **Raw Transitions**: Full game state records in `rl_data.csv`
2. **Processed Features**: ML-ready features in `training_data.csv` with:
   - Card values, position indicators
   - Pot and chip ratios
   - Betting history statistics
   - One-hot encoded actions
   - Final rewards

### RL Specific

The engine automatically collects data for reinforcement learning:

1. **RL Transitions**: Recorded at each decision point:
   ```python
   transition_data = {
       "session_id": unique identifier,
       "round": hand number,
       "decision_index": action sequence number,
       "stage": "first" or "second" betting round,
       "current_player": player ID,
       "state": {
           # Detailed state information
           "player0_card": card of player 0,
           "player1_card": card of player 1,
           "player2_card": card of player 2 (if applicable),
           "pot": current pot size,
           "chips": string of chip counts,
           "betting_history": action history,
           "highest_bet": current highest bet,
           "last_bettor": player who bet/raised last
       },
       "legal_actions": available actions,
       "chosen_action": action taken,
       "reward": 0,  # Updated later at showdown
       "done": False  # Updated later at showdown
   }
   ```

2. **Reward Assignment**: During showdown:
   ```python
   # Group transitions by player and round
   for each player's transitions:
       # Calculate reward (chip difference)
       reward = chips_after - chips_before
       
       # Update all transitions with reward
       for transition in player_transitions[:-1]:
           transition["reward"] = reward
           transition["done"] = False
           
       # Mark last transition as terminal
       player_transitions[-1]["reward"] = reward
       player_transitions[-1]["done"] = True
   ```

3. **Federated Learning**: For players implementing federated learning:
   ```python
   # Record transition in player's local format
   def record_local_transition(self, transition):
       formatted = {
           'local_state': str(state),
           'local_action': action,
           'local_reward': reward,
           'local_done': done
       }
       self.local_transitions.append(formatted)
   
   # Export data at the end of the session
   def flush_federated_transitions(player):
       write_to_csv(player.local_transitions)
       player.local_transitions.clear()
   ```
