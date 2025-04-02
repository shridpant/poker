# utilities.py

import datetime
import csv
import os
import ast

# Centralized log directory references
LOG_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "logs", "game_data")
RL_DATA_FILE = os.path.join(LOG_DATA_DIR, "rl_data.csv")
TRAINING_DATA_FILE = os.path.join(LOG_DATA_DIR, "training_data.csv")
GAME_LOG_FILE = os.path.join(LOG_DATA_DIR, "game_log.txt")

class RLDataLogger:
    """Handles RL transitions and data exporting."""
    def __init__(self):
        # Store transitions here instead of the engine
        self.transitions = []

    def record_transition(self, transition):
        self.transitions.append(transition)

    def write_transitions(self, log_callback, filename=RL_DATA_FILE):
        if self.transitions:
            from engine.utilities import write_transitions_to_csv
            write_transitions_to_csv(self.transitions, log_callback, filename)
            # Also export training data
            self.export_training_data(log_callback)
        
    def export_training_data(self, log_callback, filename=TRAINING_DATA_FILE):
        if not self.transitions:
            log_callback("No transitions to export.")
            return
        
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # Card mapping to numeric values
        card_map = {"J": 1, "Q": 2, "K": 3, "A": 4}
        
        # Prepare structured data
        training_data = []
        
        for t in self.transitions:
            try:
                # Parse the state string into a dictionary
                if isinstance(t["state"], str):
                    state = ast.literal_eval(t["state"])
                else:
                    state = t["state"]
                    
                # Feature vector
                features = {
                    # Convert cards to numeric values
                    "player_card": card_map.get(state.get(f"player{t['current_player']}_card", ""), 0),
                    "pot_ratio": state.get("pot", 0) / 10.0,  # Normalize pot size
                    "chips_ratio": float(state.get("chips", "0;0").split(";")[min(t["current_player"], len(state.get("chips", "0;0").split(";"))-1)]) / 10.0 if state.get("chips") else 0.0,
                    "round": state.get("round", 1) / 5.0,  # Normalize round number
                    "is_first_round": 1.0 if state.get("stage") == "first" else 0.0,
                    # One-hot encode betting position
                    "position_p0": 1.0 if t["current_player"] == 0 else 0.0,
                    "position_p1": 1.0 if t["current_player"] == 1 else 0.0,
                    "position_p2": 1.0 if t["current_player"] == 2 else 0.0,
                    # Add betting history features
                    "history_bet_count": state.get("betting_history", "").count("bet") / 5.0,
                    "history_raise_count": state.get("betting_history", "").count("raise") / 5.0,
                    "history_fold_count": state.get("betting_history", "").count("fold") / 5.0,
                }
                
                # One-hot encode action
                action_vec = [0] * 5  # 5 possible actions
                action_idx = int(t.get("chosen_action", 0))
                if 0 <= action_idx < 5:
                    action_vec[action_idx] = 1
                    
                # Row with features, action, reward, done
                row = {
                    "player_id": t["current_player"],
                    "episode": f"{t['session_id']}_{t['round']}",
                    "step": t["decision_index"],
                    **features,
                    "action_check": action_vec[0],
                    "action_bet": action_vec[1], 
                    "action_call": action_vec[2],
                    "action_fold": action_vec[3],
                    "action_raise": action_vec[4],
                    "reward": t["reward"],
                    "done": 1 if t["done"] else 0
                }
                
                training_data.append(row)
            except Exception as e:
                log_callback(f"Error processing transition: {e}")
        
        # Write
        if training_data:
            with open(filename, 'w', newline='') as csvfile:
                fieldnames = list(training_data[0].keys())
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(training_data)
                log_callback(f"Training data exported to {filename}")
        else:
            log_callback("No valid training data to export.")

def get_timestamp():
    """Returns the current time in HH:MM:SS format."""
    return datetime.datetime.now().strftime("%H:%M:%S")

def write_transitions_to_csv(transitions, log_func, filename="logs/game_data/rl_data.csv"):
    """
    Appends RL training transitions to a CSV file, creating headers if needed.

    :param transitions: A list of dictionaries describing transitions
    :param log_func: A logging function
    :param filename: The CSV file path to append to
    """
    file_exists = os.path.isfile(filename)
    fieldnames = [
        "session_id", "round", "decision_index", "stage", "current_player",
        "state", "legal_actions", "chosen_action", "reward", "done"
    ]

    with open(filename, "a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

        for t in transitions:
            # Convert complex fields to strings
            t["state"] = str(t["state"])
            t["legal_actions"] = str(t["legal_actions"])
            writer.writerow(t)

    log_func(f"Transitions written to {filename}.")

def write_federated_data(transitions, filename):
    """
    Writes federated learning transitions to a CSV file.
    
    Args:
        transitions: List of transition dictionaries
        filename: Path to CSV file
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Determine if file already exists (to know whether to write header)
    file_exists = os.path.isfile(filename)
    
    # Define column names based on transition format
    fieldnames = ["local_state", "local_action", "local_reward", "local_done"]
    
    # Write to CSV file
    with open(filename, "a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        for t in transitions:
            writer.writerow(t)

def log_message(message, filename=GAME_LOG_FILE):
    """Write a log message with a timestamp to the specified file."""
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    formatted_msg = f"[{timestamp}] {message}"
    print(formatted_msg)
    try:
        with open(filename, "a") as f:
            f.write(formatted_msg + "\n")
    except Exception as e:
        print(f"Failed to write log to {filename}: {e}")

def record_local_transition_if_applicable(player, action_idx):
    if getattr(player, 'supports_federated', False):
        transition_data = {
            "local_state": "example_state",
            "local_action": action_idx,
            "local_reward": 0,
            "local_done": False
        }
        player.record_local_transition(transition_data)

def flush_federated_transitions(player):
    filename = f"{LOG_DATA_DIR}/federated_player_{player.player_id}_data.csv"
    write_federated_data(player.local_transitions, filename)
    player.local_transitions.clear()