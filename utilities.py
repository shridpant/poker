# utilities.py

import datetime
import csv
import os

def get_timestamp():
    """Returns the current time in HH:MM:SS format."""
    return datetime.datetime.now().strftime("%H:%M:%S")

def log_to_console_and_store(message, log_list):
    """
    Prints a timestamped message to the console and saves it in a log list.
    """
    timestamp = get_timestamp()
    full_message = f"[{timestamp}] {message}"
    print(full_message)
    log_list.append(full_message)

def append_to_log_file(log_lines, filename="logs/game_log.txt"):
    """
    Appends the given log lines to a file, with each session separated for clarity.
    """
    with open(filename, "a") as f:
        f.write("=== New Game Session ===\n")
        for line in log_lines:
            f.write(line + "\n")
        f.write("\n")

def write_transitions_to_csv(transitions, log_func, filename="logs/game_data/rl_data.csv"):
    """
    Appends RL training transitions to a CSV file, creating headers if needed.

    :param transitions: A list of dictionaries describing transitions
    :param log_func: A logging function (for example, log_to_console_and_store)
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
    import os
    import csv
    
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