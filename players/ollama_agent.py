# players/ollama_agent.py

import json
import requests
import re
from players.base import Player

# Possible betting actions to their names
ACTION_CODES = {0: "check", 1: "bet", 2: "call", 3: "fold", 4: "raise"}

class OllamaPlayer(Player):
    """
    A poker player that uses Ollama's deepseek model to make decisions.
    """
    
    def __init__(self, model_name="deepseek-r1:1.5b", temperature=0.7):
        """
        Initialize the Ollama-based player.
        
        Args:
            model_name: Name of the Ollama model to use
            temperature: Controls randomness in model outputs (0.0-1.0)
        """
        self.model_name = model_name
        self.temperature = temperature
        self.ollama_api_url = "http://localhost:11434/api/generate"
        self.history = []  # Track hands and decisions for context
    
    def query_ollama(self, prompt):
        """
        Send a prompt to the Ollama API and get a response.
        
        Args:
            prompt: The text prompt to send to the model
            
        Returns:
            The model's text response
        """
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "temperature": self.temperature,
            "stream": False
        }
        
        try:
            response = requests.post(self.ollama_api_url, json=payload)
            response.raise_for_status()  # Raise exception for HTTP errors
            return response.json().get("response", "")
        except Exception as e:
            print(f"Error querying Ollama API: {e}")
            # Fallback to a random action if API call fails
            return "random"
    
    def format_prompt(self, card, available_actions, round_num, chips_remaining):
        """
        Format the game state into a prompt for the LLM.
        """
        # Convert card to a more descriptive format
        card_names = {"J": "Jack", "Q": "Queen", "K": "King", "A": "Ace"}
        card_name = card_names.get(card, card)
        
        action_descriptions = "\n".join([f"{idx}: {desc}" for idx, desc in available_actions.items()])
        
        # Keep context brief but informative
        context = f"""
You are an AI poker player in a game of Kuhn Poker. Your goal is to maximize winnings.
Here is the current situation:
- Your card: {card_name}
- Current betting round: {round_num}
- Your remaining chips: {chips_remaining}
- Available actions:
{action_descriptions}

Based on this information, which action should you take? Respond with ONLY the action number.
If you want to raise, add the amount after the action number (e.g., "4 3" to raise by 3 chips).
Do not include any explanations, thinking, or other text in your response.
"""
        
        # Add some history for context if available
        if self.history:
            recent_history = self.history[-3:] if len(self.history) > 3 else self.history
            history_str = "\nRecent game history:\n" + "\n".join(recent_history)
            context += history_str

        # Add more explicit instruction to produce only a numeric answer:
        context += """
IMPORTANT:
- You must output exactly one line containing only the action number, or "action_number raise_amount".
- Do not include explanations, chain-of-thought, or additional text.
- Example valid outputs: "0" or "4 2" (if raising by 2).
"""

        return context
    
    def extract_action_number(self, response):
        """
        Extract the numeric action (and optional raise amount) by searching only
        for lines that match the final one-line format, ignoring chain-of-thought.
        """
        # Only look at each line and find something like "4 2" or "3"
        lines = response.strip().splitlines()
        final_line = None
        for line in reversed(lines):
            # e.g. line = "4 2" or "0"
            match = re.match(r'^(\d+)(?:\s+(\d+))?$', line.strip())
            if match:
                final_line = match
                break

        if not final_line:
            return None, None

        action_str, raise_str = final_line.groups()
        return action_str, raise_str
    
    def get_action(self, card, available_actions, round_num, chips_remaining):
        """
        Use Ollama to choose an action based on the current game state.
        
        Args:
            card: The player's card
            available_actions: Dictionary of available actions
            round_num: Current betting round (1 or 2)
            chips_remaining: The number of chips the player has
            
        Returns:
            tuple: (Index of chosen action, raise amount)
        """
        # Format the prompt with game state information
        prompt = self.format_prompt(card, available_actions, round_num, chips_remaining)
        
        # Query the LLM
        response = self.query_ollama(prompt)
        
        # For debugging, log the raw response
        print(f"Raw LLM response: '{response}'")
        
        # Extract the action number from the response
        action_str, raise_str = self.extract_action_number(response)
        
        # Record this decision for future context
        self.history.append(
            f"Round {round_num}: Card {card}, chose '{response}' with {chips_remaining} chips"
        )
        
        try:
            # If we got an action number, parse it
            if action_str is not None:
                action_idx = int(action_str)
                
                # If action isn't valid, default to a safe action
                if action_idx not in available_actions:
                    print(f"Invalid action {action_idx}, defaulting to first available action")
                    action_idx = list(available_actions.keys())[0]
                
                # Handle raise with amount
                if action_idx == 4 and raise_str is not None:
                    try:
                        raise_amount = int(raise_str)
                        # Ensure raise is within limits
                        raise_amount = min(raise_amount, chips_remaining)
                        raise_amount = max(1, raise_amount)  # Minimum raise is 1
                        print(f"Choosing action {action_idx} with raise amount {raise_amount}")
                        return (action_idx, raise_amount)
                    except ValueError:
                        # If parsing raise amount fails, use a default raise
                        default_raise = min(2, chips_remaining)
                        return (4, default_raise)
                
                # For non-raise actions
                print(f"Choosing action {action_idx}")
                return (action_idx, 0)
            else:
                # If we couldn't extract an action, fall back to a random choice
                raise ValueError("Could not extract action from response")
                
        except (ValueError, IndexError) as e:
            # If parsing fails completely, default to checking/calling
            print(f"Error parsing model response: {e}, selecting fallback action")
            # Use random for fallback
            import random
            
            if 0 in available_actions:
                return (0, 0)  # Check if possible
            elif 2 in available_actions:
                return (2, 0)  # Call if possible
            else:
                fallback = random.choice(list(available_actions.keys()))
                if fallback == 4:
                    return (4, 1)  # minimal raise
                return (fallback, 0)
    
    def record_local_transition(self, transition):
        """
        Store transition for future learning.
        """
        pass
