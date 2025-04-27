import random
import json
import argparse
import os
import time

# If you want ML integration, uncomment the following:
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import numpy as np

# -----------------------------------------------------------------------------
# Counterfactual Regret Minimization Trainer for Kuhn Poker (PASS/BET)
# -----------------------------------------------------------------------------

class CFRNode:
    def __init__(self, infoset):
        self.infoset = infoset
        self.regret_sum = [0.0, 0.0]
        self.strategy = [0.0, 0.0]
        self.strategy_sum = [0.0, 0.0]

    def get_strategy(self, realization_weight):
        """
        Compute current strategy via regret-matching and accumulate for averaging.
        """
        normalizing_sum = 0.0
        for a in range(2):
            self.strategy[a] = self.regret_sum[a] if self.regret_sum[a] > 0 else 0.0
            normalizing_sum += self.strategy[a]
        for a in range(2):
            if normalizing_sum > 0:
                self.strategy[a] /= normalizing_sum
            else:
                self.strategy[a] = 1.0 / 2.0
            self.strategy_sum[a] += realization_weight * self.strategy[a]
        return self.strategy

    def get_average_strategy(self):
        """
        Returns the average strategy across all training iterations.
        """
        normalizing_sum = sum(self.strategy_sum)
        if normalizing_sum > 0:
            return [s / normalizing_sum for s in self.strategy_sum]
        else:
            return [1.0 / 2.0, 1.0 / 2.0]


class CFRTrainer:
    def __init__(self, iterations=1000000, out_dir=None):
        self.node_map = {}  # maps infoset -> CFRNode
        self.iterations = iterations
        # Determine output directory for strategy file
        base = out_dir or os.path.join(os.path.dirname(__file__), '..', 'logs', 'game_data')
        self.out_dir = os.path.normpath(base)
        os.makedirs(self.out_dir, exist_ok=True)

    def train(self):
        """
        Train CFR for the specified number of iterations and save strategy to JSON.
        """
        util = 0.0
        cards = [0, 1, 2]  # 0:J, 1:Q, 2:K
        for i in range(self.iterations):
            random.shuffle(cards)
            util += self._cfr(cards, "", 1.0, 1.0)
        avg_game_value = util / self.iterations
        print(f"Average game value: {avg_game_value:.3f}")

        # Save average strategy
        strategy = {infoset: node.get_average_strategy()
                    for infoset, node in self.node_map.items()}
        filepath = os.path.join(self.out_dir, "cfr_strategy.json")
        with open(filepath, "w") as f:
            json.dump(strategy, f, indent=2)
        print(f"[CFRTrainer] Strategy saved to {filepath}")

    def _cfr(self, cards, history, p0, p1):
        plays = len(history)
        # Terminal state check
        if self._is_terminal(history):
            return self._payoff(history, cards)

        player = plays % 2
        opponent = 1 - player
        infoset = str(cards[player]) + history

        # Get or create node
        node = self.node_map.get(infoset)
        if node is None:
            node = CFRNode(infoset)
            self.node_map[infoset] = node

        # Get regret-matched mixed strategy
        strategy = node.get_strategy(p0 if player == 0 else p1)
        util = [0.0, 0.0]
        node_util = 0.0

        # For each action a in {PASS (p), BET (b)}
        for a in range(2):
            next_history = history + ("p" if a == 0 else "b")
            if player == 0:
                util[a] = -self._cfr(cards, next_history, p0 * strategy[a], p1)
            else:
                util[a] = -self._cfr(cards, next_history, p0, p1 * strategy[a])
            node_util += strategy[a] * util[a]

        # Compute and accumulate regrets
        for a in range(2):
            regret = util[a] - node_util
            node.regret_sum[a] += (p1 if player == 0 else p0) * regret

        return node_util

    @staticmethod
    def _is_terminal(history):
        # Terminal if someone passes twice or double bet occurs
        if len(history) >= 2:
            if history[-1] == 'p':
                return True
            if history[-2:] == 'bb':
                return True
        return False

    @staticmethod
    def _payoff(history, cards):
        # Payoff for player 0
        rank = {0: 1, 1: 2, 2: 3}
        terminal_pass = history[-1] == 'p'
        double_bet = history[-2:] == 'bb'
        if terminal_pass and history == "pp":
            return 1 if rank[cards[0]] > rank[cards[1]] else -1
        if terminal_pass and history != "pp":
            return 1
        if double_bet:
            return 2 if rank[cards[0]] > rank[cards[1]] else -2
        return 0


class CFRAgent:
    """
    Agent that plays Kuhn Poker according to a precomputed CFR strategy.
    """
    def __init__(self, strategy_file=None):
        # Default to logs/game_data/cfr_strategy.json
        base = strategy_file or os.path.join(os.path.dirname(__file__), '..', 'logs', 'game_data', 'cfr_strategy.json')
        path = os.path.normpath(base)
        with open(path, "r") as f:
            self.strategy = json.load(f)

    def get_action(self, card, available_actions, round_num, chips, public_state):
        history = "".join(
            ['p' if act == 'check' else 'b' for act in public_state.get('betting_history', [])]
        )
        infoset = str(self._card_to_index(card)) + history
        probs = self.strategy.get(infoset, [0.5, 0.5])

        # PASS=0 -> engine check, BET=1 -> engine bet
        if set(available_actions.keys()) == {0, 1}:
            choice = 0 if random.random() < probs[0] else 1
            return (choice, None)
        # If only fold/call options, call
        if 2 in available_actions:
            return (2, None)
        return (0, None)

    @staticmethod
    def _card_to_index(card):
        return {"J": 0, "Q": 1, "K": 2}[card]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CFR or run sample game.")
    parser.add_argument("--train", action="store_true", help="Train CFR and save strategy.")
    parser.add_argument("--play", action="store_true", help="Play a sample match between two CFR agents.")
    parser.add_argument("--iterations", type=int, default=1000000,
                        help="Number of CFR training iterations.")
    args = parser.parse_args()

    if args.train:
        trainer = CFRTrainer(iterations=args.iterations)
        start = time.time()
        trainer.train()
        print(f"Training took {time.time() - start:.2f} seconds.")
    elif args.play:
        from engine.KuhnPokerEngine import KuhnPokerEngine
        agent0 = CFRAgent()
        agent1 = CFRAgent()
        game = KuhnPokerEngine(agent0, agent1, auto_rounds=100)
        game.run_game()
    else:
        print("Specify --train to train CFR or --play to play a match.")
