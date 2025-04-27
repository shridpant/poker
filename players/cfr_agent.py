import random
import json
import argparse
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader

# -----------------------------------------------------------------------------
# Constants & Utilities
# -----------------------------------------------------------------------------
NUM_ACTIONS = 2
CARD_MAP = {'J': 0, 'Q': 1, 'K': 2, 0: 0, 1: 1, 2: 2}  # supports int or str cards
ACTION_HISTORY_MAP = {'check': 'p', 'bet': 'b'}
DEFAULT_STRAT = [0.5, 0.5]
POT_NORM = 10.0  # normalization denominators (keep consistent)
CHIPS_NORM = 10.0
ROUND_NORM = 5.0
HIST_NORM = 5.0

def safe_infoset(card, betting_history):
    # returns robustly encoded infoset string
    c = CARD_MAP.get(card, '?')
    hist = ''.join(ACTION_HISTORY_MAP.get(a, '?') for a in betting_history)
    return f"{c}{hist}"

def safe_action_prob_lookup(strategy, infoset):
    v = strategy.get(infoset)
    if v is None or len(v) != 2:
        return list(DEFAULT_STRAT)
    return v

def safe_available(available_actions):
    # returns integer set for action matching
    return set(int(k) for k in available_actions.keys())

# -----------------------------------------------------------------------------
# CFRNode class
# -----------------------------------------------------------------------------
class CFRNode:
    def __init__(self, infoset):
        self.infoset = infoset
        self.regret_sum = [0.0] * NUM_ACTIONS
        self.strategy = [0.0] * NUM_ACTIONS
        self.strategy_sum = [0.0] * NUM_ACTIONS

    def get_strategy(self, realization_weight):
        normalizing_sum = sum(max(r, 0.0) for r in self.regret_sum)
        for a in range(NUM_ACTIONS):
            self.strategy[a] = max(self.regret_sum[a], 0.0)
        if normalizing_sum > 0:
            for a in range(NUM_ACTIONS):
                self.strategy[a] /= normalizing_sum
        else:
            for a in range(NUM_ACTIONS):
                self.strategy[a] = 1.0 / NUM_ACTIONS
        for a in range(NUM_ACTIONS):
            self.strategy_sum[a] += realization_weight * self.strategy[a]
        return list(self.strategy)

    def get_average_strategy(self):
        total = sum(self.strategy_sum)
        if total > 0:
            return [s / total for s in self.strategy_sum]
        return [1.0 / NUM_ACTIONS] * NUM_ACTIONS

    @staticmethod
    def is_terminal(history):
        if len(history) >= 2:
            if history[-1] == 'p':
                return True
            if history[-2:] == 'bb':
                return True
        return False

    @staticmethod
    def payoff(history, cards, player_iter):
        rank = {0: 1, 1: 2, 2: 3}
        terminal_pass = history[-1] == 'p'
        double_bet = history[-2:] == 'bb'
        opp = 1 - player_iter
        if terminal_pass and history == 'pp':
            return 1 if rank[cards[player_iter]] > rank[cards[opp]] else -1
        if terminal_pass:
            return 1
        if double_bet:
            return 2 if rank[cards[player_iter]] > rank[cards[opp]] else -2
        return 0

# -----------------------------------------------------------------------------
# Tabular CFR: Chance Sampling
# -----------------------------------------------------------------------------
class CFRTrainerChance:
    def __init__(self, iterations=1000000, out_dir=None):
        self.node_map = {}
        self.iterations = iterations
        base = out_dir or os.path.join(os.path.dirname(__file__), '..', 'logs', 'game_data')
        self.out_dir = os.path.normpath(base)
        os.makedirs(self.out_dir, exist_ok=True)

    def train(self):
        total_util = 0.0
        cards = [0, 1, 2]
        for _ in range(self.iterations):
            random.shuffle(cards)
            total_util += self._cfr(cards, '', 1.0, 1.0)
        avg_value = total_util / self.iterations
        print(f"[ChanceCFR] Avg game value: {avg_value:.3f}")
        strat = {iset: node.get_average_strategy() for iset, node in self.node_map.items()}
        path = os.path.join(self.out_dir, 'cfr_strategy_chance.json')
        try:
            with open(path, 'w') as f:
                json.dump(strat, f, indent=2)
            print(f"[ChanceCFR] Strategy saved to {path}")
        except Exception as e:
            print(f"Error saving strategy: {e}")
        return avg_value

    def _cfr(self, cards, history, p0, p1):
        if CFRNode.is_terminal(history):
            return CFRNode.payoff(history, cards, 0)
        player = len(history) % 2
        key = str(cards[player]) + history
        node = self.node_map.setdefault(key, CFRNode(key))
        strat = node.get_strategy(p0 if player == 0 else p1)
        util = [0.0] * NUM_ACTIONS
        node_util = 0.0
        for a in range(NUM_ACTIONS):
            nxt = history + ('p' if a == 0 else 'b')
            if player == 0:
                util[a] = -self._cfr(cards, nxt, p0 * strat[a], p1)
            else:
                util[a] = -self._cfr(cards, nxt, p0, p1 * strat[a])
            node_util += strat[a] * util[a]
        for a in range(NUM_ACTIONS):
            regret = util[a] - node_util
            node.regret_sum[a] += (p1 if player == 0 else p0) * regret
        return node_util

class CFRAgentChance:
    def __init__(self, strategy_file=None):
        base = strategy_file or os.path.join(os.path.dirname(__file__), '..', 'logs', 'game_data', 'cfr_strategy_chance.json')
        path = os.path.normpath(base)
        try:
            with open(path) as f:
                self.strategy = json.load(f)
        except Exception as e:
            print(f"Error loading strategy: {e}")
            self.strategy = {}

    def get_action(self, card, available_actions, round_num, chips, public_state):
        infoset = safe_infoset(card, public_state.get('betting_history', []))
        probs = safe_action_prob_lookup(self.strategy, infoset)
        available = safe_available(available_actions)
        if available == {0,1}:  # Check/bet
            return (0 if random.random() < probs[0] else 1, None)
        elif 2 in available:
            return (2, None)
        else:
            return (min(available), None)

# -----------------------------------------------------------------------------
# Tabular CFR: External Sampling
# -----------------------------------------------------------------------------
class CFRTrainerExternal:
    def __init__(self, iterations=1000000, out_dir=None):
        self.node_map = {}
        self.iterations = iterations
        base = out_dir or os.path.join(os.path.dirname(__file__), '..', 'logs', 'game_data')
        self.out_dir = os.path.normpath(base)
        os.makedirs(self.out_dir, exist_ok=True)

    def train(self):
        total_util = 0.0
        cards = [0, 1, 2]
        for _ in range(self.iterations):
            random.shuffle(cards)
            total_util += self._cfr_ext(cards, '', 0)
        avg_value = total_util / self.iterations
        print(f"[ExternalCFR] Avg game value: {avg_value:.3f}")
        strat = {iset: node.get_average_strategy() for iset, node in self.node_map.items()}
        path = os.path.join(self.out_dir, 'cfr_strategy_external.json')
        try:
            with open(path, 'w') as f:
                json.dump(strat, f, indent=2)
            print(f"[ExternalCFR] Strategy saved to {path}")
        except Exception as e:
            print(f"Error saving strategy: {e}")
        return avg_value

    def _cfr_ext(self, cards, history, player_iter):
        if CFRNode.is_terminal(history):
            return CFRNode.payoff(history, cards, player_iter)
        player = len(history) % 2
        key = str(cards[player]) + history
        node = self.node_map.setdefault(key, CFRNode(key))
        avg_strat = node.get_average_strategy()
        if player != player_iter:
            choice = 0 if random.random() < avg_strat[0] else 1
            for a in range(NUM_ACTIONS):
                node.strategy_sum[a] += avg_strat[a]
            return self._cfr_ext(cards, history + ('p' if choice == 0 else 'b'), player_iter)
        else:
            util = [0.0] * NUM_ACTIONS
            node_util = 0.0
            for a in range(NUM_ACTIONS):
                nxt = history + ('p' if a == 0 else 'b')
                util[a] = self._cfr_ext(cards, nxt, player_iter)
                node_util += avg_strat[a] * util[a]
            for a in range(NUM_ACTIONS):
                node.regret_sum[a] += util[a] - node_util
            return node_util

class CFRAgentExternal:
    def __init__(self, strategy_file=None):
        base = strategy_file or os.path.join(os.path.dirname(__file__), '..', 'logs', 'game_data', 'cfr_strategy_external.json')
        path = os.path.normpath(base)
        try:
            with open(path) as f:
                self.strategy = json.load(f)
        except Exception as e:
            print(f"Error loading strategy: {e}")
            self.strategy = {}

    def get_action(self, card, available_actions, round_num, chips, public_state):
        infoset = safe_infoset(card, public_state.get('betting_history', []))
        probs = safe_action_prob_lookup(self.strategy, infoset)
        available = safe_available(available_actions)
        if available == {0,1}:  # Check/bet
            return (0 if random.random() < probs[0] else 1, None)
        elif 2 in available:
            return (2, None)
        else:
            return (min(available), None)

# -----------------------------------------------------------------------------
# Neural Network Policy
# -----------------------------------------------------------------------------
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, NUM_ACTIONS)
        )

    def forward(self, x):
        return self.net(x)

def train_neural(data_csv, model_out, epochs=10, batch_size=64, lr=1e-3):
    df = pd.read_csv(data_csv)
    # Only include rows where check/bet is available (ignoring fold/raise for standard 2-action CFR)
    df = df[(df['action_fold'] == 0) & (df['action_call'] == 0) & (df['action_raise'] == 0)]
    feats = [
        'player_card', 'pot_ratio', 'chips_ratio', 'round', 'is_first_round',
        'position_p0', 'position_p1', 'position_p2', 'history_bet_count', 'history_raise_count'
    ]
    if len(df) == 0:
        raise ValueError("After filtering, no data available for check/bet only states!")
    X = torch.tensor(df[feats].values, dtype=torch.float)
    y = torch.tensor(df[['action_check', 'action_bet']].values.argmax(axis=1), dtype=torch.long)
    ds = TensorDataset(X, y)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
    model = PolicyNetwork(X.shape[1])
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    losses = []
    for ep in range(epochs):
        tot = 0.0
        for xb, yb in dl:
            opt.zero_grad()
            out = model(xb)
            loss = loss_fn(out, yb)
            loss.backward()
            opt.step()
            tot += loss.item()
        avg_loss = tot / len(dl)
        print(f"Epoch {ep+1}/{epochs}, Loss {avg_loss:.4f}")
        losses.append(avg_loss)
    try:
        torch.save(model.state_dict(), model_out)
        print(f"[Neural] Model saved to {model_out}")
    except Exception as e:
        print(f"Error saving model: {e}")
    return losses

class NeuralAgent(nn.Module):
    def __init__(self, model_path, input_dim, device='cpu'):
        super().__init__()
        self.device = device
        self.model = PolicyNetwork(input_dim).to(device)
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=device))
            self.model.eval()
        except Exception as e:
            print(f"Error loading neural model: {e}")
            self.model = PolicyNetwork(input_dim)
            self.model.eval()

    def get_action(self, card, available_actions, round_num, chips, public_state):
        bhist = public_state.get('betting_history', [])
        features = [
            CARD_MAP.get(card, 0),
            public_state.get('pot_size', 0) / POT_NORM,
            public_state.get('chip_counts', [0])[0] / CHIPS_NORM,
            round_num / ROUND_NORM,
            1.0 if public_state.get('stage') == 'first' else 0.0,
            1.0 if public_state.get('current_player') == 0 else 0.0,
            1.0 if public_state.get('current_player') == 1 else 0.0,
            bhist.count('bet') / HIST_NORM,
            bhist.count('raise') / HIST_NORM
        ]
        x = torch.tensor([features], dtype=torch.float, device=self.device)
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        available = safe_available(available_actions)
        if available == {0, 1}:  # Check/bet
            return (0 if random.random() < probs[0] else 1, None)
        elif 2 in available:
            return (2, None)
        else:
            return (min(available), None)

# -----------------------------------------------------------------------------
# CLI Interface
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-chance', action='store_true')
    parser.add_argument('--train-external', action='store_true')
    parser.add_argument('--train-neural', nargs=2, metavar=('DATA_CSV', 'OUT_MODEL'))
    parser.add_argument('--play-chance', action='store_true')
    parser.add_argument('--play-external', action='store_true')
    parser.add_argument('--iterations', type=int, default=1000000)
    args = parser.parse_args()
    if args.train_chance:
        CFRTrainerChance(iterations=args.iterations).train()
    elif args.train_external:
        CFRTrainerExternal(iterations=args.iterations).train()
    elif args.train_neural:
        train_neural(args.train_neural[0], args.train_neural[1])
    elif args.play_chance:
        from engine.KuhnPokerEngine import KuhnPokerEngine
        agent = CFRAgentChance()
        game = KuhnPokerEngine(agent, agent, auto_rounds=100)
        game.run_game()
    elif args.play_external:
        from engine.KuhnPokerEngine import KuhnPokerEngine
        agent = CFRAgentExternal()
        game = KuhnPokerEngine(agent, agent, auto_rounds=100)
        game.run_game()
    else:
        print('Specify --train-chance, --train-external, --train-neural DATA OUT, --play-chance, or --play-external')
