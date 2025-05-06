# Regret Guided Neural Fictitious Self Player (RG-NFSP) for Kuhn Poker

This document explains how to train the RGNFSP agents and gives a brief overview of how they work.

## How It Works

RGNFSP agents learn from self-play data that is stored for each agent in a memory attribute which is implemented by a reservoir buffer.

## Training the RG-NFSP Agents

To train the RGNFSP 2 player agent and you can run:

```bash
python3 scripts/rgnfsp/rg_nfsp_2p_trainer.py
```

To train the RGNFSP 3 player agent and you can run:

```bash
python3 scripts/rgnfsp/rg_nfsp_3p_trainer.py
```

I did not implement command line arguments, so if you need to configure any hyperparamters please modify the relevant python file.