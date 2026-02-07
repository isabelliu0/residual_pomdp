# Residual Controllers for POMDP TAMP

Online residual reinforcement learning to improve controller reliability in partially observable task and motion planning (TAMP) systems.

## Overview

This repository implements **skill-specific residual policies** that learn online corrections to base controllers to handle partial observability and action uncertainty in POMDP environments.

### Key Idea

In POMDPs, motion planners operating on mean/canonical states can produce unreliable actions due to uncertainty. We learn residual policies:

```
π*(belief) = π_base(selectCanonicalState(belief)) + π_residual(belief)
```

The residual policy learns small corrections that improve the success rate of achieving the expected verified effects predicted by the symbolic planner.

## Quick Start

### Training Residual Policies

**Standard Training:**

```bash
python experiments/train_cover2d.py --num-episodes 1000 --max-steps 100 --seed 0
```

**Training with Domain Randomization:**

```bash
python experiments/train_cover2d_dr.py --num-episodes 1000 --max-steps 100 --seed 0
```

Domain randomization varies:

- Transition noise std: [0.2, 0.5]
- Action effectiveness scale: [0.8, 1.2]
- Rotation noise scale: [0.3, 0.7]

**Pure RL Baseline (for comparison):**

```bash
python experiments/train_cover2d_pure_rl.py --num-episodes 1000 --max-steps 200 --seed 0
```

### Evaluating Trained Policies

**Evaluate with residual policies:**

```bash
python experiments/eval_cover2d.py \
  --pick-model trained_models/20250105_143022/residual_pick_final.pkl \
  --place-model trained_models/20250105_143022/residual_place_final.pkl \
  --num-episodes 100 \
  --seed 42
```

Important evaluation parameters:

- `--no-residual`: Whether we want to evaluate with residual policies
- `--stochastic`: Whether we want to make the trained policies deterministic at evaluation
