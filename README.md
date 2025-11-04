# Residual Controllers for POMDP TAMP

Online residual reinforcement learning to improve controller reliability in partially observable task and motion planning (TAMP) systems.

## Overview

This repository implements **skill-specific residual policies** that learn online corrections to base controllers to handle partial observability and action uncertainty in POMDP environments.

### Key Idea

In POMDPs, motion planners operating on mean/canonical states can produce unreliable actions due to uncertainty. We learn residual policies:

```
π*(belief) = π_base(selectCanonicalState(belief)) + π_residual(belief)
```

The residual policy learns small corrections that improve the success rate of achieving the expected verified effects predicted by the symbolic planner (SymK).
