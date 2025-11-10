"""Utilities for belief encoding and particle filtering."""

from typing import Any, Callable

import numpy as np


def encode_belief_generic(belief: Any, encoder_fn: Callable) -> np.ndarray:
    """Generic belief encoder that uses a custom encoding function."""
    return encoder_fn(belief)
