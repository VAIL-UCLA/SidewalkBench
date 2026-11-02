"""Random streams for reproducible scene resets, independent of policy execution."""

from __future__ import annotations

import hashlib

import numpy as np


SCENE_RNG_VERSION = 1


def derive_seed(seed: int, *parts: int | str) -> int:
    """Derive a stable uint32 seed without Python's process-dependent hash()."""
    words = [SCENE_RNG_VERSION, int(seed)]
    for part in parts:
        if isinstance(part, str):
            digest = hashlib.sha256(part.encode("utf-8")).digest()
            words.extend(int.from_bytes(digest[i:i + 4], "little") for i in range(0, 32, 4))
        else:
            words.append(int(part))
    return int(np.random.SeedSequence(words).generate_state(1)[0])


class SceneRandom:
    """Key streams by cell and accepted episode, with a separate retry counter.

    Each accessor returns a fresh generator. Callers retain it for all draws in
    one operation, or use distinct purpose/item keys for independent operations.
    Long-horizon cells have one layout regardless of visits or recovery duration.
    """

    def __init__(self, seed: int, *, long_horizon: bool = False):
        self.long_horizon = long_horizon
        self.reseed(seed)

    def reseed(self, seed: int) -> None:
        self.seed = int(seed)
        self._episodes: dict[tuple[int, int], tuple[int, int]] = {}

    def begin_reset(self, cells, *, retry: bool = False) -> None:
        for cell in sorted(set((int(row), int(col)) for row, col in cells)):
            episode, attempt = self._episodes.get(cell, (-1, 0))
            if self.long_horizon:
                episode, attempt = 0, 0
            elif retry:
                if episode < 0:
                    raise RuntimeError("Cannot retry a scene before its initial reset")
                attempt += 1
            else:
                episode, attempt = episode + 1, 0
            self._episodes[cell] = (episode, attempt)

    def seed_for(self, row: int, col: int, purpose: str, item: int = 0) -> int:
        episode, attempt = self._episodes.get((int(row), int(col)), (0, 0))
        return derive_seed(self.seed, int(row), int(col), episode, attempt, purpose, item)

    def numpy(self, row: int, col: int, purpose: str, item: int = 0) -> np.random.RandomState:
        # RandomState retains the sampling API used by the terrain callbacks.
        return np.random.RandomState(self.seed_for(row, col, purpose, item))

    def torch(self, row: int, col: int, purpose: str, item: int = 0, *, device="cpu"):
        import torch

        return torch.Generator(device=device).manual_seed(self.seed_for(row, col, purpose, item))
