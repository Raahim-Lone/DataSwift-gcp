#!/usr/bin/env python3
"""
eim.py – Embedding-Indexed Memory v2  (2025-05-24)
───────────────────────────────────────────────────
•  k-NN (Faiss IndexFlatL2) on Plan-GNN + PCA embeddings
•  Each stored point keeps a miniature history so we can                              
   down-weight “bad” hints and bubble-up consistently good ones
•  Radius τ is *adaptive* – it shrinks when we keep finding
   useful neighbours and widens only when we get misses
"""

from __future__ import annotations
import pickle, threading
from pathlib import Path
from collections import deque

import numpy as np
import faiss                              # pip install faiss-cpu


__all__ = ["EmbeddingMemory"]

# ──────────────────────────────────────────────────────────────────────
_MAX_BAD_RATIO = 0.40      # neighbour discarded if >60 % of its trials are bad
_MIN_TRIALS     = 8        # need at least this many observations to judge it
_SHRINK_FACTOR  = 0.92     # τ gets multiplied by this after every “hit”
_GROW_FACTOR    = 1.07     # τ gets multiplied by this after every “miss”
_MIN_TAU        = 2.0
_MAX_TAU        = 20.0


class _Entry(tuple):
    """
    Lightweight immutable record:
        (hint_idx, best_lat_ms, n_trials, n_bad)
    """
    __slots__ = ()             # no per-instance dict


class EmbeddingMemory:
    def __init__(
        self,
        dim: int,
        tau: float = 8.0,
        persist: str | Path = "~/models/eim.pkl",
        keep: int = 100_000,
    ):
        self.dim, self.tau, self.keep = dim, float(tau), keep
        self.persist = Path(persist).expanduser()

        # -- Faiss
        self._keys  = np.empty((0, dim), "float32")
        self._index = faiss.IndexFlatL2(dim)

        # each _vals[i] is _Entry
        self._vals: deque[_Entry] = deque(maxlen=keep)

        if self.persist.exists():
            self._load()

    # ───────────────────── public API ────────────────────────────────
    def query(
        self, z: np.ndarray, k: int = 5
    ) -> tuple[int | None, float | None]:
        """
        Return the best neighbour inside current τ (if any) **and**
        with a healthy win-rate.  Otherwise → (None, None).
        """
        if not self._vals:
            self._adapt_tau(hit=False)
            return None, None

        z = z.astype("float32")
        k = min(k, len(self._vals))
        D, I = self._index.search(z, k)        # (1, k)

        best_idx, best_lat = None, None
        for dist2, row in zip(D[0], I[0]):
            if dist2 > self.tau * self.tau:
                break                          # all remaining neighbours outside τ
            hint_idx, lat, n_total, n_bad = self._vals[row]
            if n_total >= _MIN_TRIALS and (n_bad / n_total) > _MAX_BAD_RATIO:
                continue                       # mostly bad → skip
            if best_lat is None or lat < best_lat:
                best_idx, best_lat = hint_idx, lat

        self._adapt_tau(hit=(best_idx is not None))
        return best_idx, best_lat

    def best_of_knn(
        self, z: np.ndarray, k: int = 20
    ) -> tuple[int | None, float | None]:
        """
        Same as v1 but ignores the τ filter (pure k-NN).
        Still applies bad-ratio pruning.
        """
        if not self._vals:
            return None, None
        k = min(k, len(self._vals))
        D, I = self._index.search(z.astype("float32"), k)
        best_idx, best_lat = None, None
        for row in I[0]:
            hint_idx, lat, n_total, n_bad = self._vals[row]
            if n_total >= _MIN_TRIALS and (n_bad / n_total) > _MAX_BAD_RATIO:
                continue
            if best_lat is None or lat < best_lat:
                best_idx, best_lat = hint_idx, lat
        return best_idx, best_lat

    # feedback from the outside world
    def update_stats(self, z: np.ndarray, hint_idx: int, latency_ms: float, ok: bool):
        """
        Called *after* the true latency is known so we can increment n_trials / n_bad.
        """
        if not self._vals:
            return
        z = z.astype("float32")
        D, I = self._index.search(z, 1)
        if D[0, 0] > 1e-6:                      # must be an exact vector we added
            return
        i = I[0, 0]
        h, best_lat, n_total, n_bad = self._vals[i]

        best_lat = min(best_lat, latency_ms)
        n_total += 1
        n_bad   += int(not ok)
        self._vals[i] = _Entry((h, best_lat, n_total, n_bad))

    def add(self, z: np.ndarray, hint_idx: int, latency_ms: float, ok: bool):
        if len(self._vals) >= self.keep:
            self._evict(10_000)

        self._keys  = np.vstack([self._keys, z])
        self._index.add(z.astype("float32"))
        self._vals.append(
            _Entry((hint_idx, latency_ms, 1, int(not ok)))
        )

        # async save every 5 k inserts
        if len(self._vals) % 5_000 == 0:
            threading.Thread(target=self._dump, daemon=True).start()
    def seed_default(self, z: np.ndarray, latency_ms: float):
        """
        Explicitly add the default-plan latency for this vector.
        Use only once per template to initialize baseline.
        """
        # hint_idx should be 0 (or whatever your default index is)
        self.add(z, hint_idx=0, latency_ms=latency_ms, ok=True)

    def get_baseline(self, z: np.ndarray) -> float | None:
        """
        Return the stored default-plan latency for z, or None.
        """
        return self.query(z, k=1)[1]

    # ──────────────────── helpers / persistence ──────────────────────
    def _adapt_tau(self, hit: bool):
        """
        Very cheap “self-tuning” radius:
        shrink when we keep finding matches (higher precision),
        grow when queries miss a neighbour (higher recall).
        """
        if hit:
            self.tau = max(_MIN_TAU, self.tau * _SHRINK_FACTOR)
        else:
            self.tau = min(_MAX_TAU, self.tau * _GROW_FACTOR)

    def _evict(self, n: int):
        self._keys  = self._keys[n:]
        self._vals  = deque(list(self._vals)[n:], maxlen=self.keep)
        self._index = faiss.IndexFlatL2(self.dim)
        self._index.add(self._keys.astype("float32"))

    def _dump(self):
        with open(self.persist, "wb") as f:
            pickle.dump((self._keys, list(self._vals), self.tau), f)

    def _load(self):
        k, v, tau = pickle.load(open(self.persist, "rb"))
        self._keys = k.astype("float32")
        self._vals = deque(map(_Entry, v), maxlen=self.keep)
        self._index.add(self._keys)
        self.tau = float(tau)
