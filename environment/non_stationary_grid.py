"""
non_stationary_grid.py
======================
2-D grid-world benchmark with non-stationary dynamics.

State space
───────────
  • W × H grid of integer cell types:
      0 = empty, 1 = obstacle, 2 = resource, 3 = hazard, 4 = agent.
  • The agent occupies exactly one cell.

Observation space
─────────────────
  • 5 × 5 egocentric local view  (integer matrix).
  • 1-D internal state vector   [energy_level].

Action space  (stress-test: Consume removed — auto-consume on contact)
────────────
  0 = Up, 1 = Down, 2 = Left, 3 = Right.

Non-stationarity
────────────────
  ``trigger_regime_shift()`` permutes the *cell types* of resources and
  hazards (and optionally relocates them), forcing the agent to
  re-learn which visual cues predict reward vs. danger.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from config import (
    GridConfig,
    CELL_EMPTY,
    CELL_OBSTACLE,
    CELL_RESOURCE,
    CELL_HAZARD,
    CELL_AGENT,
    ACTION_UP,
    ACTION_DOWN,
    ACTION_LEFT,
    ACTION_RIGHT,
    NUM_ACTIONS,
)


class NonStationaryGrid:
    """
    2-D grid world with periodic regime shifts for continual-learning
    evaluation.
    """

    # ── Construction / Reset ──────────────────────────────────────────

    def __init__(self, cfg: GridConfig, rng: np.random.Generator) -> None:
        self.cfg = cfg
        self.rng = rng
        self.width = cfg.width
        self.height = cfg.height
        self.regime: int = 0          # current regime index (A=0, B=1, C=2)
        self.step_count: int = 0

        # Internal grid (cell types)
        self._grid: np.ndarray = np.zeros(
            (self.height, self.width), dtype=np.int32
        )
        # Agent position
        self.agent_row: int = 0
        self.agent_col: int = 0
        # Agent energy
        self.energy: int = cfg.initial_energy

        self.reset()

    def reset(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Re-initialise the grid.  Returns the first observation pair
        (egocentric_view, internal_state).
        """
        self._grid[:, :] = CELL_EMPTY
        self.step_count = 0
        self.energy = self.cfg.initial_energy

        # Place obstacles
        self._scatter_cells(CELL_OBSTACLE, self.cfg.num_obstacles)
        # Place resources & hazards
        self._scatter_cells(CELL_RESOURCE, self.cfg.num_resources)
        self._scatter_cells(CELL_HAZARD, self.cfg.num_hazards)

        # Place agent in a random empty cell
        self.agent_row, self.agent_col = self._random_empty_cell()

        return self._observe()

    # ── Step ──────────────────────────────────────────────────────────

    def step(self, action: int) -> Tuple[np.ndarray, np.ndarray, dict]:
        """
        Execute *action*, return (egocentric_view, internal_state, info).

        info contains:
          "energy", "consumed", "hit_hazard", "regime", "step".
        """
        self.step_count += 1
        info: Dict = {
            "consumed": False,
            "hit_hazard": False,
            "regime": self.regime,
            "step": self.step_count,
        }

        # ── Movement (agent MUST move every step — no rest) ──────────
        dr, dc = 0, 0
        if action == ACTION_UP:
            dr = -1
        elif action == ACTION_DOWN:
            dr = 1
        elif action == ACTION_LEFT:
            dc = -1
        elif action == ACTION_RIGHT:
            dc = 1

        new_r = max(0, min(self.height - 1, self.agent_row + dr))
        new_c = max(0, min(self.width - 1, self.agent_col + dc))
        target = self._grid[new_r, new_c]

        if target != CELL_OBSTACLE:
            self.agent_row, self.agent_col = new_r, new_c

        # ── Auto-consume: stepping onto a resource collects it ────
        if target == CELL_RESOURCE:
            self.energy += self.cfg.energy_gain_resource
            self._grid[new_r, new_c] = CELL_EMPTY
            info["consumed"] = True
            # Respawn a new resource elsewhere
            self._scatter_cells(CELL_RESOURCE, 1)

        if target == CELL_HAZARD:
            self.energy -= self.cfg.energy_loss_hazard
            info["hit_hazard"] = True

        self.energy -= self.cfg.energy_cost_step

        # Clamp energy
        self.energy = max(0, self.energy)
        info["energy"] = self.energy

        obs_view, obs_internal = self._observe()
        return obs_view, obs_internal, info

    # ── Non-stationarity ──────────────────────────────────────────────

    def trigger_regime_shift(self) -> int:
        """
        Swap the visual codes (and relocate) resources ↔ hazards.

        Regime sequence:
          A (0): normal
          B (1): resources become hazards and vice versa (code swap)
          C (2): all dynamic cells relocated + partial code swap

        Returns the new regime index.
        """
        self.regime = (self.regime + 1) % 3

        if self.regime == 1:
            # ── Regime B: swap resource ↔ hazard cell types ──────────
            mask_res = self._grid == CELL_RESOURCE
            mask_haz = self._grid == CELL_HAZARD
            self._grid[mask_res] = CELL_HAZARD
            self._grid[mask_haz] = CELL_RESOURCE

        elif self.regime == 2:
            # ── Regime C: relocate all dynamic objects ────────────────
            self._grid[self._grid == CELL_RESOURCE] = CELL_EMPTY
            self._grid[self._grid == CELL_HAZARD] = CELL_EMPTY
            self._scatter_cells(CELL_RESOURCE, self.cfg.num_resources)
            self._scatter_cells(CELL_HAZARD, self.cfg.num_hazards)

        else:
            # ── Back to A: swap again to restore original semantics ──
            mask_res = self._grid == CELL_RESOURCE
            mask_haz = self._grid == CELL_HAZARD
            self._grid[mask_res] = CELL_HAZARD
            self._grid[mask_haz] = CELL_RESOURCE

        return self.regime

    # ── Observation generation ────────────────────────────────────────

    def _observe(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns
        -------
        ego_view : ndarray of shape (2R+1, 2R+1) — egocentric local view.
        internal : ndarray of shape (1,) — [energy_normalised].
        """
        R = self.cfg.egocentric_radius
        size = 2 * R + 1
        view = np.full((size, size), CELL_OBSTACLE, dtype=np.int32)

        for dy in range(-R, R + 1):
            for dx in range(-R, R + 1):
                r = self.agent_row + dy
                c = self.agent_col + dx
                if 0 <= r < self.height and 0 <= c < self.width:
                    view[dy + R, dx + R] = self._grid[r, c]

        # Mark agent at centre
        view[R, R] = CELL_AGENT

        internal = np.array(
            [self.energy / max(self.cfg.initial_energy, 1)],
            dtype=np.float64,
        )
        return view, internal

    # ── Helpers ───────────────────────────────────────────────────────

    def _scatter_cells(self, cell_type: int, count: int) -> None:
        """Place *count* cells of *cell_type* on random empty positions."""
        for _ in range(count):
            r, c = self._random_empty_cell()
            self._grid[r, c] = cell_type

    def _random_empty_cell(self) -> Tuple[int, int]:
        """Return (row, col) of a uniformly-random empty cell."""
        empties = np.argwhere(self._grid == CELL_EMPTY)
        if len(empties) == 0:
            # Fallback: just pick any cell
            return int(self.rng.integers(self.height)), int(
                self.rng.integers(self.width)
            )
        idx = self.rng.integers(len(empties))
        return int(empties[idx, 0]), int(empties[idx, 1])

    def dominant_obs_type(self, view: np.ndarray) -> int:
        """
        Return the most common *non-empty, non-agent* cell type in the
        egocentric view.  Used by the agent for a simplified observation
        index.
        """
        flat = view.flatten()
        # Mask out empty and agent cells
        masked = flat[(flat != CELL_EMPTY) & (flat != CELL_AGENT)]
        if len(masked) == 0:
            return CELL_EMPTY
        counts = np.bincount(masked, minlength=5)
        # Ignore empty (0) and agent (4) bins
        counts[CELL_EMPTY] = 0
        counts[CELL_AGENT] = 0
        return int(np.argmax(counts))
