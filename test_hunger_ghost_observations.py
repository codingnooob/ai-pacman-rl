#!/usr/bin/env python3
"""
Deterministic regression suite covering hunger enforcement, ghost reward decomposition,
and observation tensor dimensions.

Usage
-----
    python test_hunger_ghost_observations.py

This script is intentionally self-contained so that analysts can repeatedly run it after
changing hunger/ghost/observation code. All checks are deterministic and will raise
AssertionError with descriptive messages if any regression is detected.
"""

from __future__ import annotations

import copy
import math
import random
from collections import deque
from types import MethodType

import numpy as np

try:  # Torch is required by the training stack, but guard the import for clarity
    import torch
except Exception:  # pragma: no cover - torch is expected to be available in this repo
    torch = None

from agent import (
    GhostTeam,
    PacmanAgent,
    GLOBAL_STATE_DIM,
    GHOST_STATE_DIM,
    PACMAN_STATE_DIM,
)
from game import PacmanGame
from trainer import Trainer

RNG_SEED = 1337
FAST_HUNGER_CONFIG = {
    # Force hunger detection within ~40 scripted steps to keep tests lightweight
    'hunger_idle_threshold': 2,
    'hunger_decay_rate': 4.0,
    'hunger_decay_growth': 1.75,
    'hunger_termination_limit': -40.0,
    'stagnation_tile_window': 16,
    'stagnation_tile_min': 12,
    'survival_grace_steps': 12,
    'hunger_termination_reward': -100.0,
}


def _set_global_seed(seed: int = RNG_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)


def _clone_state(state: dict) -> dict:
    """Return a deep copy so patched step functions cannot mutate shared references."""

    return copy.deepcopy(state)


def _apply_fast_hunger_config(trainer: Trainer) -> None:
    trainer.hunger_config.update(FAST_HUNGER_CONFIG)
    for game in trainer.games:
        if hasattr(game, 'hunger_config'):
            game.hunger_config.update(trainer.hunger_config)
        if hasattr(game, 'hunger_config_snapshot'):
            game.hunger_config_snapshot.update(trainer.hunger_config)
    for env_id, game in enumerate(trainer.games):
        trainer._reset_progress_tracker(env_id, game.get_state())


def run_hunger_enforcement_regression() -> dict:
    """Force Pacman to idle so that hunger termination must occur quickly."""

    trainer = Trainer(n_envs=1)
    _apply_fast_hunger_config(trainer)
    game = trainer.games[0]
    original_step = game.step
    original_force_hunger = getattr(game, 'force_hunger_termination', None)

    def idle_step(pacman_action, ghost_actions):  # noqa: ANN001
        if getattr(game, 'done', False):
            return _clone_state(game.get_state()), 0.0, True
        game.steps += 1
        game.game_timer += 1
        # Pacman never moves, so no pellets are eaten and no score progress is made.
        return _clone_state(game.get_state()), -0.1, False

    def safe_force_hunger(reason='HUNGER'):  # noqa: ANN001
        game.done = True
        game.termination_reason = reason or 'HUNGER'

    game.step = idle_step
    if original_force_hunger is not None:
        game.force_hunger_termination = safe_force_hunger

    hunger_entry = None
    initial_counts = dict(trainer.hunger_termination_counts)
    try:
        max_scripted_steps = 50  # With the fast config the meter should trip well before this
        for _ in range(max_scripted_steps):
            trainer.train_step()
            if not trainer.reward_breakdown_history:
                continue
            entry = trainer.reward_breakdown_history[-1]
            if entry['termination_reason'] in ('HUNGER_METER', 'HUNGER_SCORE_FREEZE'):
                hunger_entry = entry
                break
    finally:
        game.step = original_step
        if original_force_hunger is not None:
            game.force_hunger_termination = original_force_hunger

    assert hunger_entry is not None, (
        'Hunger termination never triggered in the scripted idle scenario; '
        'verify hunger thresholds and penalties are still wired correctly.'
    )
    assert hunger_entry['hunger_penalty'] < 0.0, (
        'Reward breakdown should store hunger penalties as negative values '
        'once the penalty is applied.'
    )

    final_counts = trainer.hunger_termination_counts
    total_hunger_events = sum(final_counts.values()) - sum(initial_counts.values())
    assert total_hunger_events >= 1, 'Trainer hunger termination counters never incremented.'

    return {
        'termination_reason': hunger_entry['termination_reason'],
        'hunger_penalty': hunger_entry['hunger_penalty'],
        'hunger_meter': hunger_entry['hunger_meter'],
        'steps_since_progress': hunger_entry['steps_since_progress'],
        'score_freeze_steps': hunger_entry['score_freeze_steps'],
        'unique_tiles': hunger_entry['unique_tiles'],
        'events_recorded': total_hunger_events,
    }


def validate_ghost_reward_decomposition() -> dict:
    """Mock a single trainer step and ensure ghost rewards match the documented formula."""

    trainer = Trainer(n_envs=1)
    game = trainer.games[0]

    base_state = _clone_state(game.get_state())
    scripted_state = _clone_state(base_state)
    scripted_state.update(
        {
            'pacman': (10, 10),
            'ghosts': [(10, 12), (12, 10), (15, 10), (10, 5)],
            'ghost_vulnerable': [False, True, False, False],
            'pellets': {(i, (i * 2) % 28) for i in range(300)},
            'power_pellets': {(1, 1)},
            'steps': 40,
        }
    )
    scripted_state['initial_counts']['pellets'] = 400
    scripted_state['initial_counts']['power_pellets'] = 4

    scripted_next_state = _clone_state(scripted_state)
    scripted_next_state['pellets'] = {(i, (i * 3) % 28) for i in range(295)}  # drop 5 pellets
    scripted_next_state['steps'] = 60
    scripted_next_state['ghost_vulnerable'] = [False, True, True, False]

    original_get_state = game.get_state
    original_step = game.step
    original_force_hunger = getattr(game, 'force_hunger_termination', None)

    def scripted_get_state():
        return _clone_state(scripted_state)

    def scripted_step(pacman_action, ghost_actions):  # noqa: ANN001
        game.steps = scripted_next_state['steps']
        return _clone_state(scripted_next_state), 0.0, False

    def safe_force_hunger(reason='HUNGER'):  # noqa: ANN001
        game.done = True
        game.termination_reason = reason or 'HUNGER'

    game.get_state = scripted_get_state
    game.step = scripted_step
    if original_force_hunger is not None:
        game.force_hunger_termination = safe_force_hunger

    prev_pellets = 310
    trainer.prev_pellet_counts[0] = prev_pellets

    unique_positions = [(r, r + 1) for r in range(trainer.hunger_config['stagnation_tile_min'] + 2)]
    tracker = {
        'steps_since_progress': 36,
        'score_freeze_steps': 180,
        'hunger_meter': -25.0,
        'unique_tile_window': deque(unique_positions, maxlen=trainer.hunger_config['stagnation_tile_window']),
        'last_score': 0,
        'last_pellet_count': prev_pellets,
        'last_position': scripted_state['pacman'],
    }
    trainer.progress_trackers[0] = tracker

    original_update_tracker = trainer._update_progress_tracker

    def fixed_tracker(self, env_id, next_state):  # noqa: ANN001
        return self.progress_trackers[env_id], False

    trainer._update_progress_tracker = MethodType(fixed_tracker, trainer)

    try:
        trainer.train_step()
    finally:
        trainer._update_progress_tracker = original_update_tracker
        game.get_state = original_get_state
        game.step = original_step
        if original_force_hunger is not None:
            game.force_hunger_termination = original_force_hunger

    assert trainer.reward_breakdown_history, 'Trainer never logged a reward breakdown entry.'
    entry = trainer.reward_breakdown_history[-1]

    current_pellets = len(scripted_next_state['pellets']) + len(scripted_next_state['power_pellets'])
    delta_pellets = max((prev_pellets - current_pellets), 0)
    idle_threshold = trainer.hunger_config['hunger_idle_threshold']
    idle_ratio = float(np.clip(tracker['steps_since_progress'] / max(idle_threshold, 1), 0.0, 2.0))
    survival_ratio = float(
        np.clip(scripted_next_state['steps'] / max(trainer.hunger_config['survival_grace_steps'], 1), 0.0, 2.0)
    )

    pacman_position = np.array(scripted_next_state['pacman'], dtype=np.float32)
    norm = trainer.maze_diagonal or 30.0
    distances = []
    for ghost_pos in scripted_next_state['ghosts']:
        ghost_vec = np.array(ghost_pos, dtype=np.float32)
        distances.append(np.linalg.norm(pacman_position - ghost_vec) / norm)
    ghost_ring_distance = float(np.clip(np.mean(distances), 0.0, 1.0))
    vulnerability_fraction = float(np.mean(scripted_next_state['ghost_vulnerable']))

    cfg = trainer.ghost_reward_cfg
    expected_team_pressure = (
        -cfg['kappa1'] * float(delta_pellets) + cfg['kappa2'] * idle_ratio + cfg['kappa3'] * survival_ratio
    )
    expected_proximity_drive = cfg['kappa4'] * (1.0 - ghost_ring_distance)
    expected_vulnerability_tax = -cfg['kappa5'] * vulnerability_fraction

    def _close(a, b, tol=1e-6):
        return math.isclose(a, b, rel_tol=tol, abs_tol=tol)

    assert _close(entry['ghost_team_pressure'], expected_team_pressure), (
        f"Team pressure mismatch: expected {expected_team_pressure:.6f}, got {entry['ghost_team_pressure']:.6f}"
    )
    assert _close(entry['ghost_proximity_drive'], expected_proximity_drive), (
        f"Proximity drive mismatch: expected {expected_proximity_drive:.6f}, got {entry['ghost_proximity_drive']:.6f}"
    )
    assert _close(entry['ghost_vulnerability_tax'], expected_vulnerability_tax), (
        f"Vulnerability tax mismatch: expected {expected_vulnerability_tax:.6f}, got {entry['ghost_vulnerability_tax']:.6f}"
    )

    expected_total = expected_team_pressure + expected_proximity_drive + expected_vulnerability_tax
    expected_with_bonus = expected_total + entry['ghost_termination_bonus']
    assert _close(entry['ghost_total_reward'], expected_with_bonus), (
        f"Ghost total reward mismatch: expected {expected_with_bonus:.6f}, got {entry['ghost_total_reward']:.6f}"
    )
    if entry['termination_reason'] in ('HUNGER_METER', 'HUNGER_SCORE_FREEZE'):
        assert entry['ghost_termination_bonus'] == -trainer.ghost_reward_cfg['hunger_fail_penalty'], (
            "Hunger termination bonus should equal the configured hunger_fail_penalty"
        )

    return {
        'delta_pellets': delta_pellets,
        'idle_ratio': idle_ratio,
        'survival_ratio': survival_ratio,
        'ghost_ring_distance': ghost_ring_distance,
        'vulnerability_fraction': vulnerability_fraction,
        'team_pressure': entry['ghost_team_pressure'],
        'proximity_drive': entry['ghost_proximity_drive'],
        'vulnerability_tax': entry['ghost_vulnerability_tax'],
        'ghost_total_reward': entry['ghost_total_reward'],
    }


def validate_observation_dimensions() -> dict:
    """Ensure raw observation builders still emit the documented tensor lengths."""

    pacman_agent = PacmanAgent()
    ghost_team = GhostTeam()
    game = PacmanGame()
    state = game.reset()

    pacman_repr = pacman_agent.get_state_repr(state)
    assert pacman_repr.shape[0] == PACMAN_STATE_DIM, (
        f"Pacman observation length changed: expected {PACMAN_STATE_DIM}, got {pacman_repr.shape[0]}"
    )

    ghost_reprs = []
    for idx in range(4):
        ghost_repr = ghost_team.get_ghost_state(state, idx)
        ghost_reprs.append(ghost_repr)
        assert ghost_repr.shape[0] == GHOST_STATE_DIM, (
            f"Ghost {idx} observation length changed: expected {GHOST_STATE_DIM}, got {ghost_repr.shape[0]}"
        )

    global_repr = ghost_team.get_global_state(state)
    assert global_repr.shape[0] == GLOBAL_STATE_DIM, (
        f"Global mixer state length changed: expected {GLOBAL_STATE_DIM}, got {global_repr.shape[0]}"
    )

    return {
        'pacman_dim': pacman_repr.shape[0],
        'ghost_dims': [vec.shape[0] for vec in ghost_reprs],
        'global_dim': global_repr.shape[0],
    }


def main() -> None:
    _set_global_seed()
    hunger_stats = run_hunger_enforcement_regression()
    ghost_stats = validate_ghost_reward_decomposition()
    observation_stats = validate_observation_dimensions()

    print('\n=== Hunger Enforcement Regression ===')
    for key, value in hunger_stats.items():
        print(f"  {key}: {value}")

    print('\n=== Ghost Reward Decomposition Regression ===')
    for key, value in ghost_stats.items():
        print(f"  {key}: {value}")

    print('\n=== Observation Dimension Regression ===')
    print(f"  Pacman vector length: {observation_stats['pacman_dim']}")
    print(f"  Ghost vector lengths: {observation_stats['ghost_dims']}")
    print(f"  Global state length: {observation_stats['global_dim']}")


if __name__ == "__main__":
    main()
