#!/usr/bin/env python3
"""Regression tests for the telemetry buffering and GUI dispatcher plumbing."""

from __future__ import annotations

import random
import time
import unittest
from contextlib import contextmanager
from types import MethodType, SimpleNamespace
from typing import Dict, List, Optional
from unittest.mock import patch

import numpy as np

import telemetry as telemetry_module
from telemetry import (
    EpisodeTelemetry,
    TelemetryChannel,
    TelemetryCollector,
    TelemetryConfig,
    TelemetryDispatcher,
)
from trainer import Trainer


def seed_all(seed: int = 1337) -> None:
    random.seed(seed)
    np.random.seed(seed)


class DummyGame:
    width = 5
    height = 5
    max_steps = 3
    initial_pellet_override: Optional[int] = None
    initial_power_override: Optional[int] = None

    def __init__(self, custom_map_path=None, hunger_config=None) -> None:
        self.steps = 0
        self.ghost_released = [True] * 4
        self.ghost_in_house = [False] * 4
        self.termination_reason = None
        self._score = 0
        self._pellet_seed = (
            self.initial_pellet_override if self.initial_pellet_override is not None else 3
        )
        self._power_seed = (
            self.initial_power_override if self.initial_power_override is not None else 0
        )
        self.last_episode_remaining: Optional[int] = None
        self._initialize_board()

    def _initialize_board(self) -> None:
        pellet_total = max(0, int(self._pellet_seed))
        power_total = max(0, int(self._power_seed))
        self._pellets = [(0, idx) for idx in range(pellet_total)]
        self._power_pellets = [(1, idx) for idx in range(power_total)]
        self.power_pellet_positions = list(self._power_pellets)
        self.initial_pellet_total = len(self._pellets)
        self.initial_power_total = len(self._power_pellets)

    def get_state(self):
        return {
            'pellets': list(self._pellets),
            'power_pellets': list(self._power_pellets),
            'ghosts': [(i, i) for i in range(4)],
            'ghost_vulnerable': [False] * 4,
            'pacman': (self.steps % self.height, self.steps % self.width),
            'walls': [],
            'score': self._score,
            'steps': self.steps,
            'dimensions': {'height': self.height, 'width': self.width},
            'termination_reason': self.termination_reason,
            'initial_counts': {
                'pellets': self.initial_pellet_total,
                'power_pellets': self.initial_power_total,
            },
        }

    def step(self, pacman_action, ghost_actions):
        self.steps += 1
        reward = 10.0 if self._pellets else 0.0
        if self._pellets:
            self._pellets.pop()
            self._score += 10
        done = self.steps >= self.max_steps
        if done:
            self.termination_reason = 'TEST_DONE'
        next_state = self.get_state()
        return next_state, reward, done

    def reset(self):
        if hasattr(self, '_pellets'):
            current_remaining = len(self._pellets) + len(self._power_pellets)
            self.last_episode_remaining = current_remaining
        else:
            self.last_episode_remaining = None
        self.steps = 0
        self._score = 0
        self.termination_reason = None
        self._initialize_board()
        return self.get_state()

    def update_hunger_stats(self, stats):  # pragma: no cover - hook
        self._last_hunger_stats = stats

    def force_hunger_termination(self, reason):
        self.termination_reason = reason

    @classmethod
    def configure_initial_counts(cls, *, pellets: Optional[int] = None, power_pellets: Optional[int] = None) -> None:
        cls.initial_pellet_override = pellets
        cls.initial_power_override = power_pellets


class _AgentCore:
    def __init__(self, lr: float) -> None:
        self.optimizer = SimpleNamespace(param_groups=[{'lr': lr}])
        self.entropy_coef = 0.02

    def get_grad_stats(self):
        return {'mean': 0.0}


class DummyPacmanAgent:
    def __init__(self) -> None:
        self.agent = _AgentCore(lr=0.001)

    def get_state_repr(self, state):  # pragma: no cover - deterministic return
        return np.zeros(1, dtype=np.float32)

    def get_action(self, _state_repr):
        return 0

    def update(self, *args, **kwargs):  # pragma: no cover - noop for tests
        return None


class DummyQMIX:
    def __init__(self) -> None:
        self.optimizer = SimpleNamespace(param_groups=[{'lr': 0.002}])
        self.epsilon = 0.5
        self.epsilon_decay = 0.99

    def get_grad_stats(self):
        return {'mean': 0.0}


class DummyGhostTeam:
    def __init__(self) -> None:
        self.qmix = DummyQMIX()

    def get_actions(self, _state):
        return [0, 1, 2, 3]

    def update(self, *args, **kwargs):  # pragma: no cover - noop
        return None


@contextmanager
def patched_trainer_env():
    with patch('trainer.PacmanGame', DummyGame), patch('trainer.PacmanAgent', DummyPacmanAgent), patch('trainer.GhostTeam', DummyGhostTeam):
        yield


def build_test_trainer(capacity: int = 4, initial_counts: Optional[Dict[str, int]] = None) -> Trainer:
    config = TelemetryConfig(
        enable_collection=True,
        enable_dispatcher=False,
        channel_capacity=capacity,
        drop_oldest=True,
    )
    with patched_trainer_env():
        previous_pellet_override = DummyGame.initial_pellet_override
        previous_power_override = DummyGame.initial_power_override
        if initial_counts is not None:
            DummyGame.configure_initial_counts(
                pellets=initial_counts.get('pellets'),
                power_pellets=initial_counts.get('power_pellets'),
            )
        try:
            trainer = Trainer(n_envs=1, telemetry_config=config)
        finally:
            DummyGame.initial_pellet_override = previous_pellet_override
            DummyGame.initial_power_override = previous_power_override
    trainer.save_checkpoint = MethodType(lambda self, name='checkpoint': None, trainer)
    return trainer


def run_small_batch(trainer: Trainer, target_episodes: int = 2) -> None:
    guard = 0
    while trainer.episode < target_episodes and guard < 50:
        trainer.train_step()
        guard += 1


class TelemetryPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        seed_all()

    def test_collector_clamps_negative_pellets(self) -> None:
        config = TelemetryConfig(enable_collection=True, enable_dispatcher=False)
        collector = TelemetryCollector(config=config)
        telemetry_sample = EpisodeTelemetry(
            episode_index=1,
            env_count=1,
            avg_reward=0.0,
            last_reward=0.0,
            pacman_win_rate=0.0,
            ghost_win_rate=0.0,
            avg_length=1.0,
            avg_pellets=10.0,
            episode_length=1,
            pellets_collected=-5,
        )
        with self.assertLogs('telemetry', level='WARNING') as log_ctx:
            collector.record_episode(telemetry_sample)
        drained = collector.channel.drain()
        self.assertEqual(len(drained), 1)
        self.assertEqual(drained[0].pellets_collected, 0)
        self.assertTrue(any('pellets_collected' in entry for entry in log_ctx.output))

    def test_dispatcher_processes_clamped_pellets(self) -> None:
        config = TelemetryConfig(enable_collection=True, enable_dispatcher=True)
        collector = TelemetryCollector(config=config)
        collector.record_episode(
            EpisodeTelemetry(
                episode_index=2,
                env_count=1,
                avg_reward=0.0,
                last_reward=0.0,
                pacman_win_rate=0.0,
                ghost_win_rate=0.0,
                avg_length=2.0,
                avg_pellets=8.0,
                episode_length=2,
                pellets_collected=-10,
            )
        )
        dispatched: List[EpisodeTelemetry] = []
        dispatcher = TelemetryDispatcher(
            channel=collector.channel,
            on_telemetry_batch=lambda batch: dispatched.extend(batch),
            config=config,
            cpu_sampler=None,
        )
        dispatcher.tick()
        self.assertEqual(len(dispatched), 1)
        self.assertEqual(dispatched[0].pellets_collected, 0)

    def test_trainer_handles_large_initial_pellet_total(self) -> None:
        trainer = build_test_trainer(
            capacity=5,
            initial_counts={'pellets': 400, 'power_pellets': 4},
        )
        run_small_batch(trainer, target_episodes=1)
        self.assertTrue(trainer.pellets_collected)
        self.assertGreaterEqual(trainer.pellets_collected[-1], 0)
        samples = trainer.telemetry_collector.channel.drain()
        self.assertTrue(samples)
        sample = samples[-1]
        remaining = trainer.games[0].last_episode_remaining
        self.assertIsNotNone(remaining)
        initial_total = trainer.initial_pellet_totals[0]
        expected_collected = max(initial_total - int(remaining), 0)
        self.assertEqual(sample.pellets_collected, expected_collected)

    def test_trainer_emit_enqueues_episode_telemetry_fields(self) -> None:
        trainer = build_test_trainer(capacity=5)
        run_small_batch(trainer, target_episodes=2)
        channel = trainer.telemetry_collector.channel
        channel.drain()

        trainer.episode = 5
        trainer.episode_rewards = [10.0, 12.0, 14.0, 16.0, 18.0]
        trainer.episode_lengths = [20, 21, 22, 23, 24]
        trainer.pellets_collected = [100, 110, 120, 130, 140]
        trainer.pacman_wins = 2
        trainer.ghost_wins = 3
        hunger_metrics = {
            'hunger_meter': -12.5,
            'steps_since_progress': 6,
            'score_freeze_steps': 9,
            'unique_tiles': 4,
        }
        trainer._telemetry_total_steps = 120
        trainer._telemetry_last_sample_steps = 100
        trainer._telemetry_last_sample_time = time.time() - 1.0

        trainer._emit_episode_telemetry(
            episode_reward=42.0,
            episode_length=30,
            pellets_collected=180,
            hunger_metrics=hunger_metrics,
            hunger_reason='HUNGER_SCORE_FREEZE',
        )

        batch = channel.drain()
        self.assertEqual(len(batch), 1)
        telemetry_sample = batch[0]
        self.assertEqual(telemetry_sample.episode_index, 5)
        self.assertEqual(telemetry_sample.env_count, trainer.n_envs)
        self.assertAlmostEqual(telemetry_sample.last_reward, 42.0)
        self.assertGreaterEqual(telemetry_sample.sim_speed_fps, 0.0)
        self.assertEqual(
            telemetry_sample.custom_metrics['hunger_termination_reason'],
            'HUNGER_SCORE_FREEZE',
        )
        self.assertIn('ghost_epsilon', telemetry_sample.custom_metrics)

    def test_channel_drop_oldest_policy_preserves_latest_events(self) -> None:
        trainer = build_test_trainer(capacity=3)
        channel = trainer.telemetry_collector.channel
        channel.drain()
        trainer.episode_rewards = []
        trainer.episode_lengths = []
        trainer.pellets_collected = []
        trainer.pacman_wins = 0
        trainer.ghost_wins = 0

        hunger_metrics = {
            'hunger_meter': 0.0,
            'steps_since_progress': 1,
            'score_freeze_steps': 1,
            'unique_tiles': 1,
        }

        for idx in range(5):
            trainer.episode = idx
            trainer.episode_rewards.append(float(idx))
            trainer.episode_lengths.append(idx + 10)
            trainer.pellets_collected.append(50 + idx)
            trainer._emit_episode_telemetry(
                episode_reward=float(idx),
                episode_length=idx + 10,
                pellets_collected=100 + idx,
                hunger_metrics=hunger_metrics,
                hunger_reason=None,
            )

        self.assertEqual(len(channel), 3)
        survivors = [t.episode_index for t in channel.drain()]
        self.assertEqual(survivors, [2, 3, 4])

    def test_dispatcher_tick_batches_and_cpu_sampling(self) -> None:
        config = TelemetryConfig(enable_dispatcher=True)
        channel = TelemetryChannel(capacity=5)
        for idx in range(4):
            channel.push(
                EpisodeTelemetry(
                    episode_index=idx,
                    env_count=1,
                    avg_reward=0.0,
                    last_reward=float(idx),
                    pacman_win_rate=0.0,
                    ghost_win_rate=0.0,
                    avg_length=10.0,
                    avg_pellets=100.0,
                    episode_length=idx + 1,
                    pellets_collected=idx,
                    custom_metrics={'source': 'dispatcher-test'},
                )
            )

        dispatched_batches: List[List[EpisodeTelemetry]] = []
        target_cpu = 27.5

        def _capture_batch(batch):
            dispatched_batches.append(list(batch))

        dispatcher = TelemetryDispatcher(
            channel=channel,
            on_telemetry_batch=_capture_batch,
            config=config,
            cpu_sampler=lambda: target_cpu,
        )
        dispatcher.tick()

        self.assertEqual(len(dispatched_batches), 1)
        batch = dispatched_batches[0]
        self.assertEqual([t.episode_index for t in batch], [0, 1, 2, 3])
        self.assertTrue(all(t.cpu_percent == target_cpu for t in batch))

    def test_dispatcher_handles_missing_psutil(self) -> None:
        config = TelemetryConfig(enable_dispatcher=True)
        channel = TelemetryChannel(capacity=2)
        channel.push(
            EpisodeTelemetry(
                episode_index=0,
                env_count=1,
                avg_reward=0.0,
                last_reward=0.0,
                pacman_win_rate=0.0,
                ghost_win_rate=0.0,
                avg_length=5.0,
                avg_pellets=50.0,
                episode_length=5,
                pellets_collected=5,
                custom_metrics={'source': 'psutil-fallback'},
            )
        )
        dispatched: List[EpisodeTelemetry] = []
        original_psutil = telemetry_module.psutil
        telemetry_module.psutil = None
        try:
            dispatcher = TelemetryDispatcher(
                channel=channel,
                on_telemetry_batch=lambda batch: dispatched.extend(batch),
                config=config,
                cpu_sampler=None,
            )
            dispatcher.tick()
        finally:
            telemetry_module.psutil = original_psutil

        self.assertEqual(len(dispatched), 1)
        self.assertIsNone(dispatched[0].cpu_percent)


if __name__ == '__main__':
    unittest.main()
