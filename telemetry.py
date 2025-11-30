from __future__ import annotations

import json
import os
import threading
import time
import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Union

try:
    import psutil  # type: ignore
except ImportError:  # pragma: no cover - psutil is optional at runtime
    psutil = None


TelemetryMetric = Union[int, float, str]

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class EpisodeTelemetry:
    """Immutable snapshot of a single episode's aggregate metrics.

    Instances are produced inside the trainer threads and must remain read-only once
    enqueued so the GUI thread can safely consume them without additional locking.
    """

    episode_index: int
    env_count: int
    avg_reward: float
    last_reward: float
    pacman_win_rate: float
    ghost_win_rate: float
    avg_length: float
    avg_pellets: float
    episode_length: int
    pellets_collected: int
    sim_speed_fps: Optional[float] = None
    cpu_percent: Optional[float] = None
    batch_id: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    custom_metrics: Dict[str, TelemetryMetric] = field(default_factory=dict)


@dataclass(slots=True)
class TelemetryConfig:
    """Configuration shared between trainer- and GUI-side telemetry plumbing."""

    enable_collection: bool = True
    enable_dispatcher: bool = True
    channel_capacity: int = 512
    dispatch_interval_ms: int = 150
    cpu_sample_interval: float = 0.75
    enable_json_logging: bool = False
    json_log_path: Optional[str] = None
    drop_oldest: bool = True
    enable_verbose_logging: bool = False


class TelemetryChannel:
    """Thread-safe telemetry queue with drop-oldest semantics.

    Trainer threads push completed `EpisodeTelemetry` objects, while the GUI thread
    drains them via `TelemetryDispatcher`. A simple deque + lock provides predictable
    behaviour without cross-thread contention from Tk/Tkinter internals.
    """

    def __init__(self, capacity: int = 512, drop_oldest: bool = True) -> None:
        self.capacity = max(1, capacity)
        self.drop_oldest = drop_oldest
        self._buffer: List[EpisodeTelemetry] = []
        self._lock = threading.Lock()

    def push(self, telemetry: EpisodeTelemetry) -> None:
        with self._lock:
            if len(self._buffer) >= self.capacity:
                if self.drop_oldest:
                    self._buffer.pop(0)
                else:
                    return  # drop newest when instructed not to evict oldest
            self._buffer.append(telemetry)

    def drain(self, max_items: Optional[int] = None) -> List[EpisodeTelemetry]:
        with self._lock:
            if not self._buffer:
                return []
            if max_items is None or max_items >= len(self._buffer):
                payload = self._buffer[:]
                self._buffer.clear()
                return payload
            payload = self._buffer[:max_items]
            del self._buffer[:max_items]
            return payload

    def __len__(self) -> int:  # pragma: no cover - convenience only
        with self._lock:
            return len(self._buffer)


class TelemetryCollector:
    """Trainer-side helper that buffers per-episode telemetry snapshots."""

    def __init__(
        self,
        config: Optional[TelemetryConfig] = None,
        channel: Optional[TelemetryChannel] = None,
    ) -> None:
        self.config = config or TelemetryConfig()
        self.channel = channel or TelemetryChannel(
            capacity=self.config.channel_capacity,
            drop_oldest=self.config.drop_oldest,
        )
        self._active_batch_id: Optional[str] = None
        self._json_lock = threading.Lock()
        self._prime_json_path()

    def _prime_json_path(self) -> None:
        if not (self.config.enable_json_logging and self.config.json_log_path):
            return
        directory = os.path.dirname(self.config.json_log_path)
        if directory:
            os.makedirs(directory, exist_ok=True)

    def set_batch_id(self, batch_id: Optional[str]) -> None:
        """Assign the batch identifier propagated to every telemetry sample."""

        self._active_batch_id = batch_id

    def record_episode(
        self,
        telemetry: Union[EpisodeTelemetry, Dict[str, TelemetryMetric]],
        custom_metrics: Optional[Dict[str, TelemetryMetric]] = None,
    ) -> None:
        if not self.config.enable_collection:
            return

        if isinstance(telemetry, dict):
            telemetry = EpisodeTelemetry(**telemetry)  # type: ignore[arg-type]

        if telemetry.batch_id is None:
            telemetry.batch_id = self._active_batch_id
        if custom_metrics:
            telemetry.custom_metrics.update(custom_metrics)

        telemetry = self._sanitize_episode_payload(telemetry)
        self.channel.push(telemetry)
        if self.config.enable_json_logging and self.config.json_log_path:
            self._append_json(telemetry)

    def _append_json(self, telemetry: EpisodeTelemetry) -> None:
        payload = {
            'episode_index': telemetry.episode_index,
            'env_count': telemetry.env_count,
            'avg_reward': telemetry.avg_reward,
            'last_reward': telemetry.last_reward,
            'pacman_win_rate': telemetry.pacman_win_rate,
            'ghost_win_rate': telemetry.ghost_win_rate,
            'avg_length': telemetry.avg_length,
            'avg_pellets': telemetry.avg_pellets,
            'episode_length': telemetry.episode_length,
            'pellets_collected': telemetry.pellets_collected,
            'sim_speed_fps': telemetry.sim_speed_fps,
            'cpu_percent': telemetry.cpu_percent,
            'batch_id': telemetry.batch_id,
            'timestamp': telemetry.timestamp,
            'custom_metrics': telemetry.custom_metrics,
        }
        line = json.dumps(payload, separators=(',', ':'))
        with self._json_lock:
            with open(self.config.json_log_path, 'a', encoding='utf-8') as handle:
                handle.write(line + '\n')

    def _sanitize_episode_payload(self, telemetry: EpisodeTelemetry) -> EpisodeTelemetry:
        """Enforce non-negative pellet metrics before dispatching to consumers."""

        try:
            pellets_value = int(telemetry.pellets_collected)
        except (TypeError, ValueError):
            pellets_value = 0
        if pellets_value < 0:
            logger.warning(
                "Collector clamped negative pellets_collected",
                extra={
                    'episode_index': telemetry.episode_index,
                    'batch_id': telemetry.batch_id,
                }
            )
            pellets_value = 0
        telemetry.pellets_collected = pellets_value
        return telemetry


class TelemetryDispatcher:
    """GUI-side dispatcher that drains the channel on the Tk main thread."""

    def __init__(
        self,
        channel: TelemetryChannel,
        on_telemetry_batch: Callable[[Sequence[EpisodeTelemetry]], None],
        config: Optional[TelemetryConfig] = None,
        cpu_sampler: Optional[Callable[[], float]] = None,
    ) -> None:
        self.channel = channel
        self.config = config or TelemetryConfig()
        self._on_batch = on_telemetry_batch
        self._cpu_sampler = cpu_sampler or self._default_cpu_sampler()
        self._last_cpu_sample_time = 0.0
        self._cached_cpu: Optional[float] = None

    def _default_cpu_sampler(self) -> Optional[Callable[[], float]]:
        if psutil is None:
            return None
        process = psutil.Process(os.getpid())
        try:  # prime measurement to avoid returning 0.0 the first time
            process.cpu_percent(None)
        except Exception:  # pragma: no cover - platform specific
            pass
        return process.cpu_percent

    def _sample_cpu(self) -> Optional[float]:
        if self._cpu_sampler is None:
            return None
        now = time.time()
        if (
            self._cached_cpu is None
            or (now - self._last_cpu_sample_time) >= self.config.cpu_sample_interval
        ):
            try:
                self._cached_cpu = float(self._cpu_sampler())
                self._last_cpu_sample_time = now
            except Exception:
                self._cached_cpu = None
        return self._cached_cpu

    def tick(self) -> None:
        """Drain the telemetry buffer and forward snapshots to the GUI widgets."""

        if not self.config.enable_dispatcher:
            return
        batch = self.channel.drain()
        if not batch:
            return
        cpu_value = self._sample_cpu()
        if cpu_value is not None:
            for telemetry in batch:
                if telemetry.cpu_percent is None:
                    telemetry.cpu_percent = cpu_value
        if self._on_batch:
            self._on_batch(batch)


__all__ = [
    'EpisodeTelemetry',
    'TelemetryChannel',
    'TelemetryCollector',
    'TelemetryConfig',
    'TelemetryDispatcher',
]
