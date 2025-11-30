from __future__ import annotations

from typing import Callable, Dict, List, Optional

from telemetry import (
    EpisodeTelemetry,
    TelemetryCollector,
    TelemetryConfig,
    TelemetryDispatcher,
)


class TelemetrySummaryTracker:
    """Utility that mirrors GUI batching to surface telemetry statistics."""

    def __init__(
        self,
        config: Optional[TelemetryConfig] = None,
        flush_every: int = 2,
        cpu_sampler: Optional[Callable[[], float]] = None,
    ) -> None:
        self.config = config or TelemetryConfig(
            enable_collection=True,
            enable_dispatcher=True,
            channel_capacity=32,
            drop_oldest=True,
        )
        self.collector = TelemetryCollector(config=self.config)
        self.flush_every = max(1, flush_every)
        self._stats: Dict[str, float] = {
            'produced': 0,
            'dispatched': 0,
            'flushes': 0,
        }
        self._cpu_samples: List[float] = []
        self.dispatcher = TelemetryDispatcher(
            channel=self.collector.channel,
            on_telemetry_batch=self._on_batch,
            config=self.config,
            cpu_sampler=cpu_sampler,
        )

    def _on_batch(self, batch) -> None:
        if not batch:
            return
        self._stats['flushes'] += 1
        self._stats['dispatched'] += len(batch)
        for telemetry in batch:
            if telemetry.cpu_percent is not None:
                self._cpu_samples.append(float(telemetry.cpu_percent))

    def record(self, telemetry: EpisodeTelemetry) -> None:
        self._stats['produced'] += 1
        self.collector.record_episode(telemetry)
        if self._stats['produced'] % self.flush_every == 0:
            self.dispatcher.tick()

    def finalize(self) -> None:
        """Flush any buffered telemetry and finalize statistics."""

        self.dispatcher.tick()

    def summary(self) -> Dict[str, Optional[float]]:
        cpu_samples = self._cpu_samples
        avg_cpu = sum(cpu_samples) / len(cpu_samples) if cpu_samples else None
        return {
            'produced': self._stats['produced'],
            'dispatched': self._stats['dispatched'],
            'flushes': self._stats['flushes'],
            'avg_cpu': avg_cpu,
            'channel_capacity': self.config.channel_capacity,
        }
