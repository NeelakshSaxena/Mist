"""
src/core/p5_profiler.py  –  Phase 5 Profiling Instrumentation

Measures per-stage runtime, P4 call counts, candidate rejection reasons,
RS decode attempts, and geometry convergence statistics.

Usage:
    with P5Profile() as prof:
        result = estimate_geometry(image, key)
        # prof is populated with metrics

    print(prof.summary())
"""

from __future__ import annotations

import time
import threading
from dataclasses import dataclass, field
from typing import Optional


# Thread-local storage for the active profiler
_local = threading.local()


def get_active_profile() -> Optional["P5Profile"]:
    """Return the currently active P5Profile, or None."""
    return getattr(_local, "active_profile", None)


@dataclass
class P5Profile:
    """
    Profiling data for a single Phase 5 detection run.

    Populated by instrumentation hooks in wm_engine_p5.py.
    """
    # ── Per-stage runtime (seconds) ──────────────────────────────────
    stage_times: dict[str, float] = field(default_factory=dict)

    # ── P4 call tracking ─────────────────────────────────────────────
    p4_call_count: int = 0
    p4_call_times: list[float] = field(default_factory=list)

    # ── Candidate tracking ───────────────────────────────────────────
    candidates_generated: int = 0
    candidates_after_clustering: int = 0
    candidates_evaluated: int = 0
    candidates_promoted: int = 0

    # ── Rejection reasons ────────────────────────────────────────────
    rejection_reasons: dict[str, int] = field(default_factory=lambda: {
        "low_canary": 0,
        "geometry_bounds": 0,
        "crc_no_improvement": 0,
        "unstable_refinement": 0,
        "duplicate_cluster": 0,
        "image_too_small": 0,
        "runtime_abort": 0,
    })

    # ── RS decode tracking ───────────────────────────────────────────
    rs_decode_attempts: int = 0
    rs_decode_successes: int = 0

    # ── Geometry convergence ─────────────────────────────────────────
    geometry_estimates: list[dict] = field(default_factory=list)
    best_crc_history: list[int] = field(default_factory=list)

    # ── Confidence components ────────────────────────────────────────
    confidence_components: dict[str, float] = field(default_factory=dict)

    # ── Total runtime ────────────────────────────────────────────────
    total_time: float = 0.0
    _start_time: float = 0.0

    def __enter__(self):
        self._start_time = time.perf_counter()
        _local.active_profile = self
        return self

    def __exit__(self, *exc):
        self.total_time = time.perf_counter() - self._start_time
        _local.active_profile = None
        return False

    def record_p4_call(self, duration: float):
        """Record a P4 scoring call."""
        self.p4_call_count += 1
        self.p4_call_times.append(duration)

    def record_rejection(self, reason: str):
        """Record a candidate rejection with reason."""
        self.rejection_reasons[reason] = self.rejection_reasons.get(reason, 0) + 1

    def record_geometry(self, angle: float, scale: float, crc: int, method: str):
        """Record a geometry estimate."""
        self.geometry_estimates.append({
            "angle": angle, "scale": scale, "crc": crc, "method": method,
        })

    def stage_timer(self, name: str) -> "_StageTimer":
        """Context manager for timing a stage."""
        return _StageTimer(self, name)

    @property
    def p4_total_time(self) -> float:
        """Total time spent in P4 calls."""
        return sum(self.p4_call_times)

    def summary(self) -> str:
        """Human-readable profiling summary."""
        lines = [
            "─── Phase 5 Profile ───",
            f"  Total time:           {self.total_time:.3f}s",
            f"  P4 calls:             {self.p4_call_count} ({self.p4_total_time:.3f}s)",
            f"  Candidates generated: {self.candidates_generated}",
            f"  After clustering:     {self.candidates_after_clustering}",
            f"  Evaluated (P4):       {self.candidates_evaluated}",
            f"  Promoted:             {self.candidates_promoted}",
            f"  RS decode attempts:   {self.rs_decode_attempts}",
            f"  RS decode successes:  {self.rs_decode_successes}",
        ]

        if self.stage_times:
            lines.append("  ── Stage times ──")
            for name, dt in sorted(self.stage_times.items()):
                lines.append(f"    {name:25s} {dt:.3f}s")

        if any(v > 0 for v in self.rejection_reasons.values()):
            lines.append("  ── Rejections ──")
            for reason, count in sorted(self.rejection_reasons.items()):
                if count > 0:
                    lines.append(f"    {reason:25s} {count}")

        if self.confidence_components:
            lines.append("  ── Confidence components ──")
            for name, val in sorted(self.confidence_components.items()):
                lines.append(f"    {name:25s} {val:.4f}")

        if self.geometry_estimates:
            lines.append("  ── Geometry history ──")
            for g in self.geometry_estimates[-5:]:  # last 5
                lines.append(
                    f"    {g['angle']:+6.1f}° × {g['scale']:.2f}× "
                    f"crc={g['crc']} ({g['method']})"
                )

        lines.append("───────────────────────")
        return "\n".join(lines)


class _StageTimer:
    """Context manager for timing a named stage."""
    def __init__(self, profile: P5Profile, name: str):
        self.profile = profile
        self.name = name
        self.start = 0.0

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        elapsed = time.perf_counter() - self.start
        self.profile.stage_times[self.name] = elapsed
        return False
