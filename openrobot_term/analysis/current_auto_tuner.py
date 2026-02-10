"""
Automatic Current Control Loop tuning.

Two-phase evaluation:
  Phase 1: Current step during acceleration (transient response)
  Phase 2: Steady-state current analysis during speed control
"""

import time
from dataclasses import dataclass, field
from typing import Optional
from collections import deque

import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal, QCoreApplication

from ..protocol.serial_transport import SerialTransport
from ..protocol.commands import (
    VescValues, build_get_values,
    build_set_current, build_set_rpm,
    build_set_mcconf_with_foc_cc,
)
from .current_metrics import CurrentMetrics, analyze_current_step, analyze_current_steady_state
from .llm_advisor import CurrentControlAdvisor, CurrentAnalysisResult, FOCCurrentGains


@dataclass
class CurrentTuningIteration:
    iteration: int = 0
    phase: str = ""  # "step", "steady", or "combined"
    gains: FOCCurrentGains = field(default_factory=FOCCurrentGains)
    step_metrics: Optional[CurrentMetrics] = None
    steady_metrics: Optional[CurrentMetrics] = None
    analysis: Optional[CurrentAnalysisResult] = None
    score: float = 0.0


class CurrentAutoTuner(QThread):
    """
    Runs current control auto-tuning in background thread.
    """

    status_update = pyqtSignal(str)
    iteration_complete = pyqtSignal(object)  # CurrentTuningIteration
    tuning_finished = pyqtSignal(str)
    data_collecting = pyqtSignal(float)  # progress 0-1
    metrics_ready = pyqtSignal(object)  # CurrentMetrics

    def __init__(
        self,
        transport: SerialTransport,
        advisor: CurrentControlAdvisor,
        target_current: float,
        initial_gains: FOCCurrentGains,
        original_mcconf: bytes,
        target_rpm_for_steady: float = 3000.0,
        max_iterations: int = 5,
        step_duration_s: float = 1.0,
        steady_duration_s: float = 3.0,
        sample_rate_hz: float = 50.0,
        target_score: float = 85.0,
    ):
        super().__init__()
        self._transport = transport
        self._advisor = advisor
        self._target_current = target_current
        self._current_gains = initial_gains
        self._original_mcconf = original_mcconf
        self._target_rpm = target_rpm_for_steady
        self._max_iter = max_iterations
        self._step_duration = step_duration_s
        self._steady_duration = steady_duration_s
        self._sample_rate = sample_rate_hz
        self._target_score = target_score
        self._running = False

        # Data buffers
        self._current_buf = []
        self._voltage_buf = []
        self._rpm_buf = []
        self._collecting = False

    def stop(self):
        self._running = False

    def on_values(self, v: VescValues):
        """Feed real-time data during collection."""
        if self._collecting:
            self._current_buf.append(v.avg_motor_current)
            self._voltage_buf.append(v.v_in)
            self._rpm_buf.append(v.rpm)

    def _sleep_with_events(self, seconds: float):
        """Sleep while processing Qt events."""
        end_time = time.time() + seconds
        while time.time() < end_time:
            QCoreApplication.processEvents()
            QThread.msleep(50)
            if not self._running:
                return

    def run(self):
        self._running = True
        history = []

        best_score = 0.0
        best_gains = self._current_gains
        best_iteration = 0

        # Countdown
        for countdown in [3, 2, 1]:
            if not self._running:
                self.tuning_finished.emit("Cancelled")
                return
            self.status_update.emit(f"Starting current control tuning in {countdown}...")
            self._sleep_with_events(1.0)

        for iteration in range(1, self._max_iter + 1):
            if not self._running:
                self._stop_motor()
                self.tuning_finished.emit("Cancelled by user")
                return

            # ========== PHASE 1: Current Step Test ==========
            self.status_update.emit(f"Iteration {iteration}/{self._max_iter} - Phase 1: Current Step Test")

            # Ensure motor stopped
            self._stop_motor()
            self._sleep_with_events(1.5)

            # Clear buffers
            self._current_buf.clear()
            self._voltage_buf.clear()
            self._rpm_buf.clear()

            # Start collecting and apply current step
            self._collecting = True
            self._transport.send_packet(build_set_current(self._target_current))

            # Collect during step response
            interval = 1.0 / self._sample_rate
            t0 = time.time()
            while time.time() - t0 < self._step_duration:
                if not self._running:
                    self._collecting = False
                    self._stop_motor()
                    self.tuning_finished.emit("Cancelled")
                    return
                # Keep sending current command (VESC timeout)
                self._transport.send_packet(build_set_current(self._target_current))
                self._transport.send_packet(build_get_values())
                progress = (time.time() - t0) / self._step_duration * 0.5  # 0-50%
                self.data_collecting.emit(progress)
                QCoreApplication.processEvents()
                QThread.msleep(int(interval * 1000))

            self._collecting = False
            self._stop_motor()
            self._sleep_with_events(1.0)

            # Analyze Phase 1
            if len(self._current_buf) < 10:
                self.status_update.emit(f"Phase 1: Insufficient data ({len(self._current_buf)} samples)")
                continue

            step_metrics = analyze_current_step(
                current_data=np.array(self._current_buf),
                target_current=self._target_current,
                sample_rate=self._sample_rate,
                voltage_data=np.array(self._voltage_buf) if self._voltage_buf else None,
            )

            self.status_update.emit(
                f"Phase 1 Result: Score={step_metrics.quality_score:.1f}, "
                f"Rise={step_metrics.rise_time_ms:.1f}ms, Overshoot={step_metrics.overshoot_pct:.1f}%"
            )
            self.metrics_ready.emit(step_metrics)

            # ========== PHASE 2: Steady-State During Speed Control ==========
            self.status_update.emit(f"Iteration {iteration}/{self._max_iter} - Phase 2: Steady-State Analysis")

            self._current_buf.clear()
            self._voltage_buf.clear()
            self._rpm_buf.clear()

            # Start speed control to reach target RPM
            self._transport.send_packet(build_set_rpm(int(self._target_rpm)))
            self._sleep_with_events(2.0)  # Wait for motor to reach speed

            # Collect steady-state data
            self._collecting = True
            t0 = time.time()
            while time.time() - t0 < self._steady_duration:
                if not self._running:
                    self._collecting = False
                    self._stop_motor()
                    self.tuning_finished.emit("Cancelled")
                    return
                self._transport.send_packet(build_set_rpm(int(self._target_rpm)))
                self._transport.send_packet(build_get_values())
                progress = 0.5 + (time.time() - t0) / self._steady_duration * 0.5  # 50-100%
                self.data_collecting.emit(progress)
                QCoreApplication.processEvents()
                QThread.msleep(int(interval * 1000))

            self._collecting = False
            self._stop_motor()
            self._sleep_with_events(1.5)

            # Analyze Phase 2
            if len(self._current_buf) < 10:
                self.status_update.emit(f"Phase 2: Insufficient data")
                continue

            steady_metrics = analyze_current_steady_state(
                current_data=np.array(self._current_buf),
                target_current=0,  # Use measured mean as reference
                sample_rate=self._sample_rate,
                rpm_data=np.array(self._rpm_buf) if self._rpm_buf else None,
                voltage_data=np.array(self._voltage_buf) if self._voltage_buf else None,
            )

            self.status_update.emit(
                f"Phase 2 Result: Score={steady_metrics.quality_score:.1f}, "
                f"Ripple={steady_metrics.current_ripple_pct:.2f}%, THD={steady_metrics.current_thd:.2f}%"
            )
            self.metrics_ready.emit(steady_metrics)

            # Combined score (weighted average)
            combined_score = 0.4 * step_metrics.quality_score + 0.6 * steady_metrics.quality_score
            self.status_update.emit(f"Iteration {iteration}: Combined Score = {combined_score:.1f}")

            # Track best
            if combined_score > best_score:
                best_score = combined_score
                best_gains = self._current_gains
                best_iteration = iteration

            # Get LLM recommendation
            self.status_update.emit(f"Iteration {iteration}: Consulting AI advisor...")

            # Build context
            context = self._build_context(history, step_metrics, steady_metrics)

            # Use step_metrics for LLM (has transient info)
            # But we could create a combined metrics object
            analysis = self._get_llm_recommendation(step_metrics, steady_metrics, context)

            # Check if target score already reached
            if combined_score >= self._target_score:
                result = CurrentTuningIteration(
                    iteration=iteration,
                    phase="combined",
                    gains=self._current_gains,
                    step_metrics=step_metrics,
                    steady_metrics=steady_metrics,
                    analysis=None,
                    score=combined_score,
                )
                history.append(result)
                self.iteration_complete.emit(result)

                self.status_update.emit(
                    f"[Target Reached] Score {combined_score:.1f} >= {self._target_score:.1f}. "
                    f"Current gains are already optimal - no changes needed."
                )
                self.tuning_finished.emit(
                    f"Target score reached on iteration {iteration}! "
                    f"Score: {combined_score:.1f}. Gains unchanged."
                )
                return

            # Store result
            result = CurrentTuningIteration(
                iteration=iteration,
                phase="combined",
                gains=self._current_gains,
                step_metrics=step_metrics,
                steady_metrics=steady_metrics,
                analysis=analysis,
                score=combined_score,
            )
            history.append(result)
            self.iteration_complete.emit(result)

            # Apply new gains if suggested
            if analysis and analysis.suggested_gains:
                new_kp = analysis.suggested_gains.kp
                new_ki = analysis.suggested_gains.ki
                # Clamp changes
                new_gains = self._clamp_gain_change(
                    self._current_gains,
                    FOCCurrentGains(kp=new_kp, ki=new_ki)
                )
                self._apply_gains(new_gains, f"Iteration {iteration}")
                self._current_gains = new_gains

            self._sleep_with_events(2.0)

        # Rollback to best if needed
        if history and history[-1].score < best_score:
            self.status_update.emit(f"Rolling back to best gains from iteration {best_iteration}")
            self._apply_gains(best_gains, "Rollback")
            self._current_gains = best_gains

            # Verification test
            verify_score = self._run_verification()
            self.tuning_finished.emit(
                f"Completed {self._max_iter} iterations. Best: {best_score:.1f} @ iter {best_iteration}. "
                f"Verification: {verify_score:.1f}"
            )
        else:
            self.tuning_finished.emit(
                f"Completed {self._max_iter} iterations. Best score: {best_score:.1f}"
            )

    def _build_context(self, history, step_metrics, steady_metrics) -> str:
        """Build context string for LLM."""
        lines = [
            "=== Current Control Test Results ===",
            f"Phase 1 (Step): Rise={step_metrics.rise_time_ms:.1f}ms, "
            f"Overshoot={step_metrics.overshoot_pct:.1f}%, Score={step_metrics.quality_score:.1f}",
            f"Phase 2 (Steady): Ripple={steady_metrics.current_ripple_pct:.2f}%, "
            f"THD={steady_metrics.current_thd:.2f}%, Score={steady_metrics.quality_score:.1f}",
            "",
        ]

        if history:
            lines.append("=== History ===")
            for h in history:
                lines.append(
                    f"Iter {h.iteration}: Score={h.score:.1f}, "
                    f"Gains(Kp={h.gains.kp:.6f}, Ki={h.gains.ki:.3f})"
                )

        return "\n".join(lines)

    def _get_llm_recommendation(self, step_metrics, steady_metrics, context) -> Optional[CurrentAnalysisResult]:
        """Get LLM recommendation for current gains."""
        try:
            analysis = self._advisor.analyze_and_recommend(
                step_metrics=step_metrics,
                steady_metrics=steady_metrics,
                current_gains=self._current_gains,
                additional_context=context,
            )
            return analysis
        except Exception as e:
            self.status_update.emit(f"LLM Error: {e}")
            return None

    def _clamp_gain_change(self, current: FOCCurrentGains, suggested: FOCCurrentGains) -> FOCCurrentGains:
        """Limit gain changes to 20% per iteration."""
        def clamp(old, new, limit=0.2):
            """Limit change to ±limit% of current value."""
            if old == 0:
                return new
            max_change = abs(old) * limit
            diff = new - old
            if abs(diff) > max_change:
                diff = max_change if diff > 0 else -max_change
            return old + diff

        return FOCCurrentGains(
            kp=clamp(current.kp, suggested.kp),
            ki=clamp(current.ki, suggested.ki),
        )

    def _apply_gains(self, gains: FOCCurrentGains, label: str):
        """Apply FOC current gains to VESC."""
        self.status_update.emit(f"{label}: Applying Kp={gains.kp:.6f}, Ki={gains.ki:.3f}")
        packet = build_set_mcconf_with_foc_cc(self._original_mcconf, gains.kp, gains.ki)
        self._transport.send_packet(packet)

    def _run_verification(self) -> float:
        """Run verification test after rollback."""
        self.status_update.emit("[Verification] Running confirmation test...")

        self._sleep_with_events(2.0)
        self._stop_motor()
        self._sleep_with_events(1.5)

        # Quick step test
        self._current_buf.clear()
        self._voltage_buf.clear()

        self._collecting = True
        self._transport.send_packet(build_set_current(self._target_current))

        interval = 1.0 / self._sample_rate
        t0 = time.time()
        while time.time() - t0 < self._step_duration:
            if not self._running:
                break
            self._transport.send_packet(build_set_current(self._target_current))
            self._transport.send_packet(build_get_values())
            QCoreApplication.processEvents()
            QThread.msleep(int(interval * 1000))

        self._collecting = False
        self._stop_motor()

        if len(self._current_buf) < 10:
            return 0.0

        metrics = analyze_current_step(
            np.array(self._current_buf),
            self._target_current,
            self._sample_rate
        )

        self.status_update.emit(f"[Verification] Score = {metrics.quality_score:.1f}")
        self.metrics_ready.emit(metrics)

        return metrics.quality_score

    def _stop_motor(self):
        """Stop motor."""
        self._transport.send_packet(build_set_current(0.0))
