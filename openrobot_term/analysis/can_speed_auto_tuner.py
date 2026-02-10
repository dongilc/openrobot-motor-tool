"""
CAN Speed eRPM auto-tuning loop.

Cycle: stop motor → send eRPM step → collect speed data → analyze →
       LLM recommend → clamp change → write Speed PID via serial MCCONF → repeat.

Motor commands (0xA2 eRPM) go through CAN transport.
Speed PID writes go through serial transport (COMM_SET_MCCONF).
"""

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal, QCoreApplication

from ..protocol.can_transport import PcanTransport
from ..protocol.can_commands import (
    RmdStatus,
    build_speed_closed_loop,
    build_motor_off,
    build_read_motor_status_2,
)
from ..protocol.commands import build_set_mcconf_with_pid
from .signal_metrics import MotorMetrics, analyze_speed_control
from .llm_advisor import LLMAdvisor, PIDGains, AnalysisResult


@dataclass
class CanSpeedTuningIteration:
    iteration: int = 0
    pid: PIDGains = field(default_factory=PIDGains)
    metrics: Optional[MotorMetrics] = None
    analysis: Optional[AnalysisResult] = None
    score: float = 0.0


class CanSpeedAutoTuner(QThread):
    """Runs CAN speed eRPM auto-tuning in a background thread."""

    status_update = pyqtSignal(str)
    iteration_complete = pyqtSignal(object)   # CanSpeedTuningIteration
    tuning_finished = pyqtSignal(str)
    data_collecting = pyqtSignal(float)       # progress 0-1
    metrics_ready = pyqtSignal(object)        # MotorMetrics

    def __init__(
        self,
        can_transport: PcanTransport,
        advisor: LLMAdvisor,
        target_erpm: int,
        initial_pid: PIDGains,
        original_mcconf: bytes,
        max_iterations: int = 5,
        collect_duration_s: float = 3.0,
        poll_rate_hz: float = 100.0,
        target_score: float = 85.0,
        pid_change_limit: float = 0.5,
    ):
        super().__init__()
        self._can = can_transport
        self._advisor = advisor
        self._target_erpm = target_erpm
        self._current_pid = initial_pid
        self._original_mcconf = original_mcconf
        self._max_iter = max_iterations
        self._collect_s = collect_duration_s
        self._poll_rate = poll_rate_hz
        self._target_score = target_score
        self._pid_limit = pid_change_limit
        self._running = False

        # Data buffers
        self._speed_buf: list[float] = []
        self._torque_buf: list[float] = []
        self._time_buf: list[float] = []
        self._t0: Optional[float] = None
        self._collecting = False

    def stop(self):
        self._running = False

    def on_status(self, status: RmdStatus):
        """Feed real-time CAN status data during collection."""
        if not self._collecting:
            return
        self._speed_buf.append(float(status.speed_dps))
        self._torque_buf.append(float(status.torque_curr))
        if self._t0 is None:
            self._t0 = time.time()
        self._time_buf.append(time.time() - self._t0)

    def run(self):
        self._running = True
        history: list[CanSpeedTuningIteration] = []
        best_score = 0.0
        best_pid = self._current_pid
        best_iteration = 0
        decline_count = 0

        # Countdown
        for c in [3, 2, 1]:
            if not self._running:
                self.tuning_finished.emit("Cancelled by user")
                return
            self.status_update.emit(f"Starting in {c}... (Motor will spin)")
            self._sleep(1.0)

        for iteration in range(1, self._max_iter + 1):
            if not self._running:
                self._stop_motor()
                self.tuning_finished.emit("Cancelled by user")
                return

            # Phase 0: Stop motor and wait for coast-down
            self.status_update.emit(
                f"Iter {iteration}/{self._max_iter}: Stopping motor for step response..."
            )
            self._stop_motor()
            self._sleep(2.0)

            # Phase 1: Clear buffers, start polling, collect baseline
            self._clear_buffers()
            self._collecting = True

            # Pre-read baseline (poll until 3+ samples)
            for _ in range(50):
                if not self._running:
                    self._collecting = False
                    self._stop_motor()
                    self.tuning_finished.emit("Cancelled by user")
                    return
                self._can.send_frame(build_read_motor_status_2())
                QCoreApplication.processEvents()
                QThread.msleep(10)
                if len(self._speed_buf) >= 3:
                    break

            # Clear buffers after baseline — clean data from step start
            self._clear_buffers()

            self.status_update.emit(
                f"Iter {iteration}/{self._max_iter}: Step to {self._target_erpm} eRPM - collecting..."
            )

            # Start collection + send eRPM command
            self._collecting = True

            # Poll at configured rate, keep re-sending speed command every 50ms
            interval_ms = int(1000.0 / self._poll_rate)
            cmd_interval_ms = 50
            t0 = time.time()
            last_cmd_time = 0.0

            # Send initial eRPM command
            self._can.send_frame(
                build_speed_closed_loop(self._target_erpm, mode=1)
            )

            while time.time() - t0 < self._collect_s:
                if not self._running:
                    self._collecting = False
                    self._stop_motor()
                    self.tuning_finished.emit("Cancelled by user")
                    return

                now = time.time()

                # Re-send speed command every 50ms to prevent VESC timeout
                if (now - last_cmd_time) * 1000 >= cmd_interval_ms:
                    self._can.send_frame(
                        build_speed_closed_loop(self._target_erpm, mode=1)
                    )
                    last_cmd_time = now

                # Poll motor status for data
                self._can.send_frame(build_read_motor_status_2())

                progress = (now - t0) / self._collect_s
                self.data_collecting.emit(min(1.0, progress))
                QCoreApplication.processEvents()
                QThread.msleep(interval_ms)

            self._collecting = False
            self.data_collecting.emit(1.0)

            # Stop motor
            self._stop_motor()
            self._sleep(1.5)

            if len(self._speed_buf) < 10:
                self.status_update.emit(
                    f"Insufficient data ({len(self._speed_buf)} samples). Check CAN connection."
                )
                self.tuning_finished.emit("Insufficient data")
                return

            # Phase 2: Analyze
            self.status_update.emit(
                f"Iter {iteration}: Analyzing ({len(self._speed_buf)} samples)..."
            )

            actual_rate = len(self._speed_buf) / self._collect_s
            speed_data = np.array(self._speed_buf)
            torque_data = np.array(self._torque_buf) if self._torque_buf else None

            # Use last 20% average as effective target
            tail_n = max(5, len(speed_data) // 5)
            effective_target = float(np.mean(speed_data[-tail_n:]))

            metrics = analyze_speed_control(
                rpm_data=speed_data,
                target_rpm=effective_target,
                sample_rate=actual_rate,
                current_data=torque_data,
            )
            metrics.target_rpm = effective_target

            current_score = metrics.quality_score
            self.metrics_ready.emit(metrics)

            # Track best
            if current_score > best_score:
                best_score = current_score
                best_pid = PIDGains(
                    kp=self._current_pid.kp,
                    ki=self._current_pid.ki,
                    kd=self._current_pid.kd,
                    kd_filter=self._current_pid.kd_filter,
                    ramp_erpms_s=self._current_pid.ramp_erpms_s,
                )
                best_iteration = iteration
                decline_count = 0
            else:
                decline_count += 1

            # Log
            trend = ""
            if history:
                prev = history[-1].score
                if current_score > prev:
                    trend = " \u2191 improved"
                elif current_score < prev:
                    trend = " \u2193 declined"
                else:
                    trend = " \u2192 same"

            self.status_update.emit(
                f"Iter {iteration}: Score = {current_score:.1f}{trend} "
                f"(Best: {best_score:.1f} @ iter {best_iteration})"
            )
            self.status_update.emit(
                f"  Ripple={metrics.rpm_ripple_pct:.2f}% SS_Err={metrics.steady_state_error_pct:.2f}% "
                f"Rise={metrics.rise_time:.3f}s Settle={metrics.settling_time:.3f}s "
                f"Overshoot={metrics.overshoot_pct:.1f}%"
            )

            # Check target reached
            if current_score >= self._target_score:
                result = CanSpeedTuningIteration(
                    iteration=iteration, pid=self._current_pid,
                    metrics=metrics, score=current_score,
                )
                history.append(result)
                self.iteration_complete.emit(result)
                self.tuning_finished.emit(
                    f"Target reached at iteration {iteration}! Score: {current_score:.1f}"
                )
                return

            # Early stop on 2+ consecutive declines after iteration 3
            if decline_count >= 2 and iteration >= 3:
                self.status_update.emit(
                    f"[Early Stop] Score declined {decline_count} times. "
                    f"Rolling back to best PID (iter {best_iteration})..."
                )
                self._apply_pid(best_pid, "Rollback to best")
                self._current_pid = best_pid

                result = CanSpeedTuningIteration(
                    iteration=iteration, pid=self._current_pid,
                    metrics=metrics, score=current_score,
                )
                history.append(result)
                self.iteration_complete.emit(result)
                self.tuning_finished.emit(
                    f"Early stopped (declining). Best: {best_score:.1f} @ iter {best_iteration}."
                )
                return

            # Phase 3: LLM recommendation
            self.status_update.emit(f"Iter {iteration}: Consulting AI advisor...")
            context = self._build_context(history, metrics)
            analysis = self._advisor.analyze_and_recommend(
                metrics=metrics,
                current_pid=self._current_pid,
                additional_context=context,
            )

            # Phase 4: Clamp and apply new PID
            if analysis.suggested_pid:
                new_pid = self._clamp_change(self._current_pid, analysis.suggested_pid)
            else:
                new_pid = self._current_pid

            result = CanSpeedTuningIteration(
                iteration=iteration, pid=self._current_pid,
                metrics=metrics, analysis=analysis, score=current_score,
            )
            history.append(result)
            self.iteration_complete.emit(result)

            self._apply_pid(new_pid, f"Iter {iteration}")
            self._current_pid = new_pid

            # Wait for VESC to apply new MCCONF
            self.status_update.emit(f"Iter {iteration}: Waiting 2s for VESC to apply new PID...")
            self._sleep(2.0)

        # Final: rollback if needed
        if history and history[-1].score < best_score:
            self.status_update.emit(
                f"Final score ({history[-1].score:.1f}) < Best ({best_score:.1f}). "
                f"Rolling back to best PID (iter {best_iteration})..."
            )
            self._apply_pid(best_pid, "Final rollback")
            self._current_pid = best_pid

        self.tuning_finished.emit(
            f"Completed {self._max_iter} iterations. "
            f"Best: {best_score:.1f} @ iter {best_iteration}."
        )

    # ── Helpers ─────────────────────────────────────────────────────

    def _stop_motor(self):
        self._can.send_frame(build_motor_off())

    def _apply_pid(self, pid: PIDGains, label: str):
        """Apply speed PID to VESC via CAN EID MCCONF."""
        self.status_update.emit(
            f"  {label}: Kp={pid.kp:.6f} Ki={pid.ki:.6f} Kd={pid.kd:.6f} "
            f"Kd_flt={pid.kd_filter:.4f} Ramp={pid.ramp_erpms_s:.0f}"
        )
        packet = build_set_mcconf_with_pid(
            self._original_mcconf,
            pid.kp, pid.ki, pid.kd, pid.kd_filter,
            position_mode=False,
            ramp_erpms_s=pid.ramp_erpms_s,
        )
        self._can.send_vesc_to_target(packet)

    def _clear_buffers(self):
        self._speed_buf.clear()
        self._torque_buf.clear()
        self._time_buf.clear()
        self._t0 = None

    def _sleep(self, seconds: float):
        end = time.time() + seconds
        while time.time() < end:
            if not self._running:
                return
            QCoreApplication.processEvents()
            QThread.msleep(50)

    def _build_context(self, history: list[CanSpeedTuningIteration], metrics) -> str:
        lines = [
            f"Control Mode: CAN eRPM Speed Control (0xA2 mode=1)",
            f"Target Speed: {self._target_erpm} eRPM",
            f"VESC Speed PID — adjustable via COMM_SET_MCCONF",
        ]
        if history:
            lines.append("=== Score History ===")
            for h in history:
                lines.append(
                    f"Iter {h.iteration}: Score={h.score:.1f}, "
                    f"PID(Kp={h.pid.kp:.6f}, Ki={h.pid.ki:.6f}, Kd={h.pid.kd:.6f}, "
                    f"Kd_flt={h.pid.kd_filter:.4f}, Ramp={h.pid.ramp_erpms_s:.0f})"
                )
            if len(history) >= 2:
                scores = [h.score for h in history[-3:]]
                if all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1)):
                    lines.append("WARNING: Scores declining! Reverse PID change direction.")
                elif all(scores[i] <= scores[i + 1] for i in range(len(scores) - 1)):
                    lines.append("Scores improving. Continue current direction.")
            best = max(history, key=lambda h: h.score)
            lines.append(f"Best score so far: {best.score:.1f} at iteration {best.iteration}")
        lines.append(
            f"Current PID: Kp={self._current_pid.kp:.6f}, Ki={self._current_pid.ki:.6f}, "
            f"Kd={self._current_pid.kd:.6f}, Kd_flt={self._current_pid.kd_filter:.4f}, "
            f"Ramp={self._current_pid.ramp_erpms_s:.0f}"
        )
        return "\n".join(lines)

    def _clamp_change(self, current: PIDGains, suggested: PIDGains) -> PIDGains:
        def clamp(old, new, limit):
            if old == 0:
                return new
            max_change = abs(old) * limit
            diff = new - old
            if abs(diff) > max_change:
                diff = max_change if diff > 0 else -max_change
            return old + diff

        return PIDGains(
            kp=clamp(current.kp, suggested.kp, self._pid_limit),
            ki=clamp(current.ki, suggested.ki, self._pid_limit),
            kd=clamp(current.kd, suggested.kd, self._pid_limit),
            kd_filter=clamp(current.kd_filter, suggested.kd_filter, self._pid_limit),
            ramp_erpms_s=clamp(current.ramp_erpms_s, suggested.ramp_erpms_s, self._pid_limit),
        )
