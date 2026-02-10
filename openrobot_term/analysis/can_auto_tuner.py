"""
CAN/RMD Position PID auto-tuning loop.

Cycle: return to start → send position step → collect data → analyze →
       LLM recommend → clamp change → write PID to RAM → repeat.
"""

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal, QCoreApplication

from ..protocol.can_transport import PcanTransport
from ..protocol.can_commands import (
    RmdStatus,
    build_position_closed_loop_1,
    build_set_multiturn_position,
    build_motor_stop,
    build_read_motor_status_2,
    build_write_pid_to_ram,
)
from .can_position_metrics import CanPositionMetrics, analyze_position_step
from .llm_advisor import PIDGains, AnalysisResult


@dataclass
class RmdPidGains:
    """RMD motor PID gains (3-parameter)."""
    kp: float = 0.0
    ki: float = 0.0
    kd: float = 0.0


@dataclass
class CanTuningIteration:
    iteration: int = 0
    pid: RmdPidGains = field(default_factory=RmdPidGains)
    metrics: Optional[CanPositionMetrics] = None
    analysis: Optional[AnalysisResult] = None
    score: float = 0.0


class CanAutoTuner(QThread):
    """Runs CAN/RMD position PID auto-tuning in a background thread."""

    status_update = pyqtSignal(str)
    iteration_complete = pyqtSignal(object)   # CanTuningIteration
    tuning_finished = pyqtSignal(str)
    data_collecting = pyqtSignal(float)       # progress 0-1
    metrics_ready = pyqtSignal(object)        # CanPositionMetrics

    def __init__(
        self,
        transport: PcanTransport,
        advisor,
        target_pos_deg: float,
        initial_pid: RmdPidGains,
        use_multiturn_cmd: bool = False,
        dps_limit: int = 500,
        max_iterations: int = 5,
        collect_duration_s: float = 3.0,
        poll_rate_hz: float = 100.0,
        target_score: float = 85.0,
        pid_change_limit: float = 0.3,
        return_pos_deg: float = 0.0,
    ):
        super().__init__()
        self._transport = transport
        self._advisor = advisor
        self._target_pos = target_pos_deg
        self._current_pid = initial_pid
        self._use_multiturn = use_multiturn_cmd
        self._dps_limit = dps_limit
        self._max_iter = max_iterations
        self._collect_s = collect_duration_s
        self._poll_rate = poll_rate_hz
        self._target_score = target_score
        self._pid_limit = pid_change_limit
        self._return_pos = return_pos_deg
        self._running = False

        # Data buffers
        self._time_buf: list[float] = []
        self._pos_buf: list[float] = []
        self._torque_buf: list[float] = []
        self._collecting = False

        # Encoder unwrap state
        self._prev_enc_pos: Optional[float] = None
        self._enc_offset = 0.0

    def stop(self):
        self._running = False

    def on_status(self, status: RmdStatus):
        """Feed real-time CAN status data during collection."""
        if not self._collecting:
            return

        # Unwrap single-turn encoder (0-360) to continuous angle
        raw = status.enc_pos
        if self._prev_enc_pos is not None:
            delta = raw - self._prev_enc_pos
            if delta < -180.0:
                self._enc_offset += 360.0
            elif delta > 180.0:
                self._enc_offset -= 360.0
        self._prev_enc_pos = raw
        unwrapped = raw + self._enc_offset

        self._pos_buf.append(unwrapped)
        self._torque_buf.append(status.torque_curr)
        self._time_buf.append(time.time())

    def run(self):
        self._running = True
        history: list[CanTuningIteration] = []
        best_score = 0.0
        best_pid = self._current_pid
        best_iteration = 0
        decline_count = 0

        # Countdown
        for c in [3, 2, 1]:
            if not self._running:
                self.tuning_finished.emit("Cancelled by user")
                return
            self.status_update.emit(f"Starting in {c}... (Motor will move)")
            self._sleep(1.0)

        for iteration in range(1, self._max_iter + 1):
            if not self._running:
                self._send_stop()
                self.tuning_finished.emit("Cancelled by user")
                return

            # Phase 0: Return to start position (always DPS-limited for safety)
            self.status_update.emit(
                f"Iter {iteration}/{self._max_iter}: Returning to {self._return_pos:.1f} deg (safe)..."
            )
            self._safe_move(self._return_pos)
            self._sleep(2.0)
            self._send_stop()
            self._sleep(1.0)

            # Phase 1: Clear buffers, record initial position, step + collect
            self._clear_buffers()

            # Pre-read: poll until we get at least 1 sample (up to 500ms)
            self._collecting = True
            for _ in range(50):
                if not self._running:
                    self._collecting = False
                    self._send_stop()
                    self.tuning_finished.emit("Cancelled by user")
                    return
                self._transport.send_frame(build_read_motor_status_2())
                QCoreApplication.processEvents()
                QThread.msleep(10)
                if self._pos_buf:
                    break
            initial_pos = self._pos_buf[-1] if self._pos_buf else self._return_pos
            self._clear_buffers()

            self.status_update.emit(
                f"Iter {iteration}/{self._max_iter}: Step to {self._target_pos:.1f} deg - collecting..."
            )

            # Start data collection, then send step command
            self._collecting = True

            # Capture a few baseline samples before the step
            for _ in range(30):
                if not self._running:
                    self._collecting = False
                    self._send_stop()
                    self.tuning_finished.emit("Cancelled by user")
                    return
                self._transport.send_frame(build_read_motor_status_2())
                QCoreApplication.processEvents()
                QThread.msleep(10)
                if len(self._pos_buf) >= 3:
                    break

            self._send_position(self._target_pos)

            # Poll status at configured rate
            interval_ms = int(1000.0 / self._poll_rate)
            t0 = time.time()
            while time.time() - t0 < self._collect_s:
                if not self._running:
                    self._collecting = False
                    self._send_stop()
                    self.tuning_finished.emit("Cancelled by user")
                    return
                self._transport.send_frame(build_read_motor_status_2())
                progress = (time.time() - t0) / self._collect_s
                self.data_collecting.emit(min(1.0, progress))
                QCoreApplication.processEvents()
                QThread.msleep(interval_ms)

            self._collecting = False
            self.data_collecting.emit(1.0)

            if len(self._pos_buf) < 10:
                self.status_update.emit(
                    f"Insufficient data ({len(self._pos_buf)} samples). Check CAN connection."
                )
                self.tuning_finished.emit("Insufficient data")
                return

            # Phase 2: Analyze
            self.status_update.emit(f"Iter {iteration}: Analyzing ({len(self._pos_buf)} samples)...")

            actual_rate = len(self._pos_buf) / self._collect_s

            # Normalize encoder frame → commanded frame (±360° near 0°/360°)
            # Use first collected sample as fallback if pre-read failed
            ref_pos = initial_pos if initial_pos != self._return_pos else self._pos_buf[0]
            frame_offset = round((ref_pos - self._return_pos) / 360.0) * 360.0
            pos_data = np.array(self._pos_buf) - frame_offset

            metrics = analyze_position_step(
                pos_data=pos_data,
                target_deg=self._target_pos,
                sample_rate=actual_rate,
                torque_data=np.array(self._torque_buf) if self._torque_buf else None,
                initial_pos=initial_pos - frame_offset,
            )

            current_score = metrics.quality_score
            self.metrics_ready.emit(metrics)

            # Track best
            if current_score > best_score:
                best_score = current_score
                best_pid = RmdPidGains(
                    kp=self._current_pid.kp,
                    ki=self._current_pid.ki,
                    kd=self._current_pid.kd,
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
                    trend = " ↑ improved"
                elif current_score < prev:
                    trend = " ↓ declined"
                else:
                    trend = " → same"

            self.status_update.emit(
                f"Iter {iteration}: Score = {current_score:.1f}{trend} "
                f"(Best: {best_score:.1f} @ iter {best_iteration})"
            )
            self.status_update.emit(
                f"  Ripple={metrics.pos_ripple_pct:.2f}% SS_Err={metrics.steady_state_error_pct:.2f}% "
                f"Rise={metrics.rise_time_s:.3f}s Settle={metrics.settling_time_s:.3f}s "
                f"Overshoot={metrics.overshoot_pct:.1f}%"
            )

            # Check target reached
            if current_score >= self._target_score:
                result = CanTuningIteration(
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
                self._write_pid(best_pid, "Rollback to best")
                self._current_pid = best_pid

                result = CanTuningIteration(
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
            pid_for_llm = PIDGains(
                kp=self._current_pid.kp,
                ki=self._current_pid.ki,
                kd=self._current_pid.kd,
            )
            analysis = self._advisor.analyze_and_recommend(
                metrics=metrics,
                current_pid=pid_for_llm,
                additional_context=context,
            )

            # Phase 4: Clamp and apply new PID
            if analysis.suggested_pid:
                suggested = RmdPidGains(
                    kp=analysis.suggested_pid.kp,
                    ki=analysis.suggested_pid.ki,
                    kd=analysis.suggested_pid.kd,
                )
                new_pid = self._clamp_change(self._current_pid, suggested)
            else:
                new_pid = self._current_pid

            result = CanTuningIteration(
                iteration=iteration, pid=self._current_pid,
                metrics=metrics, analysis=analysis, score=current_score,
            )
            history.append(result)
            self.iteration_complete.emit(result)

            self._write_pid(new_pid, f"Iter {iteration}")
            self._current_pid = new_pid
            self._sleep(1.0)

        # Final: rollback if needed
        if history and history[-1].score < best_score:
            self.status_update.emit(
                f"Final score ({history[-1].score:.1f}) < Best ({best_score:.1f}). "
                f"Rolling back to best PID (iter {best_iteration})..."
            )
            self._write_pid(best_pid, "Final rollback")
            self._current_pid = best_pid

        self.tuning_finished.emit(
            f"Completed {self._max_iter} iterations. "
            f"Best: {best_score:.1f} @ iter {best_iteration}."
        )

    # ── Helpers ─────────────────────────────────────────────────────

    def _safe_move(self, deg: float):
        """Move to position using DPS-limited command (1000 DPS). Always safe."""
        self._transport.send_frame(
            build_set_multiturn_position(1000, deg)
        )

    def _send_position(self, deg: float):
        """Send position step command (for data collection phase)."""
        if self._use_multiturn:
            self._transport.send_frame(
                build_set_multiturn_position(self._dps_limit, deg)
            )
        else:
            self._transport.send_frame(build_position_closed_loop_1(deg))

    def _send_stop(self):
        self._transport.send_frame(build_motor_stop())

    def _write_pid(self, pid: RmdPidGains, label: str):
        self.status_update.emit(
            f"  {label}: Kp={pid.kp:.5f} Ki={pid.ki:.5f} Kd={pid.kd:.5f}"
        )
        self._transport.send_frame(build_write_pid_to_ram(pid.kp, pid.ki, pid.kd))

    def _clear_buffers(self):
        self._time_buf.clear()
        self._pos_buf.clear()
        self._torque_buf.clear()
        self._prev_enc_pos = None
        self._enc_offset = 0.0

    def _sleep(self, seconds: float):
        end = time.time() + seconds
        while time.time() < end:
            if not self._running:
                return
            QCoreApplication.processEvents()
            QThread.msleep(50)

    def _build_context(self, history: list[CanTuningIteration], metrics) -> str:
        lines = [
            f"Control Mode: CAN RMD Position Control",
            f"Target Position: {self._target_pos} degrees",
            f"Return Position: {self._return_pos} degrees",
            f"Command: {'0xA4 Multiturn+DPS' if self._use_multiturn else '0xA3 Position'}",
        ]
        if history:
            lines.append("=== Score History ===")
            for h in history:
                lines.append(
                    f"Iter {h.iteration}: Score={h.score:.1f}, "
                    f"PID(Kp={h.pid.kp:.5f}, Ki={h.pid.ki:.5f}, Kd={h.pid.kd:.5f})"
                )
            if len(history) >= 2:
                scores = [h.score for h in history[-3:]]
                if all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1)):
                    lines.append("WARNING: Scores declining! Reverse PID change direction.")
                elif all(scores[i] <= scores[i + 1] for i in range(len(scores) - 1)):
                    lines.append("Scores improving. Continue current direction.")
        return "\n".join(lines)

    def _clamp_change(self, current: RmdPidGains, suggested: RmdPidGains) -> RmdPidGains:
        def clamp(old, new, limit):
            if old == 0:
                return new
            max_change = abs(old) * limit
            diff = new - old
            if abs(diff) > max_change:
                diff = max_change if diff > 0 else -max_change
            return old + diff

        return RmdPidGains(
            kp=clamp(current.kp, suggested.kp, self._pid_limit),
            ki=clamp(current.ki, suggested.ki, self._pid_limit),
            kd=clamp(current.kd, suggested.kd, self._pid_limit),
        )
