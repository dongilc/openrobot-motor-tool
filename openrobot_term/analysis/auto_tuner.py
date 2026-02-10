"""
Automatic PID tuning loop.

Cycle: start motor → collect data → analyze → recommend PID → stop motor → apply → repeat.
"""

import time
from dataclasses import dataclass, field
from typing import Optional
from collections import deque

import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal, QCoreApplication

from ..protocol.serial_transport import SerialTransport
from ..protocol.commands import (
    CommPacketId, VescValues, build_get_values, build_set_mcconf_with_pid,
    build_set_rpm, build_set_pos, build_set_current,
)
from .signal_metrics import MotorMetrics, analyze_speed_control
from .llm_advisor import LLMAdvisor, PIDGains, AnalysisResult


@dataclass
class TuningIteration:
    iteration: int = 0
    pid: PIDGains = field(default_factory=PIDGains)
    metrics: Optional[MotorMetrics] = None
    analysis: Optional[AnalysisResult] = None
    score: float = 0.0


class AutoTuner(QThread):
    """
    Runs the auto-tuning loop in a background thread.
    Emits signals for UI updates.
    """

    status_update = pyqtSignal(str)
    iteration_complete = pyqtSignal(object)  # TuningIteration
    tuning_finished = pyqtSignal(str)        # reason
    data_collecting = pyqtSignal(float)      # progress 0-1
    metrics_ready = pyqtSignal(object)       # MotorMetrics - emitted immediately after analysis

    def __init__(
        self,
        transport: SerialTransport,
        advisor: LLMAdvisor,
        target_rpm: float,
        initial_pid: PIDGains,
        original_mcconf: bytes,
        position_mode: bool = False,
        max_iterations: int = 5,
        collect_duration_s: float = 5.0,
        sample_rate_hz: float = 20.0,
        target_score: float = 90.0,
        pid_change_limit: float = 0.5,  # max 50% change per iteration
    ):
        super().__init__()
        self._transport = transport
        self._advisor = advisor
        self._target_rpm = target_rpm
        self._current_pid = initial_pid
        self._original_mcconf = original_mcconf
        self._position_mode = position_mode
        self._max_iter = max_iterations
        self._collect_s = collect_duration_s
        self._sample_rate = sample_rate_hz
        self._target_score = target_score
        self._pid_limit = pid_change_limit
        self._running = False

        # Data collection buffers
        self._rpm_buf = []
        self._current_buf = []
        self._voltage_buf = []
        self._input_current_buf = []
        self._collecting = False

    def stop(self):
        self._running = False

    def on_values(self, v: VescValues):
        """Feed real-time data during collection phase."""
        if self._collecting:
            self._rpm_buf.append(v.rpm)
            self._current_buf.append(v.avg_motor_current)
            self._voltage_buf.append(v.v_in)
            self._input_current_buf.append(v.avg_input_current)

    def _sleep_with_events(self, seconds: float):
        """Sleep while allowing Qt to process events."""
        end_time = time.time() + seconds
        while time.time() < end_time:
            QCoreApplication.processEvents()
            QThread.msleep(50)  # 50ms chunks
            if not self._running:
                return

    def _keep_motor_running(self, duration_s: float, status_prefix: str = ""):
        """Keep motor running by sending commands periodically (VESC has timeout)."""
        end_time = time.time() + duration_s
        while time.time() < end_time:
            if not self._running:
                return False
            self._start_motor()  # Keep sending motor command
            QCoreApplication.processEvents()
            QThread.msleep(100)  # Send command every 100ms
        return True

    def run(self):
        self._running = True
        history = []

        # Track best PID for rollback
        best_score = 0.0
        best_pid = self._current_pid
        best_iteration = 0
        decline_count = 0  # Count consecutive score declines

        # Countdown before starting motor (3, 2, 1)
        for countdown in [3, 2, 1]:
            if not self._running:
                self.tuning_finished.emit("Cancelled by user")
                return
            self.status_update.emit(f"Starting in {countdown}... (Motor will rotate)")
            self._sleep_with_events(1.0)

        for iteration in range(1, self._max_iter + 1):
            if not self._running:
                self._stop_motor()
                self.tuning_finished.emit("Cancelled by user")
                return

            # Phase 0: Ensure motor is stopped before step response
            self.status_update.emit(f"Iteration {iteration}/{self._max_iter}: Preparing step response test...")
            self._stop_motor()
            self._sleep_with_events(1.5)  # Wait for motor to fully stop

            # Phase 1: Clear buffers and start motor + collection simultaneously
            self._rpm_buf.clear()
            self._current_buf.clear()
            self._voltage_buf.clear()
            self._input_current_buf.clear()

            self.status_update.emit(
                f"Iteration {iteration}/{self._max_iter}: Step to {self._target_rpm:.0f} RPM - collecting..."
            )

            # Start motor FIRST, then immediately begin collecting
            self._start_motor()
            self._collecting = True

            # Total collection time: transient (~2s) + steady state
            interval = 1.0 / self._sample_rate
            total_collect_time = self._collect_s + 2.0
            t0 = time.time()
            while time.time() - t0 < total_collect_time:
                if not self._running:
                    self._collecting = False
                    self._stop_motor()
                    self.tuning_finished.emit("Cancelled by user")
                    return
                # Keep motor running + request values
                self._start_motor()
                self._transport.send_packet(build_get_values())
                progress = (time.time() - t0) / total_collect_time
                self.data_collecting.emit(min(1.0, progress))
                QCoreApplication.processEvents()
                QThread.msleep(int(interval * 1000))

            self._collecting = False
            self.status_update.emit(
                f"Iteration {iteration}: Data collected ({len(self._rpm_buf)} samples)"
            )

            # Stop motor
            self.status_update.emit(f"Iteration {iteration}: Stopping motor...")
            self._stop_motor()
            self._sleep_with_events(2.0)

            if len(self._rpm_buf) < 10:
                self.status_update.emit(f"Not enough data collected ({len(self._rpm_buf)} samples). Check connection.")
                self.tuning_finished.emit("Insufficient data")
                return

            # Phase 2: Analyze
            self.status_update.emit(f"Iteration {iteration}: Analyzing data...")

            metrics = analyze_speed_control(
                rpm_data=np.array(self._rpm_buf),
                target_rpm=self._target_rpm,
                sample_rate=self._sample_rate,
                current_data=np.array(self._current_buf),
                voltage_data=np.array(self._voltage_buf),
                input_current_data=np.array(self._input_current_buf),
            )

            current_score = metrics.quality_score

            # Track best PID
            if current_score > best_score:
                best_score = current_score
                best_pid = self._current_pid
                best_iteration = iteration
                decline_count = 0  # Reset decline counter
            else:
                decline_count += 1

            # Display analysis results with trend indicator
            trend = ""
            if history:
                prev_score = history[-1].score
                if current_score > prev_score:
                    trend = " ↑ improved"
                elif current_score < prev_score:
                    trend = " ↓ declined"
                else:
                    trend = " → same"

            self.status_update.emit(
                f"Iteration {iteration}: Quality Score = {current_score:.1f}{trend} (Best: {best_score:.1f} @ iter {best_iteration})"
            )
            self.status_update.emit(
                f"  [Steady] Ripple={metrics.rpm_ripple_pct:.2f}%, SS_Err={metrics.steady_state_error_pct:.2f}%"
            )
            self.status_update.emit(
                f"  [Transient] Rise={metrics.rise_time:.3f}s, Settle={metrics.settling_time:.3f}s, "
                f"Overshoot={metrics.overshoot_pct:.1f}%"
            )

            # Emit metrics immediately for UI update (before LLM consultation)
            self.metrics_ready.emit(metrics)

            # Check if target score already reached - no need to change PID
            if current_score >= self._target_score:
                # Record this iteration without changing PID
                result = TuningIteration(
                    iteration=iteration,
                    pid=self._current_pid,
                    metrics=metrics,
                    analysis=None,
                    score=current_score,
                )
                history.append(result)
                self.iteration_complete.emit(result)

                self.status_update.emit(
                    f"[Target Reached] Score {current_score:.1f} >= {self._target_score:.1f}. "
                    f"Current PID is already optimal - no changes needed."
                )
                self.tuning_finished.emit(
                    f"Target score reached on iteration {iteration}! "
                    f"Score: {current_score:.1f}. PID unchanged."
                )
                return

            # Check for early stopping (2+ consecutive declines)
            if decline_count >= 2 and iteration >= 3:
                self.status_update.emit(
                    f"[Early Stop] Score declined {decline_count} times consecutively. "
                    f"Rolling back to best PID from iteration {best_iteration}..."
                )
                # Rollback to best PID
                self._apply_pid(best_pid, f"Rollback to best (iter {best_iteration})")
                self._current_pid = best_pid

                # Run verification test
                verify_score = self._run_verification_test(best_iteration)

                self.tuning_finished.emit(
                    f"Early stopped after {iteration} iterations (score declining). "
                    f"Best score: {best_score:.1f} @ iteration {best_iteration}. "
                    f"Verification score: {verify_score:.1f}. Rolled back to best PID."
                )
                return

            # Phase 3: Get LLM recommendation with detailed history
            self.status_update.emit(f"Iteration {iteration}: Consulting AI advisor...")

            # Build detailed context with score history
            context_lines = []
            if history:
                context_lines.append("=== Score History ===")
                for h in history:
                    context_lines.append(
                        f"Iter {h.iteration}: Score={h.score:.1f}, "
                        f"PID(Kp={h.pid.kp:.6f}, Ki={h.pid.ki:.6f}, Kd={h.pid.kd:.6f}, Ramp={h.pid.ramp_erpms_s:.0f})"
                    )
                # Add trend analysis
                if len(history) >= 2:
                    recent_scores = [h.score for h in history[-3:]]
                    if all(recent_scores[i] >= recent_scores[i+1] for i in range(len(recent_scores)-1)):
                        context_lines.append("⚠️ WARNING: Scores have been declining! Consider reversing direction of PID changes.")
                    elif all(recent_scores[i] <= recent_scores[i+1] for i in range(len(recent_scores)-1)):
                        context_lines.append("✓ Scores improving. Continue current tuning direction.")
                context_lines.append(f"Best score so far: {best_score:.1f} at iteration {best_iteration}")
                context_lines.append(
                    f"Current PID: Kp={self._current_pid.kp:.6f}, Ki={self._current_pid.ki:.6f}, "
                    f"Kd={self._current_pid.kd:.6f}, Ramp={self._current_pid.ramp_erpms_s:.0f}"
                )

            context = "\n".join(context_lines)

            analysis = self._advisor.analyze_and_recommend(
                metrics=metrics,
                current_pid=self._current_pid,
                additional_context=context,
            )

            # Phase 4: Validate and apply PID
            if analysis.suggested_pid:
                new_pid = self._clamp_pid_change(
                    self._current_pid, analysis.suggested_pid
                )
            else:
                new_pid = self._current_pid

            result = TuningIteration(
                iteration=iteration,
                pid=self._current_pid,
                metrics=metrics,
                analysis=analysis,
                score=current_score,
            )
            history.append(result)
            self.iteration_complete.emit(result)

            # Apply new PID to VESC via MCCONF
            self._apply_pid(new_pid, f"Iteration {iteration}")
            self._current_pid = new_pid

            # Wait for VESC to process MCCONF update (2 seconds)
            self.status_update.emit(f"Iteration {iteration}: Waiting 2s for VESC to apply new PID...")
            self._sleep_with_events(2.0)

            # Stabilization delay before next iteration
            self.status_update.emit(
                f"Iteration {iteration}: Waiting 2s before next iteration..."
            )
            self._sleep_with_events(2.0)

        # Tuning complete - rollback to best if current is worse
        if history and history[-1].score < best_score:
            self.status_update.emit(
                f"Final score ({history[-1].score:.1f}) < Best ({best_score:.1f}). "
                f"Rolling back to best PID from iteration {best_iteration}..."
            )
            self._apply_pid(best_pid, f"Rollback to best (iter {best_iteration})")
            self._current_pid = best_pid

            # Run verification test
            verify_score = self._run_verification_test(best_iteration)

            self.tuning_finished.emit(
                f"Completed {self._max_iter} iterations. "
                f"Best score: {best_score:.1f} @ iteration {best_iteration}. "
                f"Verification score: {verify_score:.1f}. Rolled back to best PID."
            )
        else:
            self.tuning_finished.emit(
                f"Completed {self._max_iter} iterations. "
                f"Best score: {best_score:.1f} @ iteration {best_iteration}. "
                f"Applied best PID."
            )

    def _apply_pid(self, pid: PIDGains, label: str):
        """Apply PID to VESC."""
        if self._position_mode:
            self.status_update.emit(
                f"{label}: Applying PID "
                f"Kp={pid.kp:.6f} Ki={pid.ki:.6f} Kd={pid.kd:.6f} Kd_flt={pid.kd_filter:.4f}"
            )
            packet = build_set_mcconf_with_pid(
                self._original_mcconf,
                pid.kp, pid.ki, pid.kd, pid.kd_filter,
                position_mode=True
            )
        else:
            self.status_update.emit(
                f"{label}: Applying PID "
                f"Kp={pid.kp:.6f} Ki={pid.ki:.6f} Kd={pid.kd:.6f} Kd_flt={pid.kd_filter:.4f} "
                f"Ramp={pid.ramp_erpms_s:.0f}"
            )
            packet = build_set_mcconf_with_pid(
                self._original_mcconf,
                pid.kp, pid.ki, pid.kd, pid.kd_filter,
                position_mode=False,
                ramp_erpms_s=pid.ramp_erpms_s
            )
        self._transport.send_packet(packet)

    def _run_verification_test(self, best_iteration: int) -> float:
        """Run verification test after rollback to confirm performance."""
        self.status_update.emit(f"[Verification] Running confirmation test for best PID (iter {best_iteration})...")

        # Wait for VESC to apply PID
        self._sleep_with_events(2.0)

        # Stop motor and prepare
        self._stop_motor()
        self._sleep_with_events(1.5)

        # Clear buffers
        self._rpm_buf.clear()
        self._current_buf.clear()
        self._voltage_buf.clear()
        self._input_current_buf.clear()

        self.status_update.emit(f"[Verification] Step to {self._target_rpm:.0f} RPM - collecting...")

        # Start motor and collect data
        self._start_motor()
        self._collecting = True

        interval = 1.0 / self._sample_rate
        total_collect_time = self._collect_s + 2.0
        t0 = time.time()
        while time.time() - t0 < total_collect_time:
            if not self._running:
                self._collecting = False
                self._stop_motor()
                return 0.0
            self._start_motor()
            self._transport.send_packet(build_get_values())
            progress = (time.time() - t0) / total_collect_time
            self.data_collecting.emit(min(1.0, progress))
            QCoreApplication.processEvents()
            QThread.msleep(int(interval * 1000))

        self._collecting = False

        # Stop motor
        self._stop_motor()
        self._sleep_with_events(2.0)

        if len(self._rpm_buf) < 10:
            self.status_update.emit(f"[Verification] Insufficient data ({len(self._rpm_buf)} samples)")
            return 0.0

        # Analyze
        metrics = analyze_speed_control(
            rpm_data=np.array(self._rpm_buf),
            target_rpm=self._target_rpm,
            sample_rate=self._sample_rate,
            current_data=np.array(self._current_buf),
            voltage_data=np.array(self._voltage_buf),
            input_current_data=np.array(self._input_current_buf),
        )

        verify_score = metrics.quality_score

        self.status_update.emit(
            f"[Verification] Score = {verify_score:.1f} "
            f"(Ripple={metrics.rpm_ripple_pct:.2f}%, SS_Err={metrics.steady_state_error_pct:.2f}%)"
        )
        self.status_update.emit(
            f"[Verification] Rise={metrics.rise_time:.3f}s, Settle={metrics.settling_time:.3f}s, "
            f"Overshoot={metrics.overshoot_pct:.1f}%"
        )

        # Emit metrics for UI update
        self.metrics_ready.emit(metrics)

        return verify_score

    def _clamp_pid_change(self, current: PIDGains, suggested: PIDGains) -> PIDGains:
        """Limit PID changes to prevent instability."""
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

    def _start_motor(self):
        """Start motor at target speed/position."""
        if self._position_mode:
            self._transport.send_packet(build_set_pos(self._target_rpm))  # target_rpm is actually target_pos
        else:
            self._transport.send_packet(build_set_rpm(int(self._target_rpm)))

    def _stop_motor(self):
        """Stop motor by setting current to 0."""
        self._transport.send_packet(build_set_current(0.0))
