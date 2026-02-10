"""
LLM-based motor control advisor.

Takes signal processing metrics and current PID gains, sends to LLM API,
and returns natural language analysis with PID gain recommendations.

Supports:
- Speed/Position PID tuning (LLMAdvisor)
- FOC Current loop tuning (CurrentControlAdvisor)
- FOC Waveform/FFT interpretation (WaveformAdvisor)
"""

import json
from dataclasses import dataclass, asdict
from typing import Optional

from .signal_metrics import MotorMetrics
from .current_metrics import CurrentMetrics


@dataclass
class PIDGains:
    kp: float = 0.0
    ki: float = 0.0
    kd: float = 0.0
    kd_filter: float = 0.0
    ramp_erpms_s: float = 0.0  # Speed setpoint ramp rate (eRPM/s) - affects transient


@dataclass
class AnalysisResult:
    summary: str = ""
    recommendations: list = None
    suggested_pid: Optional[PIDGains] = None
    confidence: float = 0.0
    raw_request: str = ""   # Raw request sent to LLM
    raw_response: str = ""  # Raw response from LLM

    def __post_init__(self):
        if self.recommendations is None:
            self.recommendations = []


SYSTEM_PROMPT = """You are an expert motor control engineer specializing in BLDC/PMSM motor control with FOC (Field Oriented Control). You analyze motor performance data and provide actionable PID tuning recommendations.

## Your Role:
1. Analyze the provided motor control quality metrics
2. Compare with previous iterations (if provided) to determine if changes helped or hurt
3. Recommend specific PID gain adjustments with clear reasoning
4. **CRITICAL: If the quality score DECREASED from previous iteration, REVERSE the direction of PID changes**

## Response Format (JSON):
{
    "summary": "Brief overview of motor control quality (2-3 sentences)",
    "issues": ["list of identified issues"],
    "score_trend": "improved/declined/stable",
    "recommendations": ["list of specific actionable recommendations"],
    "suggested_pid": {
        "kp": <float>,
        "ki": <float>,
        "kd": <float>,
        "kd_filter": <float>,
        "ramp_erpms_s": <float>
    },
    "reasoning": "Detailed explanation of why these PID values are recommended",
    "confidence": <float 0-1>
}

## PID Tuning Guidelines for SPEED CONTROL:

### Transient Response (Rise time, Overshoot, Settling time):
- Slow rise time → INCREASE Kp (proportional gain provides faster response)
- High overshoot → DECREASE Kp OR INCREASE Kd (derivative provides damping)
- Long settling time with oscillation → INCREASE Kd
- Long settling time without oscillation → INCREASE Ki slightly

### Steady-State Response (Ripple, SS Error):
- High steady-state error → INCREASE Ki (integral eliminates steady-state error)
- High RPM ripple/oscillation → DECREASE Kp, may need to INCREASE Kd
- Low frequency oscillation → DECREASE Ki (integral windup issue)

### Derivative Filter (kd_filter):
- Range: 0.0 to 1.0
- Higher = less filtering, faster but noisier
- Lower = more filtering, smoother but slower
- Typical: 0.2 to 0.5

### Speed Ramp Rate (ramp_erpms_s):
- Controls how fast the speed setpoint changes (eRPM per second)
- **CRITICAL for transient response** - limits how quickly motor can accelerate
- Higher value = faster acceleration, but may cause overshoot if PID can't track
- Lower value = smoother acceleration, but slower rise time
- Typical range: 1000 to 100000 eRPM/s
- If rise time is slow despite high Kp → INCREASE ramp rate
- If overshoot is high → DECREASE ramp rate OR improve PID damping first

## CRITICAL RULES:
1. **Score Declined?** If quality_score decreased from previous iteration:
   - The PREVIOUS change made things WORSE
   - REVERSE the direction: if you increased Kp last time, DECREASE it now
   - Try a SMALLER change magnitude (5-15% instead of 20-30%)

2. **Score Improved?** If quality_score increased:
   - The PREVIOUS change helped
   - Continue in the SAME direction but with smaller magnitude
   - Fine-tune rather than large changes

3. **Conservative Changes**:
   - First iteration: 10-20% change max
   - Subsequent iterations: 5-15% change max
   - Never change all 4 parameters at once - focus on 1-2

4. **Priority Order**:
   - Fix overshoot/oscillation FIRST (stability) → adjust Kp, Kd
   - Then fix settling time → adjust Kd, Ki
   - Finally fine-tune steady-state error → adjust Ki

## Position Control Specific:
- Position control needs HIGHER Kd relative to Kp (more damping)
- Be more conservative with Ki (position integral windup is problematic)
- Overshoot in position is more critical than in speed control"""

# Language instruction suffix
LANGUAGE_INSTRUCTION = {
    "Korean": "\n\nRespond in Korean.",
    "English": "\n\nRespond in English.",
}


class LLMAdvisor:
    """LLM-based PID tuning advisor using OpenAI API."""

    def __init__(self, api_key: str, model: str = "gpt-4.1", language: str = "Korean"):
        self._model = model
        self._openai_key = api_key
        self._language = language

    def _get_system_prompt(self) -> str:
        """Get system prompt with language instruction."""
        return SYSTEM_PROMPT + LANGUAGE_INSTRUCTION.get(self._language, LANGUAGE_INSTRUCTION["Korean"])

    def build_request_data(
        self,
        metrics: MotorMetrics,
        current_pid: PIDGains,
        additional_context: str = "",
    ) -> tuple:
        """Build request data and return (metrics_dict, pid_dict, user_msg)."""
        m = {
            "target_rpm": metrics.target_rpm,
            "rpm_mean": round(metrics.rpm_mean, 2),
            "rpm_std": round(metrics.rpm_std, 2),
            "rpm_ripple_pct": round(metrics.rpm_ripple_pct, 2),
            "steady_state_error": round(metrics.steady_state_error, 2),
            "steady_state_error_pct": round(metrics.steady_state_error_pct, 2),
            "settling_time_s": round(metrics.settling_time, 3),
            "overshoot_pct": round(metrics.overshoot_pct, 2),
            "rise_time_s": round(metrics.rise_time, 3),
            "current_thd_pct": round(metrics.current_thd, 2),
            "current_ripple_pct": round(metrics.current_ripple_pct, 2),
            "fft_dominant_freq_hz": round(metrics.fft_dominant_freq, 2),
            "fft_dominant_magnitude": round(metrics.fft_dominant_magnitude, 2),
            "power_mean_w": round(metrics.power_mean, 2),
            "quality_score": round(metrics.quality_score, 1),
        }

        pid = {
            "kp": current_pid.kp,
            "ki": current_pid.ki,
            "kd": current_pid.kd,
            "kd_filter": current_pid.kd_filter,
            "ramp_erpms_s": current_pid.ramp_erpms_s,
        }

        context_section = (
            f"## Additional Context & History\n{additional_context}"
            if additional_context else ""
        )

        user_msg = (
            "Analyze the following motor control data and recommend PID adjustments:\n\n"
            f"## Current PID Gains\n```json\n{json.dumps(pid, indent=2)}\n```\n\n"
            f"## Motor Control Quality Metrics\n```json\n{json.dumps(m, indent=2)}\n```\n\n"
            f"{context_section}\n\n"
            "Based on the metrics and history, provide your analysis and recommended PID gains in JSON format."
        )
        return m, pid, user_msg

    def analyze_and_recommend(
        self,
        metrics: MotorMetrics,
        current_pid: PIDGains,
        additional_context: str = "",
    ) -> AnalysisResult:
        """
        Send metrics to LLM and get analysis + PID recommendations.
        """
        m, pid, user_msg = self.build_request_data(metrics, current_pid, additional_context)

        try:
            import sys
            print(f"[LLM] Calling OpenAI API with model: {self._model}", flush=True)
            sys.stdout.flush()

            raw = self._call_openai(user_msg)

            print(f"[LLM] Response received", flush=True)

            # Parse JSON from response (handle markdown code blocks)
            json_str = raw
            if "```json" in raw:
                json_str = raw.split("```json")[1].split("```")[0].strip()
            elif "```" in raw:
                json_str = raw.split("```")[1].split("```")[0].strip()

            data = json.loads(json_str)

            result = AnalysisResult(
                summary=data.get("summary", ""),
                recommendations=data.get("recommendations", []),
                confidence=data.get("confidence", 0.0),
                raw_request=user_msg,
                raw_response=raw,
            )

            if "suggested_pid" in data and data["suggested_pid"]:
                sp = data["suggested_pid"]
                result.suggested_pid = PIDGains(
                    kp=sp.get("kp", current_pid.kp),
                    ki=sp.get("ki", current_pid.ki),
                    kd=sp.get("kd", current_pid.kd),
                    kd_filter=sp.get("kd_filter", current_pid.kd_filter),
                    ramp_erpms_s=sp.get("ramp_erpms_s", current_pid.ramp_erpms_s),
                )

            return result

        except Exception as e:
            import traceback
            error_detail = traceback.format_exc()
            print(f"[LLM] API Error: {e}\n{error_detail}")
            return AnalysisResult(
                summary=f"LLM API error: {type(e).__name__}: {e}",
                recommendations=[f"Check API key and network connection"],
                confidence=0.0,
                raw_request=user_msg,
                raw_response=error_detail,
            )

    def _call_openai(self, user_msg: str) -> str:
        """Call OpenAI API."""
        from openai import OpenAI

        client = OpenAI(api_key=self._openai_key, timeout=30.0)

        response = client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": self._get_system_prompt()},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.3,
            response_format={"type": "json_object"},
        )

        return response.choices[0].message.content


# =============================================================================
# CAN/RMD Position PID Advisor
# =============================================================================

CAN_POSITION_SYSTEM_PROMPT = """You are an expert motor control engineer specializing in RMD brushless actuator position control via CAN bus. You analyze position step response data and provide PID tuning recommendations.

## Motor Protocol:
- RMD motor with 3-parameter position PID: Kp, Ki, Kd
- PID encoding: Kp × 1000, Ki × 100000, Kd × 100000 (integer scale in CAN frames)
- Position command: 0xA3 (0.01 deg/LSB, int32, multi-turn capable)
- Single-turn encoder: 14-bit (16383 counts / 360°, ~0.022°/count resolution)
- PID is written to RAM (0x31) during tuning, ROM (0x32) for permanent storage

## Response Format (JSON):
{
    "summary": "Brief overview of position control quality (2-3 sentences)",
    "issues": ["list of identified issues"],
    "score_trend": "improved/declined/stable",
    "recommendations": ["list of specific actionable recommendations"],
    "suggested_pid": {
        "kp": <float>,
        "ki": <float>,
        "kd": <float>,
        "kd_filter": 0,
        "ramp_erpms_s": 0
    },
    "reasoning": "Detailed explanation of why these PID values are recommended",
    "confidence": <float 0-1>
}

## Position PID Tuning Guidelines for RMD Motors:

### Transient Response:
- Slow rise time → INCREASE Kp
- High overshoot → DECREASE Kp OR INCREASE Kd (derivative provides damping)
- Long settling with oscillation → INCREASE Kd
- Long settling without oscillation → INCREASE Ki slightly

### Steady-State Response:
- High position error → INCREASE Ki (integral eliminates error)
- Position ripple/vibration → DECREASE Kp, INCREASE Kd
- Low frequency oscillation → DECREASE Ki (integral windup)

### Position-Specific Rules:
- Position control needs HIGHER Kd relative to Kp (more damping than speed control)
- Be VERY conservative with Ki (position integral windup causes limit cycling)
- Overshoot in position control is MORE critical than in speed control
- Typical RMD values: Kp 0.005-0.100, Ki 0.00001-0.001, Kd 0.00001-0.01

## CRITICAL RULES:
1. **Score Declined?** REVERSE direction of PID changes from previous iteration
2. **Score Improved?** Continue same direction, smaller magnitude
3. **Conservative Changes**: First iteration max 15-20%, subsequent max 5-15%
4. **Priority**: Fix overshoot/oscillation FIRST → settling time → SS error
5. Focus on 1-2 parameters per iteration, not all three"""


class CanPositionAdvisor(LLMAdvisor):
    """LLM advisor specialized for CAN/RMD position PID tuning."""

    def _get_system_prompt(self) -> str:
        return CAN_POSITION_SYSTEM_PROMPT + LANGUAGE_INSTRUCTION.get(
            self._language, LANGUAGE_INSTRUCTION["Korean"]
        )

    def build_request_data(self, metrics, current_pid, additional_context=""):
        """Format CAN position metrics for LLM request."""
        m = {
            "target_pos_deg": round(metrics.target_pos_deg, 2),
            "pos_mean_deg": round(metrics.pos_mean_deg, 2),
            "pos_ripple_deg": round(metrics.pos_ripple_deg, 3),
            "pos_ripple_pct": round(metrics.pos_ripple_pct, 2),
            "steady_state_error_deg": round(metrics.steady_state_error_deg, 3),
            "steady_state_error_pct": round(metrics.steady_state_error_pct, 2),
            "settling_time_s": round(metrics.settling_time_s, 3),
            "overshoot_pct": round(metrics.overshoot_pct, 2),
            "rise_time_s": round(metrics.rise_time_s, 3),
            "torque_peak_a": round(metrics.torque_peak_a, 2),
            "fft_dominant_freq_hz": round(metrics.fft_dominant_freq, 2),
            "quality_score": round(metrics.quality_score, 1),
        }

        pid = {
            "kp": current_pid.kp,
            "ki": current_pid.ki,
            "kd": current_pid.kd,
            "kd_filter": getattr(current_pid, "kd_filter", 0),
            "ramp_erpms_s": getattr(current_pid, "ramp_erpms_s", 0),
        }

        context_section = (
            f"## Additional Context & History\n{additional_context}"
            if additional_context else ""
        )

        user_msg = (
            "Analyze the following CAN/RMD position control data and recommend PID adjustments:\n\n"
            f"## Current PID Gains\n```json\n{json.dumps(pid, indent=2)}\n```\n\n"
            f"## Position Control Quality Metrics\n```json\n{json.dumps(m, indent=2)}\n```\n\n"
            f"{context_section}\n\n"
            "Based on the metrics and history, provide your analysis and recommended PID gains in JSON format."
        )
        return m, pid, user_msg


# =============================================================================
# Current Control (FOC) Advisor
# =============================================================================

@dataclass
class FOCCurrentGains:
    """FOC current controller gains."""
    kp: float = 0.0
    ki: float = 0.0


@dataclass
class CurrentAnalysisResult:
    """Result from current control LLM analysis."""
    summary: str = ""
    recommendations: list = None
    suggested_gains: Optional[FOCCurrentGains] = None
    confidence: float = 0.0
    raw_request: str = ""
    raw_response: str = ""

    def __post_init__(self):
        if self.recommendations is None:
            self.recommendations = []


CURRENT_CONTROL_SYSTEM_PROMPT = """You are an expert motor control engineer specializing in BLDC/PMSM current control with FOC (Field Oriented Control). You analyze current control loop quality and provide tuning recommendations.

## Your Role:
1. Analyze current control quality metrics (ripple, THD, tracking error, transient response)
2. Recommend FOC current controller gains (foc_current_kp, foc_current_ki)
3. Consider both transient and steady-state performance

## Response Format (JSON):
{
    "summary": "Brief overview of current control quality (2-3 sentences)",
    "issues": ["list of identified issues"],
    "score_trend": "improved/declined/stable",
    "recommendations": ["list of specific actionable recommendations"],
    "suggested_gains": {
        "kp": <float>,
        "ki": <float>
    },
    "reasoning": "Detailed explanation of why these gains are recommended",
    "confidence": <float 0-1>
}

## Current Control Tuning Guidelines:

### Understanding Current Loop Behavior:
- Current loop is the INNERMOST loop (fastest response needed)
- Typical bandwidth: 1-10 kHz
- Affects torque response and current ripple

### Tuning Rules:
1. **High current ripple** → INCREASE Kp (faster correction)
2. **Slow current response (rise time)** → INCREASE Kp and/or Ki
3. **Current oscillation/ringing** → DECREASE Kp, may increase Ki slightly
4. **High THD** → Check switching frequency, may need filter tuning
5. **Poor tracking (high steady-state error)** → INCREASE Ki (integral eliminates error)
6. **Overshoot in current step** → DECREASE Kp

### Typical Value Ranges (VESC FOC):
- foc_current_kp: varies by motor (typical: 0.001 - 0.1)
- foc_current_ki: varies by motor (can range from 0.005 to 100+)
- IMPORTANT: Use the CURRENT gains as reference. Suggest changes relative to current values (±10-20% max per iteration).

### Priority:
1. STABILITY first (no oscillation)
2. Low ripple/THD second
3. Fast response third

## CRITICAL RULES:
1. **Score Declined?** REVERSE direction of gain changes
2. **Score Improved?** Continue same direction, smaller magnitude
3. **Conservative Changes**: 10-20% per iteration max
4. Current loop instability can damage hardware - be careful!"""


class CurrentControlAdvisor:
    """LLM-based FOC current control tuning advisor."""

    def __init__(self, api_key: str, model: str = "gpt-4.1", language: str = "Korean"):
        self._model = model
        self._openai_key = api_key
        self._language = language

    def _get_system_prompt(self) -> str:
        """Get system prompt with language instruction."""
        return CURRENT_CONTROL_SYSTEM_PROMPT + LANGUAGE_INSTRUCTION.get(
            self._language, LANGUAGE_INSTRUCTION["Korean"]
        )

    def build_request_data(
        self,
        step_metrics: CurrentMetrics,
        steady_metrics: CurrentMetrics,
        current_gains: FOCCurrentGains,
        additional_context: str = "",
    ) -> tuple:
        """Build request data for current control analysis."""
        # Phase 1: Step response metrics
        step_data = {
            "target_current_A": step_metrics.target_current,
            "rise_time_ms": round(step_metrics.rise_time_ms, 2),
            "settling_time_ms": round(step_metrics.settling_time_ms, 2),
            "overshoot_pct": round(step_metrics.overshoot_pct, 2),
            "quality_score": round(step_metrics.quality_score, 1),
        }

        # Phase 2: Steady-state metrics
        steady_data = {
            "current_mean_A": round(steady_metrics.current_mean, 3),
            "current_ripple_pct": round(steady_metrics.current_ripple_pct, 2),
            "current_thd_pct": round(steady_metrics.current_thd, 2),
            "tracking_error_pct": round(steady_metrics.tracking_error_pct, 2),
            "fft_dominant_freq_hz": round(steady_metrics.fft_dominant_freq, 1),
            "quality_score": round(steady_metrics.quality_score, 1),
        }

        # Combined score (40% step + 60% steady)
        combined_score = 0.4 * step_metrics.quality_score + 0.6 * steady_metrics.quality_score

        gains = {
            "kp": current_gains.kp,
            "ki": current_gains.ki,
        }

        context_section = (
            f"## Additional Context & History\n{additional_context}"
            if additional_context else ""
        )

        user_msg = (
            "Analyze the following FOC current control data and recommend gain adjustments:\n\n"
            f"## Current FOC Gains\n```json\n{json.dumps(gains, indent=2)}\n```\n\n"
            f"## Phase 1: Step Response (Transient)\n```json\n{json.dumps(step_data, indent=2)}\n```\n\n"
            f"## Phase 2: Steady-State (During Speed Control)\n```json\n{json.dumps(steady_data, indent=2)}\n```\n\n"
            f"## Combined Quality Score: {combined_score:.1f}\n\n"
            f"{context_section}\n\n"
            "Based on the metrics and history, provide your analysis and recommended gains in JSON format."
        )

        return step_data, steady_data, gains, user_msg

    def analyze_and_recommend(
        self,
        step_metrics: CurrentMetrics,
        steady_metrics: CurrentMetrics,
        current_gains: FOCCurrentGains,
        additional_context: str = "",
    ) -> CurrentAnalysisResult:
        """Send metrics to LLM and get analysis + gain recommendations."""
        step_data, steady_data, gains, user_msg = self.build_request_data(
            step_metrics, steady_metrics, current_gains, additional_context
        )

        try:
            import sys
            print(f"[CurrentLLM] Calling OpenAI API with model: {self._model}", flush=True)
            sys.stdout.flush()

            raw = self._call_openai(user_msg)

            print(f"[CurrentLLM] Response received", flush=True)

            # Parse JSON from response
            json_str = raw
            if "```json" in raw:
                json_str = raw.split("```json")[1].split("```")[0].strip()
            elif "```" in raw:
                json_str = raw.split("```")[1].split("```")[0].strip()

            data = json.loads(json_str)

            result = CurrentAnalysisResult(
                summary=data.get("summary", ""),
                recommendations=data.get("recommendations", []),
                confidence=data.get("confidence", 0.0),
                raw_request=user_msg,
                raw_response=raw,
            )

            if "suggested_gains" in data and data["suggested_gains"]:
                sg = data["suggested_gains"]
                result.suggested_gains = FOCCurrentGains(
                    kp=sg.get("kp", current_gains.kp),
                    ki=sg.get("ki", current_gains.ki),
                )

            return result

        except Exception as e:
            import traceback
            error_detail = traceback.format_exc()
            print(f"[CurrentLLM] API Error: {e}\n{error_detail}")
            return CurrentAnalysisResult(
                summary=f"LLM API error: {type(e).__name__}: {e}",
                recommendations=["Check API key and network connection"],
                confidence=0.0,
                raw_request=user_msg,
                raw_response=error_detail,
            )

    def _call_openai(self, user_msg: str) -> str:
        """Call OpenAI API."""
        from openai import OpenAI

        client = OpenAI(api_key=self._openai_key, timeout=30.0)

        response = client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": self._get_system_prompt()},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.3,
            response_format={"type": "json_object"},
        )

        return response.choices[0].message.content


# =============================================================================
# Waveform / FFT Interpretation Advisor
# =============================================================================

WAVEFORM_SYSTEM_PROMPT = """You are an expert motor control engineer specializing in BLDC/PMSM FOC (Field Oriented Control) waveform analysis. You interpret 3-phase current waveform captures and FFT spectra from VESC motor controllers.

## Your Role:
1. Analyze the provided FOC waveform metrics (phase amplitudes, balance, THD, FFT peaks)
2. Explain what each FFT peak means in practical terms
3. Identify motor/inverter issues visible in the waveform
4. Provide actionable recommendations

## IMPORTANT - Pre-computed Data:
The input data includes pre-computed harmonic analysis for each FFT peak:
- "nearest_harmonic": which integer harmonic of the fundamental this peak is closest to
- "harmonic_type": "odd" or "even"
- "deviation_from_harmonic_hz": how far from the exact harmonic frequency (small = real harmonic, large = noise/artifact)
- "relative_to_fundamental_dB": magnitude relative to fundamental in dB
- "aliasing_note": explains the aliasing conditions for this capture

**USE these pre-computed values.** Do NOT recalculate harmonic numbers yourself — trust the provided nearest_harmonic field.

## Response Format (JSON):
{
    "summary": "Brief overview of waveform quality (2-3 sentences)",
    "fft_interpretation": [
        {"freq_hz": <float>, "harmonic": "<nth> harmonic", "description": "What this peak represents and its significance"}
    ],
    "issues": ["list of identified issues or concerns"],
    "recommendations": ["list of actionable recommendations"],
    "motor_health": "excellent/good/fair/poor",
    "confidence": <float 0-1>
}

## FFT Peak Interpretation Guide (use nearest_harmonic from data):

### Fundamental (1st harmonic, nearest_harmonic=1):
- Electrical rotation frequency f_e = eRPM / 60
- Should be the strongest peak in a healthy motor

### Odd harmonics (3rd, 5th, 7th, 9th, 11th, ...):
- **3rd**: Common in trapezoidal back-EMF motors (non-sinusoidal back-EMF)
- **5th, 7th**: Dead-time distortion + PWM nonlinearity in 3-phase inverters
- **Higher odd (11th, 13th, ...)**: Decreasing magnitude expected; if strong, indicates significant distortion
- In 3-phase systems, triplen harmonics (3rd, 9th, 15th) cancel in line currents but appear in phase currents

### Even harmonics (2nd, 4th, 6th, ...):
- **Indicate asymmetry** — should be near zero in a balanced, symmetric system
- Possible causes: winding imbalance, current sensor offset, mechanical eccentricity, half-wave asymmetry
- If 2nd harmonic is significant (>-20dB from fundamental), investigate hardware

### Non-harmonic peaks (large deviation_from_harmonic_hz):
- If deviation > FFT resolution: likely NOT a true harmonic
- Could be: mechanical resonance, bearing defect frequency, intermodulation, or noise artifact
- Do not force-interpret as a harmonic

### Aliasing effects:
- Read the aliasing_note field for this specific capture's aliasing conditions
- Switching sidebands (f_sw +/- n*f_e) alias onto n*f_e when f_samp = f_sw (or f_sw/2)
- This means switching noise adds to existing harmonics, not to separate frequencies

### Phase Balance:
- >95%: Excellent — well-matched motor/inverter
- 85-95%: Good — minor asymmetry, acceptable
- 70-85%: Fair — noticeable imbalance, check connections and sensors
- <70%: Poor — likely hardware issue

## CRITICAL:
- USE the pre-computed nearest_harmonic, do NOT guess harmonic numbers from frequency
- Peaks with large deviation from nearest harmonic are likely NOT real harmonics
- Focus on practical implications (torque ripple, efficiency, noise, reliability)
- Consider the specific motor operating point (eRPM, current level)"""


@dataclass
class WaveformAnalysisResult:
    """Result from waveform/FFT LLM analysis."""
    summary: str = ""
    fft_interpretation: list = None
    issues: list = None
    recommendations: list = None
    motor_health: str = ""
    confidence: float = 0.0
    raw_request: str = ""
    raw_response: str = ""

    def __post_init__(self):
        if self.fft_interpretation is None:
            self.fft_interpretation = []
        if self.issues is None:
            self.issues = []
        if self.recommendations is None:
            self.recommendations = []


class WaveformAdvisor:
    """LLM-based FOC waveform/FFT interpretation advisor."""

    def __init__(self, api_key: str, model: str = "gpt-4.1", language: str = "Korean"):
        self._model = model
        self._openai_key = api_key
        self._language = language

    def _get_system_prompt(self) -> str:
        return WAVEFORM_SYSTEM_PROMPT + LANGUAGE_INSTRUCTION.get(
            self._language, LANGUAGE_INSTRUCTION["Korean"]
        )

    def analyze_waveform(self, waveform_data: dict) -> WaveformAnalysisResult:
        """
        Send waveform metrics to LLM and get interpretation.

        Args:
            waveform_data: dict with keys like erpm, sample_rate, phase_amps,
                           thd_values, balance_pct, quality_score, fft_peaks, etc.
        """
        user_msg = (
            "Analyze the following FOC waveform capture data:\n\n"
            f"```json\n{json.dumps(waveform_data, indent=2)}\n```\n\n"
            "Provide your interpretation of the waveform quality and FFT peaks in JSON format."
        )

        try:
            raw = self._call_openai(user_msg)

            json_str = raw
            if "```json" in raw:
                json_str = raw.split("```json")[1].split("```")[0].strip()
            elif "```" in raw:
                json_str = raw.split("```")[1].split("```")[0].strip()

            data = json.loads(json_str)

            return WaveformAnalysisResult(
                summary=data.get("summary", ""),
                fft_interpretation=data.get("fft_interpretation", []),
                issues=data.get("issues", []),
                recommendations=data.get("recommendations", []),
                motor_health=data.get("motor_health", ""),
                confidence=data.get("confidence", 0.0),
                raw_request=user_msg,
                raw_response=raw,
            )

        except Exception as e:
            import traceback
            return WaveformAnalysisResult(
                summary=f"LLM API error: {type(e).__name__}: {e}",
                issues=["Check API key and network connection"],
                confidence=0.0,
                raw_request=user_msg,
                raw_response=traceback.format_exc(),
            )

    def _call_openai(self, user_msg: str) -> str:
        from openai import OpenAI

        client = OpenAI(api_key=self._openai_key, timeout=30.0)

        response = client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": self._get_system_prompt()},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.3,
            response_format={"type": "json_object"},
        )

        return response.choices[0].message.content
