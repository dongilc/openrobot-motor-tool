# Inertia Tune and the New Position Control

> English | [한국어](position_control_ko.md)

Firmware 261119 (Motor Tool v3.9.5) adds a new mode for the position loop, shown as "new position
control" in the tools and set with the `or_ctl` terminal command. It is off by default; with it off,
the board behaves as in earlier firmware.

Boards: SPN-MC V2 / V2R2, SPN-MC D17_D20, SPN-MC D14, VESCAT HIGH V2R2 / V2R3 / V2R4.

## 1. Overview

The existing position loop (mcconf `p_pid_kp / ki / kd`) generates torque from the position error only.
Overcoming friction or holding a load requires a larger error, higher gains cause overshoot, and the
gains must be tuned separately for each actuator and load.

The new position control sets its gains from the measured inertia J and computes the current command (A) as:

```
iq = kp * (p_ref - p) + kd * (v_ref - v) + J * a_ref + d_hat

kp = J * (2*pi*bw)^2          bw   : loop bandwidth (default 31.5 Hz)
kd = J * 2*zeta*(2*pi*bw)     zeta : damping ratio  (default 1.0)
```

- `J * a_ref`: inertia feed-forward. Adds the current needed for the target acceleration.
- `d_hat`: output of the disturbance observer (DOB, default 15 Hz). Estimates and compensates torque not
  in the model, such as friction, gravity and constant loads. No steady-state error without an integral term.
- kp and kd scale with J, so the response characteristics stay the same for different loads. J has to be
  re-measured when the load changes.

Positions are in motor shaft degrees. Output angle = motor angle / gear ratio. J is in A/(deg/s²) at the
motor shaft.

## 2. Measuring J (Inertia Tune)

### Motor Tool

Parameter tab, Inertia Tune button below Detect Encoder.

1. Connect the board and fix the load to the output. For a bare actuator, run without a load.
2. Check that the output can move by the amplitude in both directions. Default amplitude is 15° at the
   motor (15° / gear ratio at the output).
3. Check Turn on and Save to apply the result immediately and keep it after reboot.
4. Click Inertia Tune. The board applies 3, 6 and 10 Hz sine excitation around the present position,
   twice. Takes about 30 s plus communication time. Stop ends after the current excitation.
5. J is computed from the measured current and position, and checked for:
   - fit between model and measurement
   - agreement between the 6 Hz and 10 Hz results
   - agreement between run 1 and run 2
   - current limit reached or not
6. On pass, J and the commands to be sent (`or_ctl j <J>`, `or_ctl on`, `or_ctl save`) are shown and sent
   after confirmation. On fail, nothing is applied.

Failures are mostly caused by a load that is not fixed rigidly (a weight on a string, coupling backlash,
fixture resonance). Fix the load or change the amplitude and measure again.

### ST-Link Tool (production)

Production step 14 measures J of the bare actuator, enables the new position control and saves it. The
board is rebooted to confirm the setting was stored, and the closed-loop frequency response and step
response are recorded in the test report. Units that pass step 14 ship with the new position control on;
if step 14 is skipped, they ship with it off.

An actuator tuned bare and then installed in a joint has a larger inertia. The disturbance observer keeps
it stable and accurate, but to get the designed response, run Inertia Tune again in the assembled state.

## 3. Commands and settings

Enter in the Motor Tool terminal or the VESC Tool terminal.

| Command | Range | Default | Description |
|---|---|---|---|
| `or_ctl` | | | Show state, J, bw, zeta, kp / kd, DOB estimate, trajectory acceleration |
| `or_ctl on` / `off` | | off | Enable / disable the new position control (J required) |
| `or_ctl j <J>` | 1e-6 .. 1e-3 | 0 | Inertia, A/(deg/s²) at the motor shaft |
| `or_ctl bw <Hz> [zeta]` | 5..60 Hz, 0.5..2 | 31.5 Hz, 1.0 | Loop bandwidth, damping ratio |
| `or_ctl dob <Hz>` | 5..30 Hz | 15 Hz | Disturbance observer bandwidth |
| `or_ctl amax <deg/s²>` | 1000..300000 | 25000 | 0xA4 trajectory acceleration |
| `or_ctl save` | | | Save the above to EEPROM |
| `or_ctl clear` | | | Delete saved values (defaults and off from the next boot) |

Values changed with `or_ctl` apply immediately but are lost on reboot unless saved with `or_ctl save`.
`or_eeprom_default all` also deletes them.

Setting guide:
- bw: higher gives more stiffness and faster response, but more noise at standstill and closer to the
  mechanical resonance. The default is set for SPN actuators. Increase in small steps.
- zeta: 1.0 gives no overshoot from the loop itself. Lower values respond faster with overshoot.
- dob: higher reacts faster to load changes but increases current noise. Keep it well below bw.
- amax: lower for loads that swing easily (long arms, hanging loads), higher for fast point-to-point moves.

## 4. Behaviour by command

| Command | With the new position control on |
|---|---|
| 0xA4 position move | Trapezoidal trajectory (acceleration `amax`, max speed from the DPS Vmax setting). No long slow-down near the target as in the old servo mode. |
| 0xA3 single position command | Small changes go directly to the loop. Large changes that would need more than about 3 A of position-error current follow the same trapezoidal trajectory as 0xA4. |
| 0xA3 continuous commands (interval below 50 ms) | The command period is measured and the commands are replayed on an internal clock with about 1.5 periods of delay. Jitter in command arrival does not reach the motor. |
| 0xA2 speed (DPS) | Uses the same inner loop. |
| MIT (0xC0), current commands | Not affected. |

Command rates: over CAN, 0xA3 is received without loss up to 1 kHz. Over EtherCAT, use a 2 ms (500 Hz)
cycle for continuous 0xA3 on SPN-MC V2 class boards.

The communication watchdog (`or_wd`) does not monitor position and speed commands by default. A single
0xA3 / 0xA4 can be sent and the position held.

## 5. Changes independent of `or_ctl`

- The position target is interpolated inside the control loop, and the D term uses the measured speed
  instead of the error derivative. Less current ripple at standstill and no current spike on a target change.
- Switching from MIT or current mode to a position command starts from the present position.
- When a position command is switched to the trajectory during rotation, the trajectory starts from the
  present speed without stopping first.

## 6. Checking the setting

```
> or_ctl
or_ctl ON: J 2.511e-05 A/(deg/s^2), bw 31.5 Hz zeta 1.00 -> kp 0.9836 A/deg kd 0.009940 A/(deg/s)
  DOB 15.0 Hz: d_hat 0.142 A, mean |d_hat| 0.150 A; servo trajectory amax 25000 deg/s^2
```

- `ON` and the J you set mean the setting is applied. If saved, the same values appear after reboot.
- `d_hat` at standstill is the load the observer is compensating (friction, gravity). If it grows over time
  without a change in load, check the mechanics.
