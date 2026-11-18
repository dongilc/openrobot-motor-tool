# Inertia Tune and the New Position Control

> English | **[한국어](position_control_ko.md)**

Firmware 261119 (Motor Tool v3.9.5) adds a second way to run the position loop. It is called **new
position control** in the tools and `or_ctl` on the terminal. It is **off by default**: a board
behaves as before until it is turned on.

Boards: SPN-MC V2 / V2R2, SPN-MC D17_D20, SPN-MC D14, VESCAT HIGH V2R2 / V2R3 / V2R4.

## 1. What it changes

The existing position loop (mcconf `p_pid_kp / ki / kd`) makes torque from the position error only.
To push through friction or hold a load it needs a large error, so stiff settings come with overshoot,
and the gains have to be found by hand for each actuator and load.

The new position control computes the current command directly in amps, from one measured number,
the **inertia J**:

```
iq = kp * (p_ref - p) + kd * (v_ref - v) + J * a_ref + d_hat

kp = J * (2*pi*bw)^2          bw   = loop bandwidth  (default 31.5 Hz)
kd = J * 2*zeta*(2*pi*bw)     zeta = damping         (default 1.0)
```

- **Inertia feed-forward** `J * a_ref`: the current needed to accelerate is applied up front, so the
  shaft does not have to fall behind first.
- **Disturbance observer** `d_hat` (DOB, default 15 Hz): estimates the torque the model does not
  explain (friction, gravity, a constant load) and cancels it. The shaft settles on target with no
  integral term and no droop under a hanging load.
- **Gains follow J**: a heavier load gets proportionally larger gains, so the response stays the same
  shape. Re-measure J when the load changes.

All positions are **motor shaft degrees** (output = motor / gear ratio). J is in A/(deg/s²) at the
motor shaft.

## 2. Measuring J (Inertia Tune)

### Motor Tool

Parameter tab, below Detect Encoder: **Inertia Tune**.

1. Connect the board, attach the load you will run with (or nothing, for a bare actuator).
   The load must be fixed rigidly to the output.
2. Make sure the output can move freely by ± the amplitude (default 15° motor = 15° / gear ratio at the
   output).
3. Leave **Turn on** and **Save** checked if the result should be used right away and kept after a reboot.
4. Press **Inertia Tune**. The board holds its present position and shakes around it with a sine at
   3, 6 and 10 Hz, twice. This takes about 30 seconds plus communication time. **Stop** ends after the current shake.
5. The tool solves J from the measured current and motion and checks the result:
   - how well the model fits the measurement,
   - whether 6 Hz and 10 Hz give the same J,
   - whether the two runs agree,
   - whether the current hit its limit.
6. On a pass it shows J and the commands it will send (`or_ctl j <J>`, `or_ctl on`, `or_ctl save`) and
   asks before sending. On a failure nothing is applied.

A failure usually means the load is not rigid: a weight hanging on a string, a loose coupling, or a
resonance in the fixture. Fix the load or change the amplitude and try again.

### ST-Link Tool (production)

Step 14 of the production inspection measures J on the bare actuator, turns the new position control
on and saves it, reboots the board to confirm it was stored, then records the closed-loop frequency
response and a step response in the test report. Units that pass step 14 ship with the new position
control on. If step 14 is skipped, the unit ships with it off.

A unit tuned bare and then built into a robot joint carries more inertia. The disturbance observer
keeps it stable and on target, but for the intended response re-run Inertia Tune with the joint attached.

## 3. Commands and settings

Terminal (Motor Tool terminal, or VESC Tool terminal):

| Command | Range | Default | Meaning |
|---|---|---|---|
| `or_ctl` | | | Show state, J, bw, zeta, resulting kp / kd, DOB estimate, trajectory limit |
| `or_ctl on` / `off` | | off | Use / stop using the new position control (needs J) |
| `or_ctl j <J>` | 1e-6 .. 1e-3 | 0 | Inertia, A/(deg/s²) at the motor shaft |
| `or_ctl bw <Hz> [zeta]` | 5..60 Hz, 0.5..2 | 31.5 Hz, 1.0 | Loop bandwidth and damping |
| `or_ctl dob <Hz>` | 5..30 Hz | 15 Hz | Disturbance observer bandwidth |
| `or_ctl amax <deg/s²>` | 1000..300000 | 25000 | Acceleration of the 0xA4 trajectory |
| `or_ctl save` | | | Store all of the above in EEPROM (kept over reboot) |
| `or_ctl clear` | | | Remove the stored values (off with defaults after the next reboot) |

Settings changed with `or_ctl` take effect at once but are lost on reboot until `or_ctl save`.
`or_eeprom_default all` also clears them.

Guidelines:
- **bw**: higher is stiffer and faster, and also louder at standstill and closer to the mechanical
  resonance of the actuator. The default suits the SPN actuators; raise it in small steps.
- **zeta**: 1.0 gives no overshoot on the loop itself. Lower values answer faster with some overshoot.
- **dob**: higher rejects load changes faster but makes the current noisier. Keep it well below bw.
- **amax**: lower it for loads that swing (long arms, hanging loads); raise it for faster point-to-point moves.

## 4. Behaviour by command

| Command | With the new position control on |
|---|---|
| **0xA4** position move | Time-optimal trapezoid to the target (acceleration `amax`, speed limited by the DPS Vmax setting). Arrives in finite time without the slow final approach of the old servo mode. |
| **0xA3** single position target | Small jumps go straight to the loop. A jump that would need more than about 3 A of position-error current goes through the same trapezoid instead of a current spike. |
| **0xA3** stream (commands less than 50 ms apart) | The firmware measures the command period and replays the stream on its own clock, about 1.5 periods behind, so jitter in arrival times does not reach the motor. |
| **0xA2** speed (DPS) | Uses the same inner loop. |
| **MIT** (0xC0) and **current** commands | Not affected. |

Stream rates: over CAN, 0xA3 streams up to 1 kHz are received in full. Over EtherCAT, use a 2 ms
(500 Hz) cycle for 0xA3 streams on SPN-MC V2 class boards.

The command watchdog (`or_wd`) does not watch position and speed commands by default, so a single
0xA3 / 0xA4 can be sent once and held.

## 5. Also in this firmware, with or without `or_ctl`

- The position reference is interpolated inside the control loop, and the D term uses the measured
  speed instead of the change of the error. Standstill current ripple is lower and a new target no
  longer gives a kick.
- Switching from MIT or current mode to a position command continues smoothly from the present position.
- A position command that is redirected to the trajectory while the shaft is moving starts the
  trajectory from the present speed instead of braking first.

## 6. Checking it works

```
> or_ctl
or_ctl ON: J 2.511e-05 A/(deg/s^2), bw 31.5 Hz zeta 1.00 -> kp 0.9836 A/deg kd 0.009940 A/(deg/s)
  DOB 15.0 Hz: d_hat 0.142 A, mean |d_hat| 0.150 A; servo trajectory amax 25000 deg/s^2
```

- `ON` and the J you expect: the setting is active. After a reboot it should read the same if it was saved.
- `d_hat` at standstill is the load the observer is carrying (friction, gravity). A value that grows
  over time with no change in load points to a mechanical problem.
