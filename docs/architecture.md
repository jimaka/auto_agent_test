# Architecture

## State (INS, body-frame velocities)

`x = [x, y, psi, u, v, r]` — position/heading in local frame; `u,v,r` from INS in ship frame.

## Control outputs

`u = [delta_cmd_rad, rpm_cmd]` published on `/control/cmd`.

## Koopman

`z = phi(x) in R^64`, `z_{k+1} = A z_k + B u_k`. No separate EKF: measured `x` from INS; lift for prediction only.

## Timing

- Control: 4 Hz (`Ts = 0.25 s`), budget < 300 ms
- Model update: ~6 months offline via `vessel_identification`

## Modes

- `koopman_mpc` — `vessel_control`
- `baseline` — `vessel_baseline` (Nomoto MPC stub)

## Units

Internal SI (rad, m, m/s). Rudder sensor may be degrees at the driver boundary only.
