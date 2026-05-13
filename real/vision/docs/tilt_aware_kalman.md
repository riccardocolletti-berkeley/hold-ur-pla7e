# Tilt-aware Kalman filter

The vision tracker estimates ``[x, y, vx, vy]`` from monocular ArUco
measurements. A pure constant-velocity model is structurally wrong on a
tilting plate (the ball is always accelerating under projected gravity),
and absorbing that acceleration into the process-noise term forces a
trade-off between noise and lag. The plate tilt is *known* at every
instant (it is the controller's own command), so feeding it into the
filter as a deterministic control input lets the model account for the
expected acceleration directly. Process noise then only has to absorb
unmodelled effects (slip, rolling resistance, sin-approximation error,
tilt-actuator backlash).

## Physics

For a uniform solid sphere rolling without slip on a plate tilted by
``(θ_x, θ_y)`` (rotations about the table axes), the in-plane
acceleration of the ball centre, expressed in the table frame, is

    a_x = +α · g · sin(θ_y)
    a_y = -α · g · sin(θ_x)

with ``g = 9.81 m/s²`` and ``α = 1 / (1 + I / (m·r²)) = 5/7`` for a
uniform solid sphere. Use ``α = 1`` for pure sliding.

Sign convention (matches `MARKER_POSITIONS_TABLE` in `vision.config`):

- ``θ_y > 0``  ⇒ the +x edge of the plate dips down ⇒ ball rolls toward +x ⇒ ``a_x > 0``.
- ``θ_x > 0``  ⇒ the +y edge of the plate rises up ⇒ ball rolls toward -y ⇒ ``a_y < 0``.

Tilt the plate by hand, command zero motor output, and verify that the
predicted velocity arrow in the debug overlay points the same way the
ball actually rolls. If the directions are reversed, flip the matching
sign in :func:`vision.tracker.kalman_filter.tilt_to_accel`.

## Discrete-time model

State ``x = [x, y, vx, vy]ᵀ`` (m, m/s); measurement ``z = [x, y]ᵀ``;
control ``u = [a_x, a_y]ᵀ`` (m/s²); time step ``Δt``:

        ┌1  0  Δt  0 ┐         ┌Δt²/2     0  ┐
   F =  │0  1   0  Δt│    B =  │   0   Δt²/2 │
        │0  0   1   0│         │  Δt      0  │
        └0  0   0   1┘         └   0     Δt  ┘

`H` and `R` are unchanged from the constant-velocity case. `Q` keeps the
continuous-white-noise-acceleration structure with ``σ_a`` interpreted as
the unmodelled acceleration noise (typically ~10⁻³ m/s²). Predict adds
``B · u`` to the state; update is unchanged.

## API

`vision.tracker.kalman_filter`:

- `tilt_to_accel(theta_x, theta_y) -> (a_x, a_y)`: table-frame helper.
- `BallKalmanFilter.set_control(a_x, a_y)`: store the latest control input.
- `BallKalmanFilter.predict(dt)`. F/B/Q are rebuilt automatically when ``dt`` changes.
- Default ``u`` is ``[0, 0]``, so a filter that never receives a control input
  behaves exactly like a constant-velocity estimator.

`vision.tracker.pipeline.TrackingPipeline`:

- `set_plate_angles(theta_x, theta_y)`: call after every controller command.
- The pipeline converts ``(θ_x, θ_y)`` to ``(a_x, a_y)`` via `tilt_to_accel`
  and pushes them into the filter before the predict step.

## Validation

Three sanity checks, in increasing cost:

1. **Direction.** Hold the plate at a fixed tilt by hand. Disable the
   controller, push ``set_plate_angles(θ_x, θ_y)`` from a small script,
   release the ball at rest. The KF velocity arrow should point the same
   way the ball rolls. If reversed, flip a sign in `tilt_to_accel`.
2. **Magnitude.** Tilt the plate to a measured angle (e.g. 5°). After
   ~0.3 s the KF velocity should match ``α · g · sin(θ) · t`` to within
   ~10 %. A larger discrepancy indicates slipping (try ``α = 1``) or a
   wrong angle calibration.
3. **Closed-loop comparison.** With the plate flat, push the ball by
   hand. Log the KF velocity against the numerical derivative of the
   raw measurement; a tilt-aware filter should be both smoother and less
   laggy at the same ``KF_PROCESS_NOISE``.

## Limitations

- **Commanded ≠ achieved tilt.** Backlash, low servo bandwidth, or
  saturation make the model diverge from reality. Mitigation: feed the
  *measured* plate angle (e.g. from servo encoders) instead of the
  command, or augment with a first-order plate-dynamics model.
- **Rolling factor.** ``α = 5/7`` assumes rolling without slipping. A
  smooth ball on a smooth plate slips during fast direction changes;
  the effective ``α`` then sits between ``5/7`` and ``1``. Treat it
  as a tunable.
- **Parallax bias.** The detected ball centre sits at height ``r``
  above the marker plane; the homography ignores this offset.
