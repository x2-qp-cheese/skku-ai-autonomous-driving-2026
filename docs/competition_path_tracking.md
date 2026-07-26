# Competition path tracking

## Why the controller changed

The old competition command used pure pursuit against one BEV lookahead point.
In `20260725_175616_debug.mp4`, the far lateral error repeatedly reached about
`-0.52` to `-0.62` while the near error stayed around `-0.10` to `-0.14`.
That single far point produced steering near `-120` to `-142`, even though the
vehicle was much closer to the desired line at its front. A dashed-line fit or
the next half of an S-curve could therefore dominate the whole command.

The competition controller now keeps the fitted BEV centerline as a sequence of
path points. It computes a weighted lateral error over the complete interval
between the far and near control rows, adds the fitted path heading and a small
error derivative, then applies asymmetric steering smoothing:

```text
e_path = weighted_mean((path_x - vehicle_x) / half_bev_width)
u_raw  = K_path * e_path + K_heading * heading + K_d * delta(e_path)
u      = EMA(u_previous, u_raw)
```

Near points receive more weight because they describe the lane occupied by the
vehicle now. Far points remain in the mean, so curves are still anticipated.
This is a preview, Stanley-style lateral and heading controller adapted to the
available camera-only BEV geometry. It is not copied external source code.

## Path-shape stabilization

YOLO can expose different portions of a dashed line on consecutive frames.
Sampling each fit only over its currently visible rows made the path points move
to different BEV distances, so the former point-by-point EMA frequently reset.
The runtime now evaluates every accepted fit at the same 24 longitudinal BEV
anchors.

Each anchor uses bounded-innovation EMA filtering. Far anchors use stronger
filtering because small segmentation changes create large preview changes;
anchors near the vehicle respond faster so curve entry is not delayed. A
measurement can move one anchor by at most `80 px` per processed frame before
filtering. Center error, near error, heading and steering are all derived from
this one stabilized path rather than from separate, potentially conflicting
filters. The competition launcher therefore uses heading smoothing `1.0`: the
path is already filtered and its heading must not pass through a second lagging
EMA.

A quadratic is used only inside the rows actually supporting the lane fit.
Outside that interval the path continues along the polynomial's endpoint
tangent, whose slope is bounded to `[-1, 1]`. This avoids the numerical
divergence caused by extending a sparse quadratic to the full image. Fixed
anchors, bounded innovation and tangent continuation are local filtering and
geometry rules implemented in this project, not copied external algorithms.

The complete path is also checked as a forward function `x(y)`. Its tangent and
frame-to-frame tangent change are bounded from the near field toward the far
field, so two independently fitted mask fragments cannot form a V-shaped or
near-horizontal splice. Points beyond the actual steering lookahead are drawn
as a tangent continuation of the control path. They remain useful for visual
preview but cannot bend the heading estimate away from the lane being driven.

## Technical basis

- Sebastian Thrun et al., *Stanley: The Robot that Won the DARPA Grand
  Challenge*, Journal of Field Robotics 23(9), 2006. The controller's use of
  cross-track error together with path heading is the basis for separating
  lateral position from road direction:
  <https://robots.stanford.edu/papers/thrun.stanley05.pdf>
- R. Craig Coulter, *Implementation of the Pure Pursuit Path Tracking
  Algorithm*, CMU-RI-TR-92-01, 1992. This is the basis for retaining forward
  preview information. The project no longer lets one preview point own the
  complete steering command:
  <https://publications.ri.cmu.edu/implementation-of-the-pure-pursuit-path-tracking-algorithm>

The implementation intentionally does not introduce MPC. This vehicle has no
reliable wheel speed, steering-angle feedback, wheelbase calibration or pose
estimate in the current runtime. An MPC model would therefore add parameters
that cannot be identified before the competition and would not have a sound
technical basis.

## Crosswalk and obstacle rules

- Crosswalk: never fit zebra stripes as lane boundaries. Continue evaluating the
  current center marking and outer boundary every frame and prefer that fresh,
  bounded path whenever either valid corridor tier is available.
- Crosswalk fallback: only while lane markings are hidden, use the last reliable
  current-lane path. Advance that vehicle-relative cache by the measured BEV
  motion of the zebra mask instead of freezing the entry pose. Reacquire a valid
  current path immediately; do not wait behind a stale heading-jump gate.
- Crosswalk ownership: pause lane-change timers and preserve only the already
  established parallel lane offset. The obstacle layer no longer keeps a second
  frozen path cache.
- Obstacle: assign each instance exclusively to the closer path, with overlap and
  pixel-distance hysteresis. A current-lane obstacle that merely touches the
  projected destination corridor cannot mark both lanes blocked. Adjacent-lane
  paths retain their parallel geometry outside the BEV image; only normalized
  control error is saturated. Clipping the path itself would create a false kink
  and make its heading inconsistent.
- Emergency stop: obstacle emergency braking is disabled by the competition
  launcher. The traffic-light controller remains the only normal mission brake.
- Speed: every non-brake command is finalized at `255`.

## Control ownership

The runtime has one owner for each decision. The BEV corridor owns the 24-point
base path. The lane-change controller may translate that complete path, but it
does not add a second steering boost or steering minimum. The path follower is
the only component that converts geometry to steering. When obstacle avoidance
geometry is unreliable, the final steering is bounded to `[-90, 90]` in every
active state, including lane-1 hold and stabilization.

The obstacle layer cannot advance its planner during a crosswalk. The traffic
light may stop the final command, and no other mission logic may brake it. This
ordering is intentional:

```text
YOLO masks -> fresh bounded BEV path or motion-adjusted fallback
           -> fixed existing lane offset during crosswalk
           -> whole-path steering
           -> traffic-light stop -> fixed-speed finalization
```

The competition launcher fixes the obstacle lane translation to `150 px`.
Using a single configured width prevents the obstacle planner and lane-change
controller from reasoning about different destination corridors.

## Center-offset validation

The competition centerline bias remains `0.50`, the geometric midpoint between
the detected center marking and outer boundary. It was not shifted inward to
hide curve lag.

Six user-provided runs were sampled at an effective 10 fps:
`20260714_103213`, `20260725_142621`, `20260725_144632`,
`20260725_145920`, `20260725_150840`, and `20260725_175616`. Only tier-1
frames with both boundaries detected were included. With the former
`path_smooth=0.36`, `path_max_step=28` settings, the median path position was
correct at `0.5000`, but the median per-video 90th-percentile center error was
`68.11 px`. This explains an apparently outer path on curve transitions: the
offset was correct, but the temporal path was late.

With `path_smooth=0.90` and `path_max_step=80`, the median path position is
`0.5000` and the same 90th-percentile error falls from `68.11 px` to
`10.20 px`. Across all six videos, strong steering sign reversals remain zero
and the median per-video 90th-percentile steering delta is `9.5`.

The complete fitted path was also checked, not only the lookahead target.
Across `35,916` valid path-point comparisons, each video's median lane-position
ratio is between `0.4993` and `0.5002`; the aggregate median is `0.5000`.
The median per-video whole-path 90th-percentile error is `10.43 px`, and the
largest absolute median signed offset is `0.11 px`. The camera-frame arithmetic
midpoint is not used for validation: a projective homography does not preserve
midpoints, so the physical BEV midpoint can appear closer to the outer line in
a distant camera row. The reproducible report is
`data/processed/center_offset_analysis.json`, generated by
`scripts/analyze_center_offset.py`.
