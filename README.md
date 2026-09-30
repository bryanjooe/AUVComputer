# ugv_nav - vision-only UGV navigation in PyBullet

Status: `sim.py` (this step). `perception.py`, `vo.py`, `planner.py`, `main.py` come next.

## Coordinate frames (defined once, used by every module)

| Frame | Origin | Axes | Notes |
|---|---|---|---|
| **World** | terrain centre, z = 0 | x forward/east, y left/north, z up | yaw is CCW about +z; yaw = 0 points along +x |
| **Base** (robot) | Husky base link (COM) | x forward, y left, z up | `get_true_pose()` = (x, y, yaw) of this frame in World |
| **Camera** (OpenCV optical) | camera centre | x right, y down, z forward | what `solvePnP`, `K` and `depth` use |

Terrain: 40 x 40 m, x, y in [-20, 20]; heightfield 161 x 161 samples (0.25 m).
Start is sampled in x in [-17, -14], goal in x in [14, 17]; y in [-14, 14] for both.

### Camera extrinsics (fixed, base -> camera)

- Position in base frame: **t = (0.35, 0.00, 0.55) m** (forward of centre, above the deck)
- Orientation: pitched **15 deg down**, no roll/yaw
- Camera axes expressed in base frame (columns of `R_base_cam`):
  `x_cam (right) = (0, -1, 0)`, `y_cam (down) = (-sin15, 0, -cos15)`, `z_cam (forward) = (cos15, 0, -sin15)`
- `sim.T_base_cam` is the 4x4 matrix; `sim.get_camera_pose()` returns the current `T_world_cam`
  (ground truth - for debugging/evaluation only).

### Intrinsics (320 x 240, vertical FOV 60 deg, square pixels)

```
K = [[207.85,   0.00, 159.5],
     [  0.00, 207.85, 119.5],
     [  0.00,   0.00,   1.0]]      # sim.K
```
`depth` is metric **z-depth** (distance along the optical axis, not Euclidean range), range
0.1-30 m; pixels with no geometry (sky) are set to exactly 30.0. Depth has Gaussian noise
sigma = 0.005 + 0.0015 z^2 m. Back-project a pixel with `X_cam = z * K^-1 [u, v, 1]^T`.

### Mask classes from `get_camera()` (uint8)

`0` traversable ground - `1` obstacle (rock, tree, moving object; **sky is also 1**, it has depth = 30 m so
the planner drops it) - `2` hazard (ditch or slope steeper than ~27 deg; found by back-projecting terrain pixels
onto a hazard grid, so ditch pixels are labelled exactly).

## Run
```
pip install -r requirements.txt
python test_sim.py                 # writes ./debug/*.png
```
