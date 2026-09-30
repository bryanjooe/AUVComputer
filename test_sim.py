"""test_sim.py - smoke test for sim.py.

    python test_sim.py                       # seed 0, medium difficulty, headless
    python test_sim.py --seed 3 --difficulty hard
    python test_sim.py --mode goal           # drive start -> goal (records a longer video)
    python test_sim.py --gui                 # watch the drive live

Checks: (1) same seed -> identical world, frames and rollout, (2) camera geometry is
self-consistent, (3) drive + MP4 video of the traversal (camera views + top-down map), (4) drive calibration.
Exit code is non-zero if a PASS/FAIL check fails.
"""
from __future__ import annotations

import argparse
import math
import os
import shutil
import subprocess
import sys
import time

import cv2
import numpy as np

import sim as S

PALETTE_BGR = np.array([[60, 180, 60],     # 0 traversable  (green)
                        [40, 40, 220],     # 1 obstacle/sky (red)
                        [0, 140, 255]],    # 2 hazard       (orange)
                       np.uint8)


def k_str(K: np.ndarray) -> str:
    return np.array2string(K, precision=2, suppress_small=True, separator=", ").replace("\n", "")


def script(i: int) -> tuple[float, float]:
    """Scripted (v, w): straight, left arc, straight, right arc, straight."""
    if i < 80:
        return 0.6, 0.0
    if i < 130:
        return 0.4, 0.5
    if i < 220:
        return 0.6, 0.0
    if i < 270:
        return 0.4, -0.5
    return 0.6, 0.0


def goal_command(pose, goal) -> tuple[float, float]:
    """Crude go-to-goal using GROUND-TRUTH pose (test/demo only; ignores obstacles)."""
    x, y, yaw = pose
    err = (math.atan2(goal[1] - y, goal[0] - x) - yaw + math.pi) % (2 * math.pi) - math.pi
    v = 0.6 * max(0.0, math.cos(err)) if abs(err) < 1.2 else 0.0
    return v, float(np.clip(1.5 * err, -1.0, 1.0))


def to_debug_images(rgb, depth, mask):
    depth_c = cv2.applyColorMap(np.clip(depth / 15.0 * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), depth_c, PALETTE_BGR[mask]


def save_frame(out: str, idx: int, rgb, depth, mask) -> None:
    bgr, depth_c, mask_c = to_debug_images(rgb, depth, mask)
    cv2.imwrite(f"{out}/frame{idx:02d}_rgb.png", bgr)
    cv2.imwrite(f"{out}/frame{idx:02d}_depth.png", depth_c)
    cv2.imwrite(f"{out}/frame{idx:02d}_mask.png", mask_c)


def label(img: np.ndarray, text: str, org=(6, 16), color=(255, 255, 255), scale=0.5) -> None:
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def verdict(name: str, cond: bool, detail: str = "") -> bool:
    """Print a PASS/FAIL line and return the boolean result."""
    cond = bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    return cond


class MiniMap:
    """Top-down (north-up) view: terrain height, hazards, rocks, trees, movers, trail, camera FOV."""
    SIZE = 480

    def __init__(self, spec: S.WorldSpec, mover_fn):
        self.spec, self.mover_fn, self.trail = spec, mover_fn, []
        self.k = self.SIZE / S.WORLD_SIZE                                # pixels per metre
        g = cv2.normalize(spec.heights, None, 60, 230, cv2.NORM_MINMAX).astype(np.uint8)
        base = np.ascontiguousarray(
            np.flipud(cv2.resize(cv2.applyColorMap(g, cv2.COLORMAP_SUMMER), (self.SIZE, self.SIZE))))
        haz = np.flipud(cv2.resize(spec.hazard.astype(np.uint8), (self.SIZE, self.SIZE),
                                   interpolation=cv2.INTER_NEAREST)) > 0
        base[haz] = (0.45 * base[haz] + 0.55 * np.array([0, 0, 255])).astype(np.uint8)
        for m in range(-20, 21, 10):                                     # 10 m grid
            p0, p1 = self.w2p(m, -20), self.w2p(m, 20)
            cv2.line(base, p0, p1, (90, 90, 90), 1)
            cv2.line(base, self.w2p(-20, m), self.w2p(20, m), (90, 90, 90), 1)
        for r in spec.rocks:
            rad = r["size"][0] if r["kind"] == "sphere" else max(r["size"][:2])
            cv2.circle(base, self.w2p(r["x"], r["y"]), max(2, int(rad * self.k)), (150, 150, 150), -1)
        for t in spec.trees:
            cv2.circle(base, self.w2p(t["x"], t["y"]), max(3, int(t["radius"] * self.k * 1.5)), (20, 90, 20), -1)
        cv2.circle(base, self.w2p(*spec.start), 7, (255, 255, 0), 2)     # start: cyan ring
        cv2.drawMarker(base, self.w2p(*spec.goal), (255, 0, 255), cv2.MARKER_STAR, 20, 2)   # goal: magenta star
        self.base = base
        self.hfov = 2.0 * math.atan((S.IMG_W / 2.0) / S.compute_intrinsics()[0, 0])

    def w2p(self, x: float, y: float) -> tuple[int, int]:
        return int((x + S.HALF) * self.k), int(self.SIZE - (y + S.HALF) * self.k)

    def render(self, pose) -> np.ndarray:
        x, y, yaw = pose
        self.trail.append(self.w2p(x, y))
        img = self.base.copy()
        cone = [self.w2p(x, y)] + [self.w2p(x + 8 * math.cos(yaw + a), y + 8 * math.sin(yaw + a))
                                   for a in np.linspace(-self.hfov / 2, self.hfov / 2, 12)]
        ov = img.copy()
        cv2.fillPoly(ov, [np.array(cone, np.int32)], (255, 255, 255))
        img = cv2.addWeighted(ov, 0.28, img, 0.72, 0)                    # camera view cone (8 m)
        if len(self.trail) > 1:
            cv2.polylines(img, [np.array(self.trail, np.int32)], False, (0, 255, 255), 2, cv2.LINE_AA)
        for mx, my in self.mover_fn():
            cv2.circle(img, self.w2p(mx, my), 6, (0, 140, 255), -1)
            cv2.circle(img, self.w2p(mx, my), 6, (0, 0, 0), 1)
        tri = [self.w2p(x + 0.9 * math.cos(yaw), y + 0.9 * math.sin(yaw)),
               self.w2p(x + 0.6 * math.cos(yaw + 2.6), y + 0.6 * math.sin(yaw + 2.6)),
               self.w2p(x + 0.6 * math.cos(yaw - 2.6), y + 0.6 * math.sin(yaw - 2.6))]
        cv2.fillPoly(img, [np.array(tri, np.int32)], (255, 255, 255))
        cv2.polylines(img, [np.array(tri, np.int32)], True, (0, 0, 0), 1, cv2.LINE_AA)
        label(img, "top-down, north up (10 m grid)", (6, 16))
        for i, (txt, col) in enumerate((("hazard", (80, 80, 255)), ("rock", (200, 200, 200)),
                                        ("tree", (60, 180, 60)), ("moving obstacle", (0, 160, 255)),
                                        ("start / goal", (255, 255, 0)))):
            label(img, txt, (6, self.SIZE - 8 - 16 * i), col, 0.42)
        return img


def compose_frame(rgb, depth, mask, minimap: np.ndarray, info: list[str], alert: bool) -> np.ndarray:
    """960x720 BGR frame: [RGB | depth | mask] on top, [minimap | info panel] below."""
    bgr, depth_c, mask_c = to_debug_images(rgb, depth, mask)
    for im, name in ((bgr, "RGB"), (depth_c, "depth (0-15 m)"), (mask_c, "class mask")):
        label(im, name)
    panel = np.full((MiniMap.SIZE, 480, 3), 30, np.uint8)
    for i, line in enumerate(info):
        label(panel, line, (14, 28 + 26 * i), (255, 255, 255), 0.55)
    if alert:
        cv2.rectangle(panel, (0, 0), (479, MiniMap.SIZE - 1), (0, 0, 255), 6)
    return np.vstack([np.hstack([bgr, depth_c, mask_c]), np.hstack([minimap, panel])])


class Recorder:
    """cv2.VideoWriter with an .avi/MJPG fallback if the mp4v codec is unavailable."""

    def __init__(self, base_path: str, fps: float, size: tuple[int, int]):
        for ext, cc in ((".mp4", "mp4v"), (".avi", "MJPG")):
            w = cv2.VideoWriter(base_path + ext, cv2.VideoWriter_fourcc(*cc), fps, size)
            if w.isOpened():
                self.writer, self.path, self.closed = w, base_path + ext, False
                return
        raise RuntimeError("Could not open a video writer (see README: failure #4)")

    def write(self, frame: np.ndarray) -> None:
        self.writer.write(frame)

    def close(self) -> str:
        """Finish the file. OpenCV can only write MPEG-4 Part 2 ('mp4v'), which Ubuntu's Video Player
        cannot decode, so re-encode to H.264 in place when ffmpeg is available."""
        if self.closed:
            return self.path
        self.closed = True
        self.writer.release()                               # writes the MP4 index (moov atom)
        if not self.path.endswith(".mp4"):
            return self.path
        ff = shutil.which("ffmpeg")
        if ff is None:
            print("NOTE: ffmpeg not found, so the video is MPEG-4 Part 2 and Ubuntu's Video Player will not "
                  "play it.\n      Fix: sudo apt install -y ffmpeg   then re-run, or convert with:\n"
                  f"      ffmpeg -i {self.path} -c:v libx264 -pix_fmt yuv420p out.mp4   (VLC also plays it as is)")
            return self.path
        tmp = self.path[:-4] + "_h264_tmp.mp4"
        r = subprocess.run([ff, "-y", "-loglevel", "error", "-i", self.path, "-c:v", "libx264",
                            "-pix_fmt", "yuv420p", "-movflags", "+faststart", tmp],
                           capture_output=True, text=True)
        if r.returncode == 0:
            os.replace(tmp, self.path)                      # same name, now H.264
        else:
            print("WARNING: ffmpeg re-encode failed:", r.stderr.strip()[:300])
        return self.path


# ------------------------------------------------------------------ 1. reproducibility
def check_reproducibility(seed: int, difficulty: str) -> bool:
    print("\n== 1. Reproducibility ==")
    a, b, c = (S.Sim(seed, difficulty=difficulty), S.Sim(seed, difficulty=difficulty),
               S.Sim(seed + 1, difficulty=difficulty))
    sa, sb, sc = a.world_signature(), b.world_signature(), c.world_signature()
    print(f"world signature: seed={seed}: {sa} | again: {sb} | seed={seed + 1}: {sc}")
    ok = verdict("same seed -> identical world", sa == sb)
    ok &= verdict("different seed -> different world", sa != sc)
    fa, fb = a.get_camera(), b.get_camera()
    ok &= verdict("same seed -> identical first RGB/depth/mask frame",
                  all(np.array_equal(x, y) for x, y in zip(fa, fb)))
    for i in range(20):
        v, w = 0.5, 0.3 * math.sin(i / 5)
        a.step(v, w)
        b.step(v, w)
    pa, pb = np.array(a.get_true_pose()), np.array(b.get_true_pose())
    ok &= verdict("same seed + same commands -> same pose after 20 steps",
                  np.allclose(pa, pb, atol=1e-6), f"max diff {np.abs(pa - pb).max():.2e}")
    for s in (a, b, c):
        s.close()
    return ok


# ------------------------------------------------------------------ 2. geometry
def check_geometry(seed: int, difficulty: str) -> bool:
    print("\n== 2. Camera geometry (back-project depth, compare with terrain) ==")
    s = S.Sim(seed, difficulty=difficulty)
    rgb, depth, mask = s.get_camera()
    pts = s.pixels_to_world(depth)
    sel = (mask == S.CLS_GROUND) & (depth > 0.5) & (depth < 8.0)
    ok = verdict("enough ground pixels visible", sel.sum() > 1000, f"{int(sel.sum())} px")
    if sel.sum() > 0:
        err = pts[..., 2][sel] - s.terrain_height(pts[..., 0][sel], pts[..., 1][sel])
        med, p90 = float(np.median(err)), float(np.percentile(np.abs(err), 90))
        ok &= verdict("ground pixels back-project onto terrain surface",
                      abs(med) < 0.08 and p90 < 0.25, f"median dz={med:+.3f} m, p90 |dz|={p90:.3f} m")
    frac = np.bincount(mask.ravel(), minlength=3) / mask.size
    print(f"class fractions (ground/obstacle+sky/hazard): {np.round(frac, 3)}")
    print(f"depth range {depth.min():.2f}..{depth.max():.2f} m, sky pixels (==FAR): "
          f"{100 * np.mean(depth >= S.FAR):.1f}%")
    s.close()
    return ok


# ------------------------------------------------------------------ 3. drive + video
def scripted_drive(seed: int, difficulty: str, gui: bool, out: str, mode: str,
                   n_steps: int, save_frames: bool) -> None:
    print(f"\n== 3. Drive ({mode}, up to {n_steps} steps) + video ==")
    os.makedirs(out, exist_ok=True)
    s = S.Sim(seed, gui=gui, difficulty=difficulty)
    print(f"start={np.round(s.start, 2)} goal={np.round(s.goal, 2)} "
          f"| lighting: brightness={s.lighting['brightness']:.2f} fog={s.lighting['fog_density']:.3f}")
    mm = MiniMap(s.spec, s.mover_positions)
    rec = Recorder(f"{out}/traversal_{difficulty}_seed{seed}", 1.0 / s.control_dt, (960, 720))
    n_coll, t_step, t_cam, recover, cmd, reached = 0, [], [], 0, (0.0, 0.0), False

    def report(i: int) -> None:
        x, y, yaw = s.get_true_pose()
        print(f"step {i:3d}  pose x={x:7.3f} y={y:7.3f} yaw={yaw:6.3f} rad | in_hazard={s.in_hazard()}"
              f" | K={k_str(s.K)}")

    i = 0
    try:
      for i in range(n_steps):
          if i % 50 == 0:
              report(i)
          t0 = time.perf_counter()
          rgb, depth, mask = s.get_camera()
          t_cam.append(time.perf_counter() - t0)
          pose = s.get_true_pose()
          dist = math.hypot(s.goal[0] - pose[0], s.goal[1] - pose[1])
          if save_frames and i % 30 == 0:
              save_frame(out, i // 30, rgb, depth, mask)
          info = [f"{difficulty}   seed {seed}   mode: {mode}",
                  f"step {i}   t = {i * s.control_dt:.1f} s",
                  f"x = {pose[0]:6.2f}  y = {pose[1]:6.2f}  yaw = {math.degrees(pose[2]):6.1f} deg",
                  f"cmd  v = {cmd[0]:+.2f} m/s   w = {cmd[1]:+.2f} rad/s",
                  f"distance to goal: {dist:5.1f} m",
                  f"collisions so far: {n_coll}" + ("   [RECOVERING]" if recover > 0 else ""),
                  f"in hazard zone: {s.in_hazard()}",
                  f"fog: {s.lighting['fog_density']:.2f}   brightness: {s.lighting['brightness']:.2f}"]
          rec.write(compose_frame(rgb, depth, mask, mm.render(pose), info, alert=recover > 0))
          if mode == "goal" and dist < 1.0:
              reached = True
              break
          if recover > 0:                                   # back up and turn away after a collision
              cmd, recover = (-0.5, 0.8), recover - 1
          else:
              cmd = script(i) if mode == "script" else goal_command(pose, s.goal)
          t0 = time.perf_counter()
          s.step(*cmd)
          t_step.append(time.perf_counter() - t0)
          if s.check_collision():
              n_coll += 1
              recover = max(recover, 12)
    finally:
        path = rec.close()          # always finalise the file, even on Ctrl-C / error
    report(i + 1)
    print(f"video: {path}  ({i + 1} frames, {(i + 1) * s.control_dt:.0f} s of simulated time)")
    if mode == "goal":
        print("goal reached" if reached else "goal NOT reached (open-loop, obstacle-unaware demo)")
    print(f"steps with collision: {n_coll}")
    print(f"timing: step() {1e3 * np.mean(t_step):.1f} ms, get_camera() {1e3 * np.mean(t_cam):.1f} ms "
          f"-> sim-only loop ~{1.0 / (np.mean(t_step) + np.mean(t_cam)):.1f} Hz")
    if gui:
        time.sleep(2)
    s.close()


# ------------------------------------------------------------------ 4. drive calibration
def calibrate_drive(seed: int, difficulty: str) -> None:
    print("\n== 4. Drive calibration (info only) ==")
    s = S.Sim(seed, difficulty=difficulty)          # start area is obstacle-free and flat
    x0, y0, _ = s.get_true_pose()
    for _ in range(15):
        s.step(0.5, 0.0)
    x1, y1, yaw1 = s.get_true_pose()
    v_meas = math.hypot(x1 - x0, y1 - y0) / (15 * s.control_dt)
    print(f"straight: commanded v=0.50 m/s, average achieved {v_meas:.2f} m/s (includes acceleration)")
    for _ in range(20):
        s.step(0.0, 0.6)
    yaw2 = s.get_true_pose()[2]
    dyaw = (yaw2 - yaw1 + math.pi) % (2 * math.pi) - math.pi
    w_meas = dyaw / (20 * s.control_dt)
    ratio = w_meas / 0.6
    print(f"turn in place: commanded w=0.60 rad/s, achieved {w_meas:.2f} rad/s (ratio {ratio:.2f})")
    if not 0.7 <= ratio <= 1.3:
        print(f"  -> tune sim.SKID_FACTOR: try {S.SKID_FACTOR / max(ratio, 1e-3):.2f} (currently {S.SKID_FACTOR})")
    s.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--difficulty", default="medium", choices=list(S.DIFFICULTY))
    ap.add_argument("--gui", action="store_true")
    ap.add_argument("--out", default="debug")
    ap.add_argument("--mode", default="script", choices=["script", "goal"],
                    help="script: 300-step pattern; goal: drive start->goal with a crude go-to-goal controller")
    ap.add_argument("--steps", type=int, default=None, help="max steps (default 300 script / 700 goal)")
    ap.add_argument("--save-frames", action="store_true", help="also save PNGs every 30 steps")
    args = ap.parse_args()
    np.random.seed(args.seed)
    ok = check_reproducibility(args.seed, args.difficulty)
    ok &= check_geometry(args.seed, args.difficulty)
    scripted_drive(args.seed, args.difficulty, args.gui, args.out, args.mode,
                   args.steps or (300 if args.mode == "script" else 700), args.save_frames)
    calibrate_drive(args.seed, args.difficulty)
    print("\nALL CHECKS PASSED" if ok else "\nSOME CHECKS FAILED - see [FAIL] lines above")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
