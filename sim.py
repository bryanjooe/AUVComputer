"""sim.py - PyBullet world, differential-drive UGV and RGB-D camera.

Public interface (kept stable for the other modules):
    Sim(seed, gui=False, difficulty="medium")
        .reset(seed) -> None
        .step(v, w) -> None
        .get_camera() -> (rgb HxWx3 uint8, depth HxW float32 [m], mask HxW uint8)
        .get_true_pose() -> (x, y, yaw)        # evaluation / debug ONLY
        .check_collision() -> bool
        .start, .goal                          # (x, y) world coordinates

Frames (see README): World = ENU (x east/forward, y north/left, z up, yaw CCW).
Base frame = robot (x forward, y left, z up). Camera frame = OpenCV optical
(x right, y down, z forward), rigidly mounted on the base (CAM_OFFSET, CAM_PITCH_DEG).

Class ids of the mask returned by get_camera():
    0 traversable, 1 obstacle (rock / tree / moving object / sky), 2 hazard (ditch / steep)
"""
from __future__ import annotations

import hashlib
import math
import os
import tempfile
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import pybullet as p
import pybullet_data
from pybullet_utils import bullet_client

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
IMG_W, IMG_H = 320, 240
VFOV_DEG = 60.0                      # vertical FOV; fx = fy (square pixels)
NEAR, FAR = 0.1, 30.0                # clip planes [m]; sky pixels get depth == FAR
CAM_OFFSET = np.array([0.35, 0.0, 0.55])   # camera position in base frame [m]
CAM_PITCH_DEG = 15.0                 # camera tilted DOWN by this angle

WORLD_SIZE = 40.0                    # square terrain, centred on the origin [m]
CELL = 0.25                          # heightfield resolution [m]
N = int(round(WORLD_SIZE / CELL)) + 1
HALF = WORLD_SIZE / 2.0
TEX_SIZE = 2048
HAZARD_SLOPE = 0.5                   # |grad h| above this counts as "steep" (~27 deg)

PHYS_DT = 1.0 / 240.0
CTRL_DT = 0.1                        # one step(v, w) == 0.1 s of simulated time
WHEEL_RADIUS = 0.1651                # Husky wheel radius [m]
TRACK = 0.5708                       # Husky wheel-centre track width [m]
SKID_FACTOR = 1.6                    # skid-steer slip compensation; calibrate (see test_sim)
WHEEL_FORCE = 100.0
V_MAX, W_MAX = 1.0, 1.5

CLS_GROUND, CLS_OBSTACLE, CLS_HAZARD = 0, 1, 2
SKY_CLASS = CLS_OBSTACLE             # "not ground"; the planner drops it via depth == FAR

DEPTH_NOISE_BASE, DEPTH_NOISE_QUAD = 0.005, 0.0015   # sigma = a + b*z^2 [m]

# (min, max) inclusive ranges are sampled per episode.
DIFFICULTY = {
    "easy":   dict(rocks=(10, 15), trees=(6, 10),  ditches=(1, 1), mounds=(0, 0),
                   movers=(1, 1), hill_amp=0.3, fog_prob=0.0),
    "medium": dict(rocks=(20, 30), trees=(10, 16), ditches=(1, 2), mounds=(0, 1),
                   movers=(1, 2), hill_amp=0.4, fog_prob=0.3),
    "hard":   dict(rocks=(35, 50), trees=(16, 26), ditches=(2, 2), mounds=(1, 2),
                   movers=(2, 2), hill_amp=0.55, fog_prob=0.6),
}


# --------------------------------------------------------------------------- #
# Pure NumPy helpers (no PyBullet) - unit-testable
# --------------------------------------------------------------------------- #
def smoothstep(t: np.ndarray) -> np.ndarray:
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def compute_intrinsics(w: int = IMG_W, h: int = IMG_H, vfov_deg: float = VFOV_DEG) -> np.ndarray:
    """3x3 pinhole K (OpenCV pixel convention: pixel centres at integer coords)."""
    fy = (h / 2.0) / math.tan(math.radians(vfov_deg) / 2.0)
    return np.array([[fy, 0.0, (w - 1) / 2.0],
                     [0.0, fy, (h - 1) / 2.0],
                     [0.0, 0.0, 1.0]])


def pixel_rays(K: np.ndarray, w: int = IMG_W, h: int = IMG_H) -> np.ndarray:
    """(h, w, 3) camera-frame rays with z == 1, so point = ray * depth."""
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    return np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u, float)],
                    axis=-1).astype(np.float32)


def camera_rotation_base_cam(pitch_deg: float = CAM_PITCH_DEG) -> np.ndarray:
    """R_base_cam: columns are the OpenCV camera axes (right, down, forward) in base frame."""
    pr = math.radians(pitch_deg)
    fwd = np.array([math.cos(pr), 0.0, -math.sin(pr)])
    up = np.array([math.sin(pr), 0.0, math.cos(pr)])
    right = np.cross(fwd, up)                       # = (0, -1, 0) for base x-fwd, y-left
    return np.stack([right, -up, fwd], axis=1)


R_BASE_CAM = camera_rotation_base_cam()


def depth_buffer_to_meters(d: np.ndarray) -> np.ndarray:
    """PyBullet non-linear depth buffer in [0,1] -> metric z-depth along the optical axis."""
    return (FAR * NEAR / (FAR - (FAR - NEAR) * d)).astype(np.float32)


def nearest_cell(x, y) -> tuple[np.ndarray, np.ndarray]:
    ix = np.clip(np.rint((np.asarray(x) + HALF) / CELL), 0, N - 1).astype(int)
    iy = np.clip(np.rint((np.asarray(y) + HALF) / CELL), 0, N - 1).astype(int)
    return ix, iy


def sample_height(heights: np.ndarray, x, y) -> np.ndarray:
    """Bilinear terrain height at world (x, y). heights is indexed [iy, ix]."""
    gx = (np.asarray(x, float) + HALF) / CELL
    gy = (np.asarray(y, float) + HALF) / CELL
    ix = np.clip(np.floor(gx).astype(int), 0, N - 2)
    iy = np.clip(np.floor(gy).astype(int), 0, N - 2)
    fx, fy = np.clip(gx - ix, 0, 1), np.clip(gy - iy, 0, 1)
    return (heights[iy, ix] * (1 - fx) * (1 - fy) + heights[iy, ix + 1] * fx * (1 - fy)
            + heights[iy + 1, ix] * (1 - fx) * fy + heights[iy + 1, ix + 1] * fx * fy)


def pixels_to_world(rays: np.ndarray, z: np.ndarray, R_wc: np.ndarray, t_wc: np.ndarray) -> np.ndarray:
    """Back-project pixels (rays[..., 3], z[...]) to world points [..., 3]."""
    return (rays * z[..., None]) @ R_wc.T + t_wc


def apply_light_post(rgb: np.ndarray, depth: np.ndarray, light: dict) -> np.ndarray:
    """Brightness gain + optional exponential fog using metric depth (OpenCV-style post-process)."""
    img = rgb.astype(np.float32) * light["brightness"]
    k = light["fog_density"]
    if k > 0.0:
        a = np.exp(-k * depth)[..., None]
        img = img * a + np.array(light["fog_color"], np.float32) * (1.0 - a)
    return np.clip(img, 0, 255).astype(np.uint8)


def paint_sky(rgb: np.ndarray, sky: np.ndarray, light: dict) -> np.ndarray:
    """PyBullet's background is white; paint a vertical blue gradient (fog colour if foggy)."""
    if not sky.any():
        return rgb
    h = rgb.shape[0]
    t = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None, None]              # 0 = top row
    col = (np.array([110, 160, 230], np.float32) * (1 - t) + np.array([205, 222, 240], np.float32) * t)
    col = col * min(light["brightness"], 1.0)
    if light["fog_density"] > 0.0:
        col = np.broadcast_to(np.array(light["fog_color"], np.float32), (h, 1, 3))
    out = rgb.copy()
    out[sky] = np.broadcast_to(col, rgb.shape)[sky].astype(np.uint8)
    return out


def classify_pixels(seg_raw: np.ndarray, z: np.ndarray, R_wc: np.ndarray, t_wc: np.ndarray,
                    rays: np.ndarray, hazard: np.ndarray,
                    obstacle_ids: set, terrain_ids: set) -> np.ndarray:
    """PyBullet body-id mask -> {0,1,2}. Ditch/steep pixels are found by back-projecting
    terrain pixels with the noise-free depth and looking them up in the hazard grid."""
    obj = np.where(seg_raw >= 0, seg_raw & 0xFFFFFF, -1)   # strip link index; -1 = background
    mask = np.zeros(obj.shape, np.uint8)                    # default: traversable
    mask[obj < 0] = SKY_CLASS
    if obstacle_ids:
        mask[np.isin(obj, list(obstacle_ids))] = CLS_OBSTACLE
    terr = np.isin(obj, list(terrain_ids)) & (z < FAR * 0.999)
    if terr.any():
        pts = pixels_to_world(rays[terr], z[terr], R_wc, t_wc)
        ix, iy = nearest_cell(pts[:, 0], pts[:, 1])
        mask[terr] = np.where(hazard[iy, ix], CLS_HAZARD, CLS_GROUND)
    return mask


# --------------------------------------------------------------------------- #
# World generation (pure NumPy, fully determined by the rng)
# --------------------------------------------------------------------------- #
@dataclass
class WorldSpec:
    heights: np.ndarray            # (N, N) float32, indexed [iy, ix]
    hazard: np.ndarray             # (N, N) bool: ditch / steep cells (dilated)
    start: tuple
    goal: tuple
    start_yaw: float
    ditches: list
    mounds: list
    rocks: list
    trees: list
    movers: list
    lighting: dict
    texture: np.ndarray            # (TEX, TEX, 3) uint8 RGB

    def signature(self) -> str:
        """Short hash of everything random in the world (used by the reproducibility check)."""
        h = hashlib.sha256()
        for a in (self.heights, self.hazard, self.texture):
            h.update(np.ascontiguousarray(a).tobytes())
        h.update(repr((self.start, self.goal, self.start_yaw, self.ditches, self.mounds,
                       self.rocks, self.trees, self.movers, self.lighting)).encode())
        return h.hexdigest()[:16]


def _smooth_noise(rng: np.random.Generator, res: int) -> np.ndarray:
    a = rng.standard_normal((res, res)).astype(np.float32)
    return cv2.resize(a, (N, N), interpolation=cv2.INTER_CUBIC)


def make_terrain_texture(rng: np.random.Generator, size: int = TEX_SIZE) -> np.ndarray:
    """High-frequency earthy texture so ORB finds plenty of corners on the ground."""
    def noise(res: int) -> np.ndarray:
        return cv2.resize(rng.random((res, res), dtype=np.float32), (size, size),
                          interpolation=cv2.INTER_CUBIC)
    low = np.clip(0.5 * noise(8) + 0.3 * noise(32) + 0.2 * noise(128), 0, 1)
    mix = smoothstep((low - 0.35) / 0.3)[..., None]
    dirt, grass = np.array([125, 98, 68], np.float32), np.array([72, 108, 56], np.float32)
    img = dirt * (1 - mix) + grass * mix
    fine = cv2.GaussianBlur(rng.random((size, size), dtype=np.float32), (0, 0), 0.8)
    img *= (0.55 + 0.9 * (fine - fine.min()) / (np.ptp(fine) + 1e-6))[..., None]
    img = np.clip(img, 0, 255).astype(np.uint8)
    for _ in range(20000):                                    # pebbles / dark & light blobs
        c = (int(rng.integers(0, size)), int(rng.integers(0, size)))
        r = int(rng.integers(2, 9))
        shade = int(rng.integers(30, 230))
        cv2.circle(img, c, r, (shade, int(shade * rng.uniform(0.8, 1.0)),
                               int(shade * rng.uniform(0.6, 0.9))), -1)
    return img


def sample_lighting(rng: np.random.Generator, fog_prob: float) -> dict:
    az, el = rng.uniform(0, 2 * math.pi), math.radians(rng.uniform(35, 85))
    foggy = rng.random() < fog_prob
    return dict(
        direction=[float(math.cos(el) * math.cos(az)), float(math.cos(el) * math.sin(az)),
                   float(math.sin(el))],
        ambient=float(rng.uniform(0.35, 0.7)), diffuse=float(rng.uniform(0.5, 0.9)),
        brightness=float(rng.uniform(0.75, 1.25)),
        fog_density=float(rng.uniform(0.03, 0.12)) if foggy else 0.0,
        fog_color=(200.0, 205.0, 210.0))


def generate_world(rng: np.random.Generator, cfg: dict) -> WorldSpec:
    xs = -HALF + CELL * np.arange(N)
    X, Y = np.meshgrid(xs, xs)                                 # X[iy, ix], Y[iy, ix]
    f = float
    start = (f(rng.uniform(-17, -14)), f(rng.uniform(-14, 14)))
    goal = (f(rng.uniform(14, 17)), f(rng.uniform(-14, 14)))
    start_yaw = f(math.atan2(goal[1] - start[1], goal[0] - start[0]) + rng.uniform(-0.26, 0.26))

    # gentle hills, flattened around start and goal
    hills = cfg["hill_amp"] * (_smooth_noise(rng, 5) + 0.35 * _smooth_noise(rng, 9)
                               + 0.1 * _smooth_noise(rng, 17))
    H = hills.copy()
    for ex, ey in (start, goal):                       # blend towards the pad height (no steep ring)
        ix, iy = nearest_cell(ex, ey)
        pad = float(hills[iy, ix])
        H = pad + (H - pad) * smoothstep((np.hypot(X - ex, Y - ey) - 2.0) / 6.0)

    def far_from_special(c, min_end, others, min_other):
        return (math.hypot(c[0] - start[0], c[1] - start[1]) > min_end
                and math.hypot(c[0] - goal[0], c[1] - goal[1]) > min_end
                and all(math.hypot(c[0] - o["cx"], c[1] - o["cy"]) > min_other for o in others))

    hazard_an = np.zeros((N, N), bool)
    ditches: list = []
    for _ in range(int(rng.integers(cfg["ditches"][0], cfg["ditches"][1] + 1))):
        for _try in range(100):
            c = rng.uniform(-13, 13, 2)
            if far_from_special(c, 7.0, ditches, 9.0):
                break
        else:
            continue
        d = dict(cx=f(c[0]), cy=f(c[1]), theta=f(rng.uniform(0, math.pi)),
                 half_len=f(rng.uniform(3.0, 4.5)), half_w=f(rng.uniform(0.8, 1.3)),
                 depth=f(rng.uniform(0.5, 0.9)))
        edge = 0.5
        u = (X - d["cx"]) * math.cos(d["theta"]) + (Y - d["cy"]) * math.sin(d["theta"])
        v = -(X - d["cx"]) * math.sin(d["theta"]) + (Y - d["cy"]) * math.cos(d["theta"])
        prof = smoothstep((d["half_w"] + edge - np.abs(v)) / edge) * \
            smoothstep((d["half_len"] - np.abs(u)) / edge)
        H = H - d["depth"] * prof
        hazard_an |= (np.abs(u) < d["half_len"] + 0.3) & (np.abs(v) < d["half_w"] + edge + 0.3)
        ditches.append(d)

    mounds: list = []
    for _ in range(int(rng.integers(cfg["mounds"][0], cfg["mounds"][1] + 1))):
        for _try in range(100):
            c = rng.uniform(-13, 13, 2)
            if far_from_special(c, 7.0, ditches + mounds, 6.0):
                break
        else:
            continue
        m = dict(cx=f(c[0]), cy=f(c[1]), sigma=f(rng.uniform(1.0, 1.5)), amp=f(rng.uniform(1.2, 2.0)))
        r2 = (X - m["cx"]) ** 2 + (Y - m["cy"]) ** 2
        H = H + m["amp"] * np.exp(-r2 / (2 * m["sigma"] ** 2))
        hazard_an |= r2 < (1.8 * m["sigma"]) ** 2
        mounds.append(m)

    H = H.astype(np.float32)
    gy, gx = np.gradient(H, CELL)
    kern = np.ones((3, 3), np.uint8)
    haz = cv2.dilate((hazard_an | (np.hypot(gx, gy) > HAZARD_SLOPE)).astype(np.uint8), kern).astype(bool)
    haz_margin = cv2.dilate(haz.astype(np.uint8), kern, iterations=4).astype(bool)   # ~1.25 m margin

    def free(x: float, y: float) -> bool:
        ix, iy = nearest_cell(x, y)
        return (math.hypot(x - start[0], y - start[1]) > 4.0
                and math.hypot(x - goal[0], y - goal[1]) > 4.0 and not haz_margin[iy, ix])

    def place() -> Optional[tuple]:
        for _try in range(60):
            x, y = f(rng.uniform(-HALF + 1.5, HALF - 1.5)), f(rng.uniform(-HALF + 1.5, HALF - 1.5))
            if free(x, y):
                return x, y
        return None

    rocks: list = []
    for _ in range(int(rng.integers(cfg["rocks"][0], cfg["rocks"][1] + 1))):
        xy = place()
        if xy is None:
            continue
        g = f(rng.uniform(0.3, 0.6))
        if rng.random() < 0.5:
            size = tuple(f(s) for s in rng.uniform(0.15, 0.5, 3))                # box half extents
            rocks.append(dict(kind="box", x=xy[0], y=xy[1], size=size, yaw=f(rng.uniform(0, math.pi)),
                              color=(g, g, g * 0.95, 1.0)))
        else:
            rocks.append(dict(kind="sphere", x=xy[0], y=xy[1], size=(f(rng.uniform(0.2, 0.5)),),
                              yaw=0.0, color=(g, g * 0.95, g * 0.9, 1.0)))

    trees: list = []
    for _ in range(int(rng.integers(cfg["trees"][0], cfg["trees"][1] + 1))):
        xy = place()
        if xy is None:
            continue
        b = f(rng.uniform(0.6, 1.0))
        trees.append(dict(x=xy[0], y=xy[1], radius=f(rng.uniform(0.12, 0.3)),
                          height=f(rng.uniform(2.0, 4.0)), color=(0.35 * b, 0.22 * b, 0.1 * b, 1.0)))

    # moving obstacles oscillate on a segment perpendicular to the start->goal line
    axis = np.array(goal) - np.array(start)
    axis /= np.linalg.norm(axis)
    normal = np.array([-axis[1], axis[0]])
    movers: list = []
    for _ in range(int(rng.integers(cfg["movers"][0], cfg["movers"][1] + 1))):
        centre = np.array(start) + axis * rng.uniform(0.3, 0.7) * np.linalg.norm(np.array(goal) - start)
        centre = centre + rng.uniform(-2.0, 2.0) * normal
        amp = f(rng.uniform(4.0, 6.0))
        movers.append(dict(cx=f(centre[0]), cy=f(centre[1]), nx=f(normal[0]), ny=f(normal[1]),
                           amp=amp, omega=f(rng.uniform(0.4, 0.8) / amp), phase=f(rng.uniform(0, 2 * math.pi)),
                           radius=0.35, color=(0.95, 0.5, 0.1, 1.0)))

    return WorldSpec(H, haz, start, goal, start_yaw, ditches, mounds, rocks, trees, movers,
                     sample_lighting(rng, cfg["fog_prob"]), make_terrain_texture(rng))


# --------------------------------------------------------------------------- #
# Simulator
# --------------------------------------------------------------------------- #
class Sim:
    """PyBullet UGV world. One Sim == one physics client, so several can coexist."""

    def __init__(self, seed: int = 0, gui: bool = False, difficulty: str = "medium",
                 *, control_dt: float = CTRL_DT):
        if difficulty not in DIFFICULTY:
            raise ValueError(f"difficulty must be one of {list(DIFFICULTY)}")
        self.difficulty, self.cfg, self.gui = difficulty, DIFFICULTY[difficulty], gui
        self.control_dt = control_dt
        self.n_substeps = max(1, round(control_dt / PHYS_DT))
        self.K = compute_intrinsics()
        self._rays = pixel_rays(self.K)
        self.T_base_cam = np.eye(4)
        self.T_base_cam[:3, :3], self.T_base_cam[:3, 3] = R_BASE_CAM, CAM_OFFSET
        self.bc = bullet_client.BulletClient(p.GUI if gui else p.DIRECT)
        if gui:
            self.bc.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        self.bc.setAdditionalSearchPath(pybullet_data.getDataPath())
        self._proj = self.bc.computeProjectionMatrixFOV(VFOV_DEG, IMG_W / IMG_H, NEAR, FAR)
        self.reset(seed)

    # ------------------------------------------------------------------ world
    def reset(self, seed: int) -> None:
        self.seed = int(seed)
        world_ss, noise_ss = np.random.SeedSequence(self.seed).spawn(2)
        rng, self.noise_rng = np.random.default_rng(world_ss), np.random.default_rng(noise_ss)
        self.spec = generate_world(rng, self.cfg)
        self.start, self.goal = self.spec.start, self.spec.goal
        self.lighting = self.spec.lighting
        self.sim_time, self._collided, self.last_raw_seg = 0.0, False, None

        bc = self.bc
        bc.resetSimulation()
        bc.setAdditionalSearchPath(pybullet_data.getDataPath())
        bc.setGravity(0, 0, -9.81)
        bc.setTimeStep(PHYS_DT)
        self._build_terrain()
        self._spawn_objects()
        self._spawn_robot()
        for _ in range(60):                                     # let the robot settle on the ground
            bc.stepSimulation()
        if self.gui:
            bc.resetDebugVisualizerCamera(14.0, -60, -50, [*self.start, 0.0])

    def _probe_terrain(self, body: int) -> float:
        """Ray-cast a 7x7 grid onto the terrain body; return the rms error vs the height grid."""
        g = np.linspace(-15, 15, 7)
        px, py = [q.ravel() for q in np.meshgrid(g, g)]
        top, bot = float(self.spec.heights.max()) + 5, float(self.spec.heights.min()) - 5
        res = self.bc.rayTestBatch([[x, y, top] for x, y in zip(px, py)],
                                   [[x, y, bot] for x, y in zip(px, py)])
        hit = np.array([r[0] == body for r in res])
        if hit.sum() < 10:
            return float("inf")
        z = np.array([r[3][2] for r in res])[hit]
        return float(np.sqrt(np.mean((z - sample_height(self.spec.heights, px[hit], py[hit])) ** 2)))

    def _build_terrain(self) -> None:
        """One static body: textured visual mesh + concave-trimesh collision built from the SAME
        vertices (a GEOM_HEIGHTFIELD collision shape also gets an unwanted default white visual)."""
        bc, spec = self.bc, self.spec
        fd, path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        cv2.imwrite(path, cv2.cvtColor(spec.texture, cv2.COLOR_RGB2BGR))
        tex = bc.loadTexture(path)
        os.remove(path)
        xs = -HALF + CELL * np.arange(N)
        X, Y = np.meshgrid(xs, xs)
        verts = np.stack([X, Y, spec.heights], -1).reshape(-1, 3)
        gy, gx = np.gradient(spec.heights, CELL)
        nrm = np.stack([-gx, -gy, np.ones_like(gx)], -1)
        nrm = (nrm / np.linalg.norm(nrm, axis=-1, keepdims=True)).reshape(-1, 3)
        uvs = np.stack([(X - X.min()) / WORLD_SIZE, (Y - Y.min()) / WORLD_SIZE], -1).reshape(-1, 2)
        idx = np.arange(N * N).reshape(N, N)
        a, b, c, d = idx[:-1, :-1], idx[:-1, 1:], idx[1:, :-1], idx[1:, 1:]
        tris = np.stack([np.stack([a, b, d], -1), np.stack([a, d, c], -1)], axis=2).reshape(-1)
        col = bc.createCollisionShape(p.GEOM_MESH, vertices=verts.tolist(), indices=tris.tolist(),
                                      flags=p.GEOM_FORCE_CONCAVE_TRIMESH)
        vis = bc.createVisualShape(p.GEOM_MESH, vertices=verts.tolist(), indices=tris.tolist(),
                                   normals=nrm.tolist(), uvs=uvs.tolist(),
                                   rgbaColor=[1, 1, 1, 1], specularColor=[0, 0, 0])
        body = bc.createMultiBody(0, col, vis, basePosition=[0, 0, 0])
        bc.changeVisualShape(body, -1, textureUniqueId=tex, rgbaColor=[1, 1, 1, 1])
        self.terrain_ids = {body}
        rms = self._probe_terrain(body)
        if not rms < 0.05:
            raise RuntimeError(f"terrain collision does not match height grid (rms={rms:.3f} m)")

    def _spawn_objects(self) -> None:
        bc, H = self.bc, self.spec.heights
        self.obstacle_ids: set = set()
        for r in self.spec.rocks:
            z0 = float(sample_height(H, r["x"], r["y"]))
            if r["kind"] == "box":
                he = list(r["size"])
                col = bc.createCollisionShape(p.GEOM_BOX, halfExtents=he)
                vis = bc.createVisualShape(p.GEOM_BOX, halfExtents=he, rgbaColor=list(r["color"]))
                z = z0 + 0.7 * he[2]
            else:
                rad = r["size"][0]
                col = bc.createCollisionShape(p.GEOM_SPHERE, radius=rad)
                vis = bc.createVisualShape(p.GEOM_SPHERE, radius=rad, rgbaColor=list(r["color"]))
                z = z0 + 0.7 * rad
            quat = bc.getQuaternionFromEuler([0, 0, r["yaw"]])
            self.obstacle_ids.add(bc.createMultiBody(0, col, vis, [r["x"], r["y"], z], quat))
        for t in self.spec.trees:
            col = bc.createCollisionShape(p.GEOM_CYLINDER, radius=t["radius"], height=t["height"])
            vis = bc.createVisualShape(p.GEOM_CYLINDER, radius=t["radius"], length=t["height"],
                                       rgbaColor=list(t["color"]))
            z = float(sample_height(H, t["x"], t["y"])) + t["height"] / 2 - 0.05
            self.obstacle_ids.add(bc.createMultiBody(0, col, vis, [t["x"], t["y"], z]))
        self.mover_ids = []
        for m in self.spec.movers:                              # kinematic: teleported each substep
            col = bc.createCollisionShape(p.GEOM_SPHERE, radius=m["radius"])
            vis = bc.createVisualShape(p.GEOM_SPHERE, radius=m["radius"], rgbaColor=list(m["color"]))
            self.mover_ids.append(bc.createMultiBody(0, col, vis, [m["cx"], m["cy"], 1.0]))
        self.obstacle_ids.update(self.mover_ids)
        self._update_movers()

    def _spawn_robot(self) -> None:
        bc = self.bc
        sx, sy = self.start
        z0 = float(sample_height(self.spec.heights, sx, sy)) + 0.3
        quat = bc.getQuaternionFromEuler([0, 0, self.spec.start_yaw])
        self.robot = bc.loadURDF("husky/husky.urdf", [sx, sy, z0], quat)
        self.left_joints, self.right_joints = [], []
        for j in range(bc.getNumJoints(self.robot)):
            name = bc.getJointInfo(self.robot, j)[1].decode()
            if "wheel" in name:
                (self.left_joints if "left" in name else self.right_joints).append(j)
        if not (self.left_joints and self.right_joints):
            raise RuntimeError("Husky wheel joints not found")
        self._set_wheels(0.0, 0.0)

    # ---------------------------------------------------------------- dynamics
    def _set_wheels(self, w_left: float, w_right: float) -> None:
        for joints, w in ((self.left_joints, w_left), (self.right_joints, w_right)):
            for j in joints:
                self.bc.setJointMotorControl2(self.robot, j, p.VELOCITY_CONTROL,
                                              targetVelocity=w, force=WHEEL_FORCE)

    def _update_movers(self) -> None:
        for bid, m in zip(self.mover_ids, self.spec.movers):
            s = m["amp"] * math.sin(m["omega"] * self.sim_time + m["phase"])
            x, y = m["cx"] + m["nx"] * s, m["cy"] + m["ny"] * s
            z = float(sample_height(self.spec.heights, x, y)) + m["radius"]
            self.bc.resetBasePositionAndOrientation(bid, [x, y, z], [0, 0, 0, 1])

    def mover_positions(self) -> list[tuple[float, float]]:
        """Current (x, y) of each moving obstacle (for visualisation / evaluation)."""
        out = []
        for m in self.spec.movers:
            d = m["amp"] * math.sin(m["omega"] * self.sim_time + m["phase"])
            out.append((m["cx"] + m["nx"] * d, m["cy"] + m["ny"] * d))
        return out

    def _touching_obstacle(self) -> bool:
        return any(c[2] in self.obstacle_ids for c in self.bc.getContactPoints(bodyA=self.robot))

    def step(self, v: float, w: float) -> None:
        """Apply forward speed v [m/s] and yaw rate w [rad/s] (CCW+) for control_dt seconds."""
        v, w = float(np.clip(v, -V_MAX, V_MAX)), float(np.clip(w, -W_MAX, W_MAX))
        half = 0.5 * TRACK * SKID_FACTOR
        self._set_wheels((v - w * half) / WHEEL_RADIUS, (v + w * half) / WHEEL_RADIUS)
        self._collided = False
        for k in range(self.n_substeps):
            self.sim_time += PHYS_DT
            self._update_movers()
            self.bc.stepSimulation()
            if (k + 1) % 6 == 0 or k == self.n_substeps - 1:    # latch contacts made mid-step
                self._collided |= self._touching_obstacle()

    # ------------------------------------------------------------------ state
    def _base_rt(self) -> tuple[np.ndarray, np.ndarray]:
        pos, orn = self.bc.getBasePositionAndOrientation(self.robot)
        return np.array(self.bc.getMatrixFromQuaternion(orn)).reshape(3, 3), np.array(pos)

    def get_true_pose(self) -> tuple[float, float, float]:
        pos, orn = self.bc.getBasePositionAndOrientation(self.robot)
        return float(pos[0]), float(pos[1]), float(self.bc.getEulerFromQuaternion(orn)[2])

    def get_camera_pose(self) -> np.ndarray:
        """4x4 T_world_cam (OpenCV optical frame) from the current true robot pose."""
        R_wb, t_wb = self._base_rt()
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = R_wb @ R_BASE_CAM, t_wb + R_wb @ CAM_OFFSET
        return T

    def check_collision(self) -> bool:
        """True if the robot touched a rock / tree / moving obstacle during the last step()."""
        return bool(self._collided)

    def terrain_height(self, x, y):
        return sample_height(self.spec.heights, x, y)

    def in_hazard(self) -> bool:
        x, y, _ = self.get_true_pose()
        ix, iy = nearest_cell(x, y)
        return bool(self.spec.hazard[iy, ix])

    def pixels_to_world(self, depth: np.ndarray) -> np.ndarray:
        """(H, W, 3) world coordinates of every pixel of a depth image (uses true camera pose)."""
        T = self.get_camera_pose()
        return pixels_to_world(self._rays, depth, T[:3, :3], T[:3, 3])

    # ----------------------------------------------------------------- camera
    def get_camera(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        T = self.get_camera_pose()
        R_wc, t = T[:3, :3], T[:3, 3]
        view = self.bc.computeViewMatrix(t.tolist(), (t + R_wc[:, 2]).tolist(), (-R_wc[:, 1]).tolist())
        L = self.lighting
        _, _, rgba, dbuf, seg = self.bc.getCameraImage(
            IMG_W, IMG_H, viewMatrix=view, projectionMatrix=self._proj,
            lightDirection=L["direction"], lightColor=[1, 1, 1], shadow=0,
            lightAmbientCoeff=L["ambient"], lightDiffuseCoeff=L["diffuse"], lightSpecularCoeff=0.1,
            renderer=p.ER_TINY_RENDERER, flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX)
        rgb = np.ascontiguousarray(np.asarray(rgba, np.uint8).reshape(IMG_H, IMG_W, 4)[..., :3])
        z = depth_buffer_to_meters(np.asarray(dbuf, np.float32).reshape(IMG_H, IMG_W))
        self.last_raw_seg = np.asarray(seg, np.int32).reshape(IMG_H, IMG_W)
        mask = classify_pixels(self.last_raw_seg, z, R_wc, t, self._rays, self.spec.hazard,
                               self.obstacle_ids, self.terrain_ids)
        rgb = apply_light_post(rgb, z, L)                       # fog uses the noise-free depth
        rgb = paint_sky(rgb, self.last_raw_seg < 0, L)
        valid = z < FAR * 0.999
        sigma = DEPTH_NOISE_BASE + DEPTH_NOISE_QUAD * z ** 2
        noisy = z + self.noise_rng.standard_normal(z.shape).astype(np.float32) * sigma
        depth = np.where(valid, np.clip(noisy, NEAR, FAR), FAR).astype(np.float32)
        return rgb, depth, mask

    # ---------------------------------------------------------------- housekeeping
    def world_signature(self) -> str:
        return self.spec.signature()

    def close(self) -> None:
        try:
            self.bc.disconnect()
        except Exception:
            pass

    def __del__(self) -> None:
        self.close()
