"""Camera & appearance model matching MuJoCo renders to the real 640x480 USB cameras.

Per-frame ISP pipeline (parameters per camera in config/sim_camera.json):

  1. lens      supersampled wide pinhole render (1280x960) -> radial fisheye remap to 640x480
               matching each camera's real field of view, distortion k1 and optical centre
  2. optics    Gaussian lens / demosaicing PSF blur + optional UVC edge peaking (sharpen)
  3. vignette  radial illumination fall-off  1 - v1 * r^2 - v2 * r^4
  4. wb + lut  3x3 white-balance / colour-correction matrix + bias, followed by a smooth
               monotone 256-entry per-channel tone curve (exposure, contrast, black floor,
               and UVC 238 highlight ceiling)
  5. sensor    signal-dependent Bayer shot + read noise + blotchy demosaic chroma noise,
               quantised to uint8 RGB
  6. codec     (for video / dataset output) SVT-AV1 (libsvtav1, crf 30, preset 12, g 2,
               yuv420p), matching LeRobot's exact video encoding

Fit all parameters against a recorded dataset:

    uv run --project plugin --with mujoco python cloud/sim_camera.py fit \
        --dataset ../local/V1SchachBauer_clean --episode 0 --scene ../Schach/chess_scene.xml
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
from fractions import Fraction
from pathlib import Path

os.environ.setdefault("SVT_LOG", "1")

import av
import cv2
import numpy as np

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "config" / "sim_camera.json"
HEIGHT, WIDTH = 480, 640

DEFAULTS: dict = {
    "real_board_layout": True,
    "board": {
        "center_xy": [0.041, -0.2415],
        "surface_z": 0.056,
        "square_size": 0.0355,
    },
    "scene": {
        "pos": [-0.0432, -0.2800, 0.3080],
        "xyaxes": [0.13148, -0.85414, -0.50315, 0.95451, -0.02795, 0.29687],
        "fisheye_hfov_deg": 131.5,
        "distortion_k1": -0.015,
        "principal_point": [320.0, 240.0],
        "render_scale": 2.0,
        "blur_sigma": 0.55,
        "sharpen": 0.15,
        "vignette": [0.04, 0.03],
        "ccm": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        "ccm_bias": [0.0, 0.0, 0.0],
        "lut": None,
        "noise_sigma": 2.2,
        "noise_chroma": 0.9,
        "bayer_scale": [1.12, 0.88, 1.15],
    },
    "wrist": {
        "pos": [0.53937, -0.2065, 0.3785],
        "xyaxes": [-1.0, 0.0, 0.0, 0.0, -0.5850, 0.8110],
        "fisheye_hfov_deg": 116.0,
        "distortion_k1": 0.08,
        "principal_point": [320.0, 240.0],
        "render_scale": 2.0,
        "blur_sigma": 0.38,
        "sharpen": 0.0,
        "vignette": [0.20, 0.14],
        "ccm": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        "ccm_bias": [0.0, 0.0, 0.0],
        "lut": None,
        "noise_sigma": 2.2,
        "noise_chroma": 0.85,
        "bayer_scale": [1.18, 0.86, 1.12],
    },
    "materials": {
        "chess_dark": [0.045, 0.045, 0.050, 1.0],
        "chess_light": [0.80, 0.79, 0.76, 1.0],
        "chess_wood": [0.05, 0.05, 0.055, 1.0],
        "piece_white": [0.95, 0.93, 0.88, 1.0],
        "piece_black": [0.04, 0.04, 0.045, 1.0],
        "accent_blue": [0.10, 0.10, 0.11, 1.0],
        "black_metal": [0.09, 0.09, 0.10, 1.0],
        "dark_bracket": [0.12, 0.12, 0.13, 1.0],
        "gripper_jaw": [0.05, 0.05, 0.055, 1.0],
        "workcell_panel": [0.58, 0.41, 0.27, 1.0],
        "workcell_plate": [0.09, 0.09, 0.10, 1.0],
    },
    "geoms": {
        "floor": [0.56, 0.39, 0.25, 1.0],
        "chessboard_slab": [0.05, 0.05, 0.055, 1.0],
    },
    "headlight": {"diffuse": [0.56, 0.52, 0.46], "ambient": [0.36, 0.33, 0.29]},
}

FILES = "abcdefgh"


def load_config(path: Path | None = None) -> dict:
    path = Path(path) if path is not None else DEFAULT_CONFIG
    cfg = json.loads(json.dumps(DEFAULTS))
    if path.is_file():
        user = json.loads(path.read_text(encoding="utf-8"))
        for key, value in user.items():
            if isinstance(value, dict) and isinstance(cfg.get(key), dict):
                cfg[key].update(value)
            else:
                cfg[key] = value
    return cfg


def _make_cardboard_rgb(h: int = 256, w: int = 256) -> bytes:
    """Procedural warm corrugated cardboard texture with subtle grain and fold lines."""
    rng = np.random.default_rng(42)
    base = np.array([154.0, 108.0, 72.0], np.float32)
    low = cv2.GaussianBlur(rng.standard_normal((h, w)).astype(np.float32), (0, 0), 18.0)[..., None] * 45.0
    rib = np.sin(np.linspace(0, 90 * math.pi, w, dtype=np.float32))[None, :, None] * 1.1
    grain = cv2.GaussianBlur(rng.standard_normal((h, w, 3)).astype(np.float32), (0, 0), 1.4) * 1.6
    img = base + low + rib + grain
    # Dark crease lines like a folded shipping carton
    img[h // 2 - 1 : h // 2 + 1, :] *= 0.68
    img[:, w // 3 : w // 3 + 1] *= 0.78
    return np.clip(img, 20, 235).astype(np.uint8).tobytes()


def _align_real_chessboard(spec, board_cfg: dict) -> None:
    """Place, orient (White near arm, 180 deg rotated) and scale the chessboard & 32 pieces
    to match the physical folding magnetic chess set recorded in V1SchachBauer_clean."""
    import mujoco

    board = spec.body("chessboard")
    if board is None:
        return
    cx, cy = board_cfg["center_xy"]
    z_surf = float(board_cfg["surface_z"])
    sq = float(board_cfg["square_size"])
    half = 4.0 * sq
    frame_w = 0.014
    slab_h = max(0.006, z_surf * 0.5)

    board.pos = [cx, cy, 0.0]
    board.quat = [0.0, 0.0, 0.0, 1.0]  # 180 deg about +Z: rank 1 near arm (+y_rel), file 'a' at +x_rel

    slab = spec.geom("chessboard_slab")
    if slab is not None:
        slab.pos = [0.0, 0.0, z_surf - slab_h]
        slab.size = [half + frame_w * 0.8, half + frame_w * 0.8, slab_h]

    frame_specs = {
        "chessboard_frame_north": ([0.0, half + frame_w * 0.5, z_surf - 0.003], [half + frame_w, frame_w * 0.5, 0.005]),
        "chessboard_frame_south": ([0.0, -(half + frame_w * 0.5), z_surf - 0.003], [half + frame_w, frame_w * 0.5, 0.005]),
        "chessboard_frame_west": ([-(half + frame_w * 0.5), 0.0, z_surf - 0.003], [frame_w * 0.5, half, 0.005]),
        "chessboard_frame_east": ([half + frame_w * 0.5, 0.0, z_surf - 0.003], [frame_w * 0.5, half, 0.005]),
    }
    for gname, (gpos, gsize) in frame_specs.items():
        g = spec.geom(gname)
        if g is not None:
            g.pos = gpos
            g.size = gsize

    # Center fold crease visible on the real magnetic folding board
    board.add_geom(
        name="chessboard_fold_crease",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[0.0, 0.0, z_surf + 0.0005],
        size=[half + frame_w, 0.0008, 0.0002],
        rgba=[0.03, 0.03, 0.03, 1.0],
        contype=0,
        conaffinity=0,
    )

    tile_half = sq * 0.492
    for f_i, f_ch in enumerate(FILES):
        for r_i in range(8):
            sq_name = f"chess_square_{f_ch}{r_i + 1}"
            g = spec.geom(sq_name)
            if g is not None:
                # In board local frame (before 180 deg Z rotation): a1 at (-3.5*sq, -3.5*sq)
                g.pos = [(f_i - 3.5) * sq, (r_i - 3.5) * sq, z_surf + 0.0003]
                g.size = [tile_half, tile_half, 0.0004]

    def square_world_xy(sq_str: str) -> tuple[float, float]:
        f_idx = FILES.index(sq_str[0])
        r_idx = int(sq_str[1]) - 1
        # After 180 deg Z rotation around (cx, cy): x = cx - (f_idx - 3.5)*sq, y = cy - (r_idx - 3.5)*sq
        return cx + (3.5 - f_idx) * sq, cy + (3.5 - r_idx) * sq

    # Scale piece visuals (Staunton base + collar + finial, visual-only) and update initial positions & keyframes
    piece_Scale = sq / 0.025
    piece_bodies = []
    for body in spec.bodies:
        bname = body.name or ""
        if (bname.startswith("w_") or bname.startswith("b_")) and "_" in bname[2:]:
            sq_str = bname.rsplit("_", 1)[-1]
            if len(sq_str) == 2 and sq_str[0] in FILES and sq_str[1] in "12345678":
                px, py = square_world_xy(sq_str)
                body.pos = [px, py, z_surf]
                mat_name = "piece_white" if bname.startswith("w_") else "piece_black"
                # Upgrade collision/main cylinder slightly taller for real set + add visual Staunton base & head
                for g in body.geoms:
                    r0, h0 = float(g.size[0]), float(g.size[1])
                    r_vis = r0 * min(piece_Scale, 1.22)
                    h_new = h0 * min(piece_Scale, 1.30)
                    # Keep physics collision cylinder slim so gripper jaws clear neighbour pawns cleanly
                    g.size = [r0 * 0.95, h_new, 0.0]
                    g.pos = [0.0, 0.0, h_new]
                    body.add_geom(
                        name=f"{bname}_vis_stem",
                        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                        size=[r_vis * 0.92, h_new * 0.92, 0.0],
                        pos=[0.0, 0.0, h_new * 0.95],
                        material=mat_name,
                        contype=0,
                        conaffinity=0,
                    )
                    body.add_geom(
                        name=f"{bname}_vis_base",
                        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                        size=[r_vis * 1.42, 0.0030, 0.0],
                        pos=[0.0, 0.0, 0.0030],
                        material=mat_name,
                        contype=0,
                        conaffinity=0,
                    )
                    body.add_geom(
                        name=f"{bname}_vis_collar",
                        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                        size=[r_vis * 1.14, 0.0016, 0.0],
                        pos=[0.0, 0.0, h_new * 1.45],
                        material=mat_name,
                        contype=0,
                        conaffinity=0,
                    )
                    body.add_geom(
                        name=f"{bname}_vis_head",
                        type=mujoco.mjtGeom.mjGEOM_SPHERE,
                        size=[r_vis * (1.05 if "pawn" in bname else 1.15), 0.0, 0.0],
                        pos=[0.0, 0.0, h_new * 1.90],
                        material=mat_name,
                        contype=0,
                        conaffinity=0,
                    )
                    break
                piece_bodies.append((bname, px, py, z_surf))

    # Update keyframes if present so mj_resetDataToKeyframe also places pieces on the real board
    if piece_bodies:
        tail = []
        for _, px, py, pz in piece_bodies:
            tail.extend([px, py, pz, 1.0, 0.0, 0.0, 0.0])
        for key in spec.keys:
            if len(key.qpos) == 8 + 7 * len(piece_bodies):
                key.qpos = [*[float(key.qpos[i]) for i in range(8)], *tail]


def apply_appearance(spec, cfg: dict) -> None:
    """Apply real-lab visual environment, board layout, lighting and camera poses to MjSpec."""
    import mujoco

    for name, rgba in cfg.get("materials", {}).items():
        mat = spec.material(name)
        if mat is not None:
            mat.rgba = rgba
    # Make chessboard squares satin-glossy so the overhead lamp creates the subtle sheen seen in real_wrist
    for sq_mat_name, spec_val, shin in (("chess_dark", 0.45, 0.35), ("chess_light", 0.35, 0.30)):
        mat = spec.material(sq_mat_name)
        if mat is not None:
            mat.specular = spec_val
            mat.shininess = shin

    for name, rgba in cfg.get("geoms", {}).items():
        geom = spec.geom(name)
        if geom is not None:
            geom.rgba = rgba
            geom.material = ""

    # Hide unrealistic white workcell enclosure walls/extrusions not present inside the cardboard box
    for geom in spec.geoms:
        gname = geom.name or ""
        if gname.startswith("workcell_panel_") or gname.startswith("workcell_enclosure_") or gname.startswith("workcell_rail_left"):
            geom.rgba = [0.0, 0.0, 0.0, 0.0]

    # Add procedural cardboard texture & surrounding cardboard box walls + white arm table
    tex = spec.add_texture(
        name="cardboard_tex",
        type=mujoco.mjtTexture.mjTEXTURE_2D,
        width=256,
        height=256,
    )
    tex.data = _make_cardboard_rgb(256, 256)
    cmat = spec.add_material(name="cardboard_mat", rgba=[1.0, 1.0, 1.0, 1.0], specular=0.06, shininess=0.1)
    cmat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "cardboard_tex"
    cmat.texrepeat = [3.0, 3.0]

    is_chess = spec.body("chessboard") is not None
    if is_chess and cfg.get("real_board_layout", True):
        _align_real_chessboard(spec, cfg["board"])

    board_z = float(cfg.get("board", {}).get("surface_z", 0.056)) if (is_chess and cfg.get("real_board_layout", True)) else 0.01
    # Cardboard tray under the chessboard and extending across the floor
    spec.worldbody.add_geom(
        name="lab_cardboard_tray",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[0.04, -0.18, max(0.002, board_z - 0.048)],
        size=[0.42, 0.45, 0.004],
        material="cardboard_mat",
        contype=0,
        conaffinity=0,
    )
    # Off-white lab tabletop under the arm base (visible left of the arm in scene & bottom corners of wrist)
    spec.worldbody.add_geom(
        name="lab_white_table",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[0.02, 0.02, 0.022],
        size=[0.24, 0.145, 0.003],
        rgba=[0.93, 0.91, 0.87, 1.0],
        contype=0,
        conaffinity=0,
    )
    # Surrounding cardboard box flaps (top/+x, bottom/-x, far/-y, behind-arm/+y) matching real_scene enclosure
    flaps = [
        ("lab_box_wall_pos_x", [0.245, -0.16, 0.22], [0.004, 0.38, 0.22], [0.985, 0.0, 0.17, 0.0]),
        ("lab_box_wall_neg_x", [-0.165, -0.16, 0.20], [0.004, 0.38, 0.21], [0.985, 0.0, -0.17, 0.0]),
        ("lab_box_wall_neg_y", [0.04, -0.435, 0.22], [0.28, 0.004, 0.22], [0.99, 0.14, 0.0, 0.0]),
        ("lab_box_wall_pos_y", [0.04, 0.20, 0.24], [0.32, 0.004, 0.24], [0.99, -0.14, 0.0, 0.0]),
    ]
    for fname, fpos, fsize, fquat in flaps:
        spec.worldbody.add_geom(
            name=fname,
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=fpos,
            size=fsize,
            quat=fquat,
            material="cardboard_mat",
            contype=0,
            conaffinity=0,
        )

    # Warm indoor overhead point lamp above the board
    spec.worldbody.add_light(
        name="lab_overhead_warm",
        pos=[0.02, -0.22, 0.72],
        dir=[0.0, 0.0, -1.0],
        diffuse=[0.48, 0.45, 0.40],
        specular=[0.30, 0.29, 0.27],
        attenuation=[1.0, 0.2, 0.05],
    )

    light = cfg.get("headlight")
    if light:
        spec.visual.headlight.diffuse = light["diffuse"]
        spec.visual.headlight.ambient = light["ambient"]
        spec.visual.headlight.specular = [0.08, 0.08, 0.08]

    # Calibrated wrist camera pose on cam_mount
    w_cfg = cfg.get("wrist", {})
    w_cam = spec.camera("wrist")
    cam_mount = spec.body("cam_mount")
    if w_cam is not None and cam_mount is not None and "pos" in w_cfg and "xyaxes" in w_cfg:
        spec.delete(w_cam)
        cam_mount.add_camera(
            name="wrist",
            mode=mujoco.mjtCamLight.mjCAMLIGHT_FIXED,
            pos=w_cfg["pos"],
            xyaxes=w_cfg["xyaxes"],
            fovy=92.2,
        )


class CameraModel:
    """Turns a wide supersampled pinhole render into a real-camera 640x480 RGB frame."""

    def __init__(self, params: dict, seed: int | None = None):
        self.p = params
        self.rng = np.random.default_rng(seed)
        hfov = math.radians(float(params.get("fisheye_hfov_deg", 115.0)))
        k1 = float(params.get("distortion_k1", 0.0))
        cx, cy = params.get("principal_point", [WIDTH / 2.0, HEIGHT / 2.0])
        th_half = hfov / 2.0
        self.f_fish = (WIDTH / 2.0) / (th_half * (1.0 + k1 * th_half**2))

        scale = float(params.get("render_scale", 2.0))
        self.src_w, self.src_h = int(round(WIDTH * scale)), int(round(HEIGHT * scale))

        # Invert r_px(theta) = f_fish * theta * (1 + k1 * theta^2) on every output pixel
        yy, xx = np.mgrid[0:HEIGHT, 0:WIDTH].astype(np.float32)
        dx, dy = xx - float(cx) + 0.5, yy - float(cy) + 0.5
        r_px = np.hypot(dx, dy)
        theta = r_px / self.f_fish
        if abs(k1) > 1e-6:
            for _ in range(6):  # Newton-Raphson inversion
                f_val = self.f_fish * theta * (1.0 + k1 * theta**2) - r_px
                df_val = self.f_fish * (1.0 + 3.0 * k1 * theta**2)
                theta = np.clip(theta - f_val / np.maximum(df_val, 1e-4), 0.0, math.radians(84.0))

        max_theta = min(float(theta.max()), math.radians(82.0))
        self.f_pin = math.hypot(self.src_w / 2.0, self.src_h / 2.0) / math.tan(max_theta)
        self.src_fovy = math.degrees(2.0 * math.atan((self.src_h / 2.0) / self.f_pin))

        valid = theta < math.radians(83.5)
        r_src = self.f_pin * np.tan(np.where(valid, theta, 0.0))
        ratio = np.where(r_px > 1e-5, r_src / np.maximum(r_px, 1e-5), self.f_pin / self.f_fish)
        self.map_x = (dx * ratio + self.src_w / 2.0 - 0.5).astype(np.float32)
        self.map_y = (dy * ratio + self.src_h / 2.0 - 0.5).astype(np.float32)
        self.map_x[~valid] = -1.0
        self.map_y[~valid] = -1.0

        rn = r_px / math.hypot(WIDTH / 2.0, HEIGHT / 2.0)
        v1, v2 = params.get("vignette", [0.1, 0.1])
        self.vignette = np.clip(1.0 - v1 * rn**2 - v2 * rn**4, 0.15, 1.0)[..., None].astype(np.float32)

        self.ccm = np.asarray(params.get("ccm", np.eye(3)), np.float32)
        self.ccm_bias = np.asarray(params.get("ccm_bias", [0.0, 0.0, 0.0]), np.float32)
        lut = params.get("lut")
        self.lut = None if lut is None else np.asarray(lut, np.float32)
        self.bayer = np.asarray(params.get("bayer_scale", [1.12, 0.88, 1.14]), np.float32)

    def geometry(self, src: np.ndarray) -> np.ndarray:
        if src.shape[:2] != (self.src_h, self.src_w):
            raise ValueError(f"render must be {self.src_w}x{self.src_h}, got {src.shape[1]}x{src.shape[0]}")
        return cv2.remap(src, self.map_x, self.map_y, cv2.INTER_AREA if False else cv2.INTER_LINEAR, borderValue=(0, 0, 0))

    def optics(self, img: np.ndarray) -> np.ndarray:
        f = img.astype(np.float32)
        s = float(self.p.get("blur_sigma", 0.0))
        if s > 0:
            f = cv2.GaussianBlur(f, (0, 0), s)
        sharp = float(self.p.get("sharpen", 0.0))
        if sharp > 0:
            fine = cv2.GaussianBlur(f, (0, 0), 1.15)
            f = f + sharp * (f - fine)
        return np.clip(f, 0.0, 255.0)

    def colour(self, f: np.ndarray) -> np.ndarray:
        f = f * self.vignette
        f = f @ self.ccm.T + self.ccm_bias
        if self.lut is not None:
            idx = np.clip(f, 0.0, 255.0)
            lo = np.floor(idx).astype(np.int32)
            hi = np.minimum(lo + 1, 255)
            w = idx - lo
            out = np.empty_like(f)
            for c in range(3):
                out[..., c] = self.lut[c][lo[..., c]] * (1.0 - w[..., c]) + self.lut[c][hi[..., c]] * w[..., c]
            f = out
        return np.clip(f, 0.0, 255.0)

    def sensor(self, f: np.ndarray) -> np.ndarray:
        sigma = float(self.p.get("noise_sigma", 0.0))
        chroma = float(self.p.get("noise_chroma", 0.0))
        if sigma > 0 or chroma > 0:
            level = np.clip(f.mean(-1, keepdims=True) / 255.0, 0.0, 1.0)
            shot = sigma * np.sqrt(0.30 + 0.70 * level)
            n_luma = self.rng.standard_normal(f.shape[:2] + (1,)).astype(np.float32) * shot
            n_bayer = self.rng.standard_normal(f.shape).astype(np.float32) * (0.35 * sigma * self.bayer)
            n_chroma = self.rng.standard_normal(f.shape).astype(np.float32) * chroma
            n_chroma = cv2.GaussianBlur(n_chroma, (0, 0), 1.2) * 2.1
            f = f + n_luma + n_bayer + n_chroma
        return np.clip(np.round(f), 0, 255).astype(np.uint8)

    def __call__(self, src: np.ndarray) -> np.ndarray:
        return self.sensor(self.colour(self.optics(self.geometry(src))))


def codec_roundtrip(frames: list[np.ndarray], fps: int = 30, crf: int = 30) -> list[np.ndarray]:
    """Encode+decode with LeRobot's AV1 parameters (libsvtav1, crf 30, preset 12, g 2, yuv420p).
    Uses lp=1 so SVT-AV1 does not spawn 76 worker threads per short clip on Windows."""
    buf = io.BytesIO()
    with av.open(buf, "w", format="mp4") as out:
        stream = out.add_stream("libsvtav1", rate=fps)
        stream.width, stream.height, stream.pix_fmt = frames[0].shape[1], frames[0].shape[0], "yuv420p"
        stream.time_base = Fraction(1, fps)
        stream.options = {"crf": str(crf), "preset": "12", "g": "2", "svtav1-params": "lp=1"}
        for i, img in enumerate(frames):
            vf = av.VideoFrame.from_ndarray(img, format="rgb24")
            vf.pts = i
            for packet in stream.encode(vf):
                out.mux(packet)
        for packet in stream.encode():
            out.mux(packet)
    buf.seek(0)
    with av.open(buf) as inp:
        return [f.to_ndarray(format="rgb24") for f in inp.decode(inp.streams.video[0])]


def fit_ccm_and_lut(
    sim_frames: list[np.ndarray], real_frames: list[np.ndarray], ridge: float = 0.25
) -> tuple[list[list[float]], list[float], list[list[float]]]:
    """Fit a pure diagonal white-balance matrix (zero off-diagonal channel crosstalk so neutral
    pieces/squares never get colour fringes) + smooth monotone 256-entry tone curve per channel."""
    pct = np.concatenate([[0.2, 1.0, 3.0], np.arange(5, 96, 2.5), [97.0, 99.0, 99.8]])
    s_flat = np.stack(sim_frames).reshape(-1, 3).astype(np.float32)
    r_flat = np.stack(real_frames).reshape(-1, 3).astype(np.float32)

    # Diagonal white balance gain anchored on the 92nd percentile (bright neutral board/pieces)
    s_hi = np.maximum(np.percentile(s_flat, 92, axis=0), 10.0)
    r_hi = np.percentile(r_flat, 92, axis=0)
    wb = np.clip(r_hi / s_hi, 0.65, 1.45)
    ccm = np.diag(wb.astype(np.float64)).round(5).tolist()
    bias = [0.0, 0.0, 0.0]

    s_wb = np.clip(s_flat * wb[None, :], 0.0, 255.0)
    lut = []
    x_grid = np.arange(256, dtype=np.float32)
    for c in range(3):
        qs = np.percentile(s_wb[:, c], pct)
        qr = np.percentile(r_flat[:, c], pct)
        qs = np.maximum.accumulate(qs + np.arange(len(qs)) * 1e-3)
        q_curve = np.interp(x_grid, qs, qr)
        # Blend with a smooth affine/gamma ramp across sparse midtones so close-up shadows never posterize
        lo_r, hi_r = float(qr[1]), float(qr[-2])
        smooth_ramp = lo_r + (hi_r - lo_r) * ((x_grid / 255.0) ** 0.95)
        curve = 0.72 * q_curve + 0.28 * smooth_ramp
        curve = cv2.GaussianBlur(curve.reshape(1, -1).astype(np.float32), (0, 0), 9.0).ravel()
        curve = np.maximum.accumulate(np.clip(curve, float(qr[0]), float(qr[-1])))
        lut.append(curve.round(2).tolist())
    return ccm, bias, lut


def _replay_raw(dataset: Path, episode: int, scene: Path, cfg: dict, every: int):
    import sys
    import mujoco
    import pandas as pd

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from camera_stats import video_path
    from sim import KEYS, MujocoRobot

    meta = pd.read_parquet(dataset / "meta/episodes/chunk-000/file-000.parquet")
    ep = meta[meta["episode_index"] == episode].iloc[0]
    data = pd.read_parquet(dataset / "data/chunk-000/file-000.parquet")
    data = data[data["episode_index"] == episode].sort_values("frame_index")
    actions = np.stack(data["action"].to_numpy())
    states = np.stack(data["observation.state"].to_numpy())
    picks = list(range(0, len(actions), every))

    real = {}
    for cam in ("scene", "wrist"):
        path, start, _ = video_path(dataset, ep, cam)
        want = {start + i: i for i in picks}
        got = {}
        with av.open(str(path)) as container:
            for n, frame in enumerate(container.decode(container.streams.video[0])):
                if n in want:
                    got[want[n]] = frame.to_ndarray(format="rgb24")
                if n >= max(want):
                    break
        real[cam] = [got[i] for i in picks]

    robot = MujocoRobot(scene, camera_config=cfg, raw_render=True)
    robot.connect()
    first = robot.to_sim(list(states[0]))
    robot.data.qpos[robot.qpos] = first
    robot.data.qpos[robot.model.joint("left_jaw_slide_joint").qposadr[0]] = first[5]
    robot.data.ctrl[:] = first
    mujoco.mj_forward(robot.model, robot.data)

    raw = {"scene": [], "wrist": []}
    for i in range(len(actions)):
        if i in picks:
            for cam in raw:
                raw[cam].append(robot.render_source(cam))
        robot.send_action(dict(zip(KEYS, actions[i].tolist(), strict=True)))
    robot.disconnect()
    static_raw = {cam: [raw[cam][0]] * 8 for cam in raw}
    return raw, static_raw, real


def fit(args) -> None:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from camera_stats import measure, real_stats

    cfg = load_config(args.config)
    raw, static_raw, real = _replay_raw(args.dataset, args.episode, args.scene, cfg, args.every)
    report = {}

    for cam in ("scene", "wrist"):
        target = real_stats(args.dataset, cam, episodes=[0, 2, 4])
        params = dict(cfg[cam])
        ridge = 0.35 if cam == "scene" else 0.50

        # Stage 1: fit optical blur_sigma / ISP sharpen to match real sharpness (HF energy + Laplacian variance)
        optics_candidates = (
            [(0.0, 0.90)] if cam == "scene" else [(0.38, 0.0)]
        ) if args.fast else [
            (0.0, 1.35),
            (0.0, 0.90),
            (0.0, 0.55),
            (0.0, 0.20),
            (0.38, 0.0),
            (0.70, 0.0),
            (1.10, 0.0),
        ]
        best_blur = None
        for blur, sharp in optics_candidates:
            params.update(
                blur_sigma=blur,
                sharpen=sharp,
                ccm=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                ccm_bias=[0, 0, 0],
                lut=None,
                noise_sigma=1.8,
                noise_chroma=0.7,
            )
            probe = CameraModel(params, seed=0)
            pre_vig = [probe.optics(probe.geometry(f)) * probe.vignette for f in raw[cam]]
            ccm, bias, lut = fit_ccm_and_lut(pre_vig, real[cam], ridge=ridge)
            params.update(ccm=ccm, ccm_bias=bias, lut=lut)
            model = CameraModel(params, seed=0)
            m = measure(codec_roundtrip([model(f) for f in raw[cam]]))
            hf_err = np.abs(
                np.log((np.array(m["hf_energy_25_50_75"]) + 1e-6) / (np.array(target["hf_energy_25_50_75"]) + 1e-6))
            )
            lap_err = abs(math.log(max(m["laplacian_var"], 1.0) / max(target["laplacian_var"], 1.0)))
            b_score = float(hf_err.sum() + 0.8 * lap_err)
            if best_blur is None or b_score < best_blur[0]:
                best_blur = (b_score, blur, sharp, ccm, bias, lut)
            print(
                f"{cam} [blur={blur:.2f}, sharp={sharp:.2f}]: hf_lap_score={b_score:.3f} "
                f"(lap={m['laplacian_var']:.0f}/{target['laplacian_var']:.0f})",
                flush=True,
            )

        _, blur, sharp, ccm, bias, lut = best_blur

        # Stage 2: fit sensor noise_sigma (shot + Bayer + chroma) through AV1 crf=30 codec
        best = None
        for noise in args.noise_grid:
            params.update(
                blur_sigma=blur,
                sharpen=sharp,
                ccm=ccm,
                ccm_bias=bias,
                lut=lut,
                noise_sigma=noise,
                noise_chroma=round(noise * 0.38, 3),
            )
            model = CameraModel(params, seed=0)
            out = codec_roundtrip([model(f) for f in raw[cam]])
            st_out = codec_roundtrip([model(f) for f in static_raw[cam]])
            m = measure(out, static=st_out)
            hf_err = np.abs(
                np.log((np.array(m["hf_energy_25_50_75"]) + 1e-6) / (np.array(target["hf_energy_25_50_75"]) + 1e-6))
            )
            lap_err = abs(math.log(max(m["laplacian_var"], 1.0) / max(target["laplacian_var"], 1.0)))
            sp_err = abs(
                math.log(
                    max(float(np.mean(m["spatial_noise_rgb"])), 0.05)
                    / max(float(np.mean(target["spatial_noise_rgb"])), 0.05)
                )
            )
            tp_err = 0.0
            if "temporal_noise_rgb" in m and "temporal_noise_rgb" in target:
                tp_err = abs(
                    math.log(
                        max(float(np.mean(m["temporal_noise_rgb"])), 0.05)
                        / max(float(np.mean(target["temporal_noise_rgb"])), 0.05)
                    )
                )
            score = float(hf_err.sum() + 0.8 * lap_err + 1.2 * sp_err + 1.0 * tp_err)
            if best is None or score < best[0]:
                best = (score, blur, sharp, noise, ccm, bias, lut, m)
            print(
                f"{cam} [noise={noise:.2f}]: score={score:.3f} "
                f"(sp={np.mean(m['spatial_noise_rgb']):.2f}/{np.mean(target['spatial_noise_rgb']):.2f}, "
                f"tp={np.mean(m.get('temporal_noise_rgb',[0])):.2f}/{np.mean(target.get('temporal_noise_rgb',[0])):.2f})",
                flush=True,
            )

        score, blur, sharp, noise, ccm, bias, lut, m = best
        cfg[cam].update(
            blur_sigma=blur,
            sharpen=sharp,
            noise_sigma=noise,
            noise_chroma=round(noise * 0.38, 3),
            ccm=ccm,
            ccm_bias=bias,
            lut=lut,
        )
        report[cam] = {
            "blur_sigma": blur,
            "sharpen": sharp,
            "noise_sigma": noise,
            "score": round(score, 4),
            "sim": m,
            "real": target,
        }
        print(f"==> {cam}: SELECTED blur={blur} sharp={sharp} noise={noise} score={score:.3f}", flush=True)

    args.config.parent.mkdir(parents=True, exist_ok=True)
    args.config.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved calibrated camera config -> {args.config}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("fit")
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--scene", type=Path, required=True)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--report", type=Path, default=None)
    p.add_argument("--every", type=int, default=14, help="use every n-th frame of the episode")
    p.add_argument("--blur-grid", type=float, nargs="+", default=[0.25, 0.5, 0.8, 1.1, 1.45, 1.85])
    p.add_argument("--noise-grid", type=float, nargs="+", default=[0.6, 1.2, 1.8, 2.5])
    p.add_argument("--fast", action="store_true", help="focused search around pre-calibrated optics")
    args = parser.parse_args()
    if args.cmd == "fit":
        fit(args)


if __name__ == "__main__":
    main()
