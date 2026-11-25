"""Image statistics used to match the simulated cameras to the real ones.

    python cloud/camera_stats.py --dataset ../local/V1SchachBauer_clean

Prints, per camera: channel percentiles, neutral (white / black) reference colours,
temporal and spatial noise, sharpness (high-frequency energy & Laplacian variance) and
the radial vignetting profile. `measure(frames)` is reused by cloud/sim_camera.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import av
import cv2
import numpy as np
import pandas as pd


def decode(path: Path, start: int, count: int, step: int = 1) -> list[np.ndarray]:
    frames = []
    with av.open(str(path)) as container:
        for number, frame in enumerate(container.decode(container.streams.video[0])):
            if number >= start and (number - start) % step == 0:
                frames.append(frame.to_ndarray(format="rgb24"))
            if len(frames) >= count:
                break
    return frames


def flat_mask(gray: np.ndarray, thresh: float = 4.0) -> np.ndarray:
    """Pixels without edges nearby, where variation is sensor noise rather than texture."""
    grad = cv2.magnitude(cv2.Sobel(gray, cv2.CV_32F, 1, 0, 5), cv2.Sobel(gray, cv2.CV_32F, 0, 1, 5)) / 16
    grad = cv2.dilate(grad, np.ones((7, 7), np.uint8))
    return (grad < thresh) & (gray > 20) & (gray < 235)


def measure(frames: list[np.ndarray], static: list[np.ndarray] | None = None) -> dict:
    stack = np.stack(frames).astype(np.float32)
    flat = stack.reshape(-1, 3)
    lum = flat @ np.array([0.299, 0.587, 0.114], np.float32)
    hsv = np.stack([cv2.cvtColor(f, cv2.COLOR_RGB2HSV) for f in frames]).reshape(-1, 3).astype(np.float32)
    sat = hsv[:, 1]
    bright = (lum >= np.percentile(lum, 90)) & (sat < 60)
    dark = lum <= np.percentile(lum, 5)
    out = {
        "resolution": [int(frames[0].shape[1]), int(frames[0].shape[0])],
        "percentiles_rgb": {
            str(p): np.percentile(flat, p, axis=0).round(1).tolist() for p in (1, 5, 25, 50, 75, 95, 99)
        },
        "mean_rgb": flat.mean(0).round(1).tolist(),
        "white_ref_rgb": flat[bright].mean(0).round(1).tolist() if bright.any() else None,
        "black_ref_rgb": flat[dark].mean(0).round(1).tolist() if dark.any() else None,
        "saturation_mean": float(sat.mean().round(1)),
    }
    res = []
    hf = []
    for f in frames:
        g = cv2.cvtColor(f, cv2.COLOR_RGB2GRAY).astype(np.float32)
        m = flat_mask(g)
        f32 = f.astype(np.float32)
        r = f32 - cv2.GaussianBlur(f32, (0, 0), 1.5)
        if m.sum() > 500:
            res.append(r[m].std(0))
        spec = np.abs(np.fft.fftshift(np.fft.fft2(g - g.mean()))) ** 2
        h, w = g.shape
        yy, xx = np.mgrid[-h // 2 : h - h // 2, -w // 2 : w - w // 2]
        rad = np.sqrt((yy / (h / 2)) ** 2 + (xx / (w / 2)) ** 2)
        hf.append([spec[rad > t].sum() / spec.sum() for t in (0.25, 0.5, 0.75)])
    out["spatial_noise_rgb"] = np.mean(res, 0).round(2).tolist() if res else [0.0, 0.0, 0.0]
    out["hf_energy_25_50_75"] = np.mean(hf, 0).round(5).tolist()
    out["laplacian_var"] = float(
        np.mean([cv2.Laplacian(cv2.cvtColor(f, cv2.COLOR_RGB2GRAY), cv2.CV_32F).var() for f in frames]).round(1)
    )
    h, w = frames[0].shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    rad = np.sqrt(((yy - h / 2) / (h / 2)) ** 2 + ((xx - w / 2) / (w / 2)) ** 2) / np.sqrt(2)
    lum_img = (stack @ np.array([0.299, 0.587, 0.114], np.float32)).mean(0)
    rings = [float(lum_img[(rad >= a) & (rad < a + 0.1)].mean()) for a in np.arange(0, 1.0, 0.1)]
    out["radial_lum"] = (np.array(rings) / max(rings[0], 1e-3)).round(3).tolist()
    if static is not None and len(static) > 2:
        s = np.stack(static).astype(np.float32)
        g = cv2.cvtColor(static[0], cv2.COLOR_RGB2GRAY)
        m = flat_mask(g.astype(np.float32))
        if m.sum() > 200:
            out["temporal_noise_rgb"] = s.std(0)[m].mean(0).round(2).tolist()
    return out


def video_path(root: Path, episode: pd.Series, camera: str) -> tuple[Path, int, int]:
    key = f"videos/observation.images.{camera}"
    path = (
        root / key / f"chunk-{int(episode[f'{key}/chunk_index']):03d}"
        / f"file-{int(episode[f'{key}/file_index']):03d}.mp4"
    )
    start = round(float(episode[f"{key}/from_timestamp"]) * 30)
    end = round(float(episode[f"{key}/to_timestamp"]) * 30)
    return path, start, end


def real_stats(root: Path, camera: str, episodes: list[int] | None = None) -> dict:
    meta = pd.read_parquet(root / "meta/episodes/chunk-000/file-000.parquet")
    wanted: dict[Path, dict[int, tuple[str, int]]] = {}
    for _, ep in meta.iterrows():
        if episodes is not None and int(ep["episode_index"]) not in episodes:
            continue
        path, start, end = video_path(root, ep, camera)
        e = int(ep["episode_index"])
        step = max(1, (end - start) // 6)
        for k in range(6):
            wanted.setdefault(path, {})[start + k * step] = ("sample", e)
        for k in range(8):
            cur = wanted.setdefault(path, {}).get(start + k)
            wanted[path][start + k] = ("both", e) if cur and cur[0] == "sample" else ("static", e)
    frames, statics = [], {}
    for path, index in wanted.items():
        last = max(index)
        with av.open(str(path)) as container:
            for number, frame in enumerate(container.decode(container.streams.video[0])):
                if number in index:
                    kind, e = index[number]
                    img = frame.to_ndarray(format="rgb24")
                    if kind in ("sample", "both"):
                        frames.append(img)
                    if kind in ("static", "both"):
                        statics.setdefault(e, []).append(img)
                if number >= last:
                    break
    stats = measure(frames)
    temporal = []
    for s in statics.values():
        if len(s) >= 4:
            arr = np.stack(s).astype(np.float32)
            # Only use genuinely stationary episode starts (skip if arm was already moving)
            if np.abs(np.diff(arr, axis=0)).mean() < 3.5:
                m_t = measure(s[:2], static=s).get("temporal_noise_rgb")
                if m_t is not None:
                    temporal.append(m_t)
    if temporal:
        stats["temporal_noise_rgb"] = np.median(temporal, axis=0).round(2).tolist()
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    result = {cam: real_stats(args.dataset, cam) for cam in ("scene", "wrist")}
    text = json.dumps(result, indent=2)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
