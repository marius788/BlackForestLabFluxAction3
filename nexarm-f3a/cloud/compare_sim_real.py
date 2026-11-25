"""Replay a recorded episode in the MuJoCo chess scene and put real and sim frames side by side.

Writes a side-by-side video (real scene | sim scene / real wrist | sim wrist) and a contact
sheet PNG with a few moments, so the visual sim-to-real gap can be judged by eye.

    python cloud/compare_sim_real.py --dataset ../local/V1SchachBauer_clean --episode 0
"""

import argparse
import sys
from pathlib import Path

import av
import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sim import KEYS, MJCF_DIR, MujocoRobot  # noqa: E402


def episode_frames(root: Path, episode: pd.Series, camera: str, length: int) -> list[np.ndarray]:
    key = f"videos/observation.images.{camera}"
    path = (
        root / f"videos/observation.images.{camera}"
        / f"chunk-{int(episode[f'{key}/chunk_index']):03d}"
        / f"file-{int(episode[f'{key}/file_index']):03d}.mp4"
    )
    start = round(float(episode[f"{key}/from_timestamp"]) * 30)
    frames = []
    with av.open(str(path)) as container:
        for number, frame in enumerate(container.decode(container.streams.video[0])):
            if number >= start:
                frames.append(frame.to_ndarray(format="rgb24"))
            if len(frames) >= length:
                break
    return frames


def label(img: np.ndarray, text: str) -> np.ndarray:
    img = img.copy()
    cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4)
    cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
    return img


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--scene", type=Path, default=MJCF_DIR / "chess_scene.xml")
    parser.add_argument("--out", type=Path, default=Path("simulation_data/03_calibrated_comparison"))
    parser.add_argument("--moments", type=int, default=5)
    parser.add_argument("--use-state", action="store_true", help="replay observation.state instead of action")
    parser.add_argument("--raw", action="store_true", help="plain MuJoCo render, without the real-camera model")
    parser.add_argument("--av1-roundtrip", action="store_true", help="also run slow SVT-AV1 encode/decode before H.264")
    parser.add_argument("--camera-config", type=Path, default=None, help="path to sim_camera.json")
    args = parser.parse_args()

    episodes = pd.read_parquet(args.dataset / "meta/episodes/chunk-000/file-000.parquet")
    episode = episodes[episodes["episode_index"] == args.episode].iloc[0]
    data = pd.read_parquet(args.dataset / "data/chunk-000/file-000.parquet")
    data = data[data["episode_index"] == args.episode].sort_values("frame_index")
    column = "observation.state" if args.use_state else "action"
    commands = np.stack(data[column].to_numpy())
    length = len(commands)

    real_scene = episode_frames(args.dataset, episode, "scene", length)
    real_wrist = episode_frames(args.dataset, episode, "wrist", length)
    length = min(length, len(real_scene), len(real_wrist))
    print(f"episode {args.episode}: {length} frames, replaying {column}")

    robot = MujocoRobot(args.scene, video=None, fps=30, camera_config=False if args.raw else args.camera_config)
    robot.connect()
    # start the sim in the recorded first pose instead of START
    first = robot.to_sim(list(np.stack(data["observation.state"].to_numpy())[0]))
    robot.data.qpos[robot.qpos] = first
    robot.data.qpos[robot.model.joint("left_jaw_slide_joint").qposadr[0]] = first[5]
    robot.data.ctrl[:] = first
    import mujoco

    mujoco.mj_forward(robot.model, robot.data)

    args.out.mkdir(parents=True, exist_ok=True)
    video_path = args.out / f"episode_{args.episode:03d}_real_vs_sim.mp4"
    sims = {"scene": [], "wrist": []}
    for i in range(length):
        for cam in sims:
            sims[cam].append(robot.render(cam))
        robot.send_action(dict(zip(KEYS, commands[i].tolist(), strict=True)))
    robot.disconnect()
    if args.av1_roundtrip and not args.raw:
        from sim_camera import codec_roundtrip

        sims = {cam: codec_roundtrip(frames) for cam, frames in sims.items()}

    picks = set(np.linspace(0, length - 1, args.moments).astype(int).tolist())
    with av.open(str(video_path), "w") as out_container:
        stream = out_container.add_stream("libx264", rate=30)
        stream.width, stream.height, stream.pix_fmt = 1280, 960, "yuv420p"
        stream.options = {"crf": "20", "preset": "veryfast"}
        for i in range(length):
            sim_scene, sim_wrist = sims["scene"][i], sims["wrist"][i]
            grid = np.vstack([
                np.hstack([label(real_scene[i], f"REAL scene #{i}"), label(sim_scene, "SIM scene")]),
                np.hstack([label(real_wrist[i], "REAL wrist"), label(sim_wrist, "SIM wrist")]),
            ])
            for packet in stream.encode(av.VideoFrame.from_ndarray(grid, format="rgb24")):
                out_container.mux(packet)
            if i in picks:
                cv2.imwrite(str(args.out / f"episode_{args.episode:03d}_full_{i:04d}.png"), grid[..., ::-1])
        for packet in stream.encode():
            out_container.mux(packet)

    print(f"video: {video_path}")
    print(f"pngs:  {args.out}")


if __name__ == "__main__":
    main()
