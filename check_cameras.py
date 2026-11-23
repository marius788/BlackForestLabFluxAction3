"""Live camera frame-rate check for the NexArm setup.

Opens the cameras the same way LeRobot does on Windows (DirectShow, 640x480, 30 fps requested),
measures the frame rate each camera ACTUALLY delivers and shows a live preview with fps and
brightness overlaid. Use it to find out whether more light (or a driver setting) brings the
cameras to 30 fps.

Usage (inside open-nexarm-shell.cmd, which starts in the finsh folder):
    python ..\\check_cameras.py                      # cameras 0 and 2, live preview
    python ..\\check_cameras.py 0 2 --seconds 10 --no-preview

Preview window:
    q / Esc            quit
    1, 2, ... / button open the driver settings dialog of the 1st, 2nd, ... camera
                       (look for "Low Light Compensation" / exposure priority and switch it off)

Rule of thumb: green = >= 28 fps (fine for 30 Hz recording), red = the camera drops frames.
Many USB cameras lower their frame rate automatically when the scene is too dark.

The preview uses tkinter because the nexarm env ships opencv-python-headless (no cv2.imshow).
"""

from __future__ import annotations

import argparse
import collections
import threading
import time

import cv2
import numpy as np

TARGET_FPS = 30
OK_FPS = 28
PREVIEW_SIZE = (480, 360)


class CameraReader(threading.Thread):
    """Reads frames from one camera as fast as it delivers them and records the timestamps."""

    def __init__(self, index: int, width: int, height: int, fps: int, fourcc: str | None = None):
        super().__init__(daemon=True)
        self.index = index
        self.cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            raise RuntimeError(f"Camera {index} could not be opened (wrong index or used by another program?)")
        if fourcc:
            # Same order as LeRobot's OpenCVCamera: pixel format first, then size and frame rate.
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.lock = threading.Lock()
        self.frame: np.ndarray | None = None
        self.stamps: collections.deque[float] = collections.deque(maxlen=2 * TARGET_FPS)
        self.total_frames = 0
        self.first_frame_time: float | None = None
        self.running = True

    def run(self) -> None:
        while self.running:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            now = time.perf_counter()
            with self.lock:
                if self.first_frame_time is None:
                    self.first_frame_time = now
                self.frame = frame
                self.stamps.append(now)
                self.total_frames += 1

    def current_fps(self) -> float:
        """Frame rate over the last ~2 seconds."""
        with self.lock:
            if len(self.stamps) < 2:
                return 0.0
            span = self.stamps[-1] - self.stamps[0]
            return (len(self.stamps) - 1) / span if span > 0 else 0.0

    def average_fps(self) -> float:
        """Frame rate from the first delivered frame until now (startup time excluded)."""
        with self.lock:
            if self.first_frame_time is None or self.total_frames < 2:
                return 0.0
            elapsed = self.stamps[-1] - self.first_frame_time
            return (self.total_frames - 1) / elapsed if elapsed > 0 else 0.0

    def latest(self) -> np.ndarray | None:
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def info(self) -> str:
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        code = int(self.cap.get(cv2.CAP_PROP_FOURCC))
        fourcc = "".join(chr((code >> 8 * i) & 0xFF) for i in range(4))
        return f"{w}x{h} {fourcc}, exposure={self.cap.get(cv2.CAP_PROP_EXPOSURE):.0f}"

    def open_settings(self) -> None:
        # DirectShow opens the driver's property dialog (exposure, low-light compensation, ...).
        self.cap.set(cv2.CAP_PROP_SETTINGS, 1)

    def stop(self) -> None:
        self.running = False
        self.join(timeout=2)
        self.cap.release()


def brightness(frame: np.ndarray) -> float:
    return float(cv2.cvtColor(frame[::4, ::4], cv2.COLOR_BGR2GRAY).mean())


def annotate(frame: np.ndarray, reader: CameraReader) -> np.ndarray:
    """Resized RGB preview with fps (green/red) and brightness drawn on top."""
    fps = reader.current_fps()
    color = (0, 200, 0) if fps >= OK_FPS else (255, 0, 0)  # RGB
    view = cv2.cvtColor(cv2.resize(frame, PREVIEW_SIZE), cv2.COLOR_BGR2RGB)
    cv2.rectangle(view, (0, 0), (PREVIEW_SIZE[0], 58), (0, 0, 0), -1)
    cv2.putText(view, f"cam {reader.index}: {fps:5.1f} fps", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(
        view, f"brightness {brightness(frame):5.1f}/255", (8, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1
    )
    return view


def print_fps_line(readers: list[CameraReader]) -> None:
    print("   ".join(f"cam {r.index}: {r.current_fps():5.1f} fps" for r in readers), flush=True)


def run_headless(readers: list[CameraReader], seconds: float | None) -> None:
    start = time.perf_counter()
    time.sleep(1.0)
    while seconds is None or time.perf_counter() - start < seconds:
        print_fps_line(readers)
        time.sleep(1.0)


def run_preview(readers: list[CameraReader], seconds: float | None) -> None:
    import tkinter as tk

    from PIL import Image, ImageTk

    root = tk.Tk()
    root.title("NexArm camera check  (q = quit, 1/2 = driver settings)")
    image_label = tk.Label(root)
    image_label.pack()
    buttons = tk.Frame(root)
    buttons.pack(fill="x")
    for i, reader in enumerate(readers):
        tk.Button(
            buttons, text=f"[{i + 1}] Settings cam {reader.index}", command=reader.open_settings
        ).pack(side="left", padx=4, pady=4)
    tk.Button(buttons, text="[q] Quit", command=root.destroy).pack(side="right", padx=4, pady=4)

    def on_key(event: tk.Event) -> None:
        if event.char == "q" or event.keysym == "Escape":
            root.destroy()
        elif event.char.isdigit() and 1 <= int(event.char) <= len(readers):
            readers[int(event.char) - 1].open_settings()

    root.bind("<Key>", on_key)
    start = time.perf_counter()
    state = {"last_print": start}

    def update() -> None:
        now = time.perf_counter()
        if seconds is not None and now - start >= seconds:
            root.destroy()
            return
        if now - state["last_print"] >= 1.0:
            state["last_print"] = now
            print_fps_line(readers)
        views = [annotate(f, r) for r in readers if (f := r.latest()) is not None]
        if views:
            photo = ImageTk.PhotoImage(Image.fromarray(np.hstack(views)))
            image_label.configure(image=photo)
            image_label.image = photo  # keep a reference, otherwise tkinter drops the image
        root.after(30, update)

    update()
    root.mainloop()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("indices", nargs="*", type=int, default=[0, 2], help="camera indices (default: 0 2)")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=TARGET_FPS)
    parser.add_argument("--seconds", type=float, default=None, help="stop automatically after N seconds")
    parser.add_argument("--no-preview", action="store_true", help="no window, print the fps once per second")
    parser.add_argument("--fourcc", default=None, help="request a pixel format, e.g. MJPG (default: camera default)")
    args = parser.parse_args()

    readers: list[CameraReader] = []
    try:
        for idx in args.indices:
            reader = CameraReader(idx, args.width, args.height, args.fps, args.fourcc)
            readers.append(reader)
            print(f"Camera {idx}: {reader.info()}")
        for reader in readers:
            reader.start()
        print("Measuring ... (q in the window or Ctrl+C to stop)")
        if args.no_preview:
            run_headless(readers, args.seconds)
        else:
            run_preview(readers, args.seconds)
    except KeyboardInterrupt:
        pass
    finally:
        print("\nAverage over the whole run:")
        for reader in readers:
            avg = reader.average_fps()
            verdict = "OK" if avg >= OK_FPS else f"too slow for {TARGET_FPS} Hz recording"
            print(f"  cam {reader.index}: {avg:5.1f} fps  ({verdict})")
            reader.stop()


if __name__ == "__main__":
    main()
