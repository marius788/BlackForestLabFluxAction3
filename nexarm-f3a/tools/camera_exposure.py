"""Measure / set the exposure mode of the two cameras (DirectShow) and the frame rate they then deliver.

USB cameras lower their frame rate in auto exposure when the scene is dark ("low light compensation").
A fixed exposure time short enough for 30 fps removes that. The setting lives in the camera driver and
stays until the camera is unplugged or the PC restarts.

Usage:
    python camera_exposure.py                      measure only (current setting)
    python camera_exposure.py --exposure -6        fixed exposure 2^-6 s = 15.6 ms on both cameras
    python camera_exposure.py --auto               back to auto exposure
    python camera_exposure.py --sweep              try auto, -5, -6, -7 and print fps + brightness
Options: indices (default 0 2), --seconds per measurement (default 4), --fourcc MJPG
"""

from __future__ import annotations

import argparse
import threading
import time

import cv2


class Cam(threading.Thread):
    def __init__(self, index: int, fourcc: str | None):
        super().__init__(daemon=True)
        self.index = index
        self.cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            raise RuntimeError(f"Camera {index} could not be opened (in use by another program?)")
        if fourcc:
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.count = 0
        self.brightness = 0.0
        self.running = True

    def run(self) -> None:
        while self.running:
            ok, frame = self.cap.read()
            if ok:
                self.count += 1
                self.brightness = float(frame[::8, ::8].mean())

    def set_auto(self) -> None:
        # DirectShow backend: 0.75 = auto, 0.25 = manual (OpenCV convention); both are tried.
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)

    def set_manual(self, exposure: int) -> None:
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        self.cap.set(cv2.CAP_PROP_EXPOSURE, exposure)

    def state(self) -> str:
        return (f"auto_exposure={self.cap.get(cv2.CAP_PROP_AUTO_EXPOSURE):g} "
                f"exposure={self.cap.get(cv2.CAP_PROP_EXPOSURE):g}")


def measure(cams: list[Cam], seconds: float, label: str) -> None:
    time.sleep(1.0)  # let the new setting take effect
    start = [c.count for c in cams]
    t0 = time.perf_counter()
    time.sleep(seconds)
    elapsed = time.perf_counter() - t0
    parts = [
        f"cam {c.index}: {(c.count - s) / elapsed:5.1f} fps, brightness {c.brightness:5.1f}, {c.state()}"
        for c, s in zip(cams, start)
    ]
    print(f"{label:<14} " + " | ".join(parts), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("indices", nargs="*", type=int, default=[0, 2])
    parser.add_argument("--exposure", type=int, default=None, help="fixed exposure as power of two, e.g. -6")
    parser.add_argument("--auto", action="store_true")
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--seconds", type=float, default=4.0)
    parser.add_argument("--fourcc", default=None)
    args = parser.parse_args()

    cams = [Cam(i, args.fourcc) for i in args.indices]
    for c in cams:
        c.start()
    try:
        if args.sweep:
            measure(cams, args.seconds, "as found")
            for c in cams:
                c.set_auto()
            measure(cams, args.seconds, "auto")
            for exposure in (-5, -6, -7):
                for c in cams:
                    c.set_manual(exposure)
                measure(cams, args.seconds, f"manual {exposure}")
            return
        if args.auto:
            for c in cams:
                c.set_auto()
        elif args.exposure is not None:
            for c in cams:
                c.set_manual(args.exposure)
        measure(cams, args.seconds, "now")
    finally:
        for c in cams:
            c.running = False
        for c in cams:
            c.join(timeout=2)
            c.cap.release()


if __name__ == "__main__":
    main()
