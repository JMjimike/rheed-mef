#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""测流式改曝光后第几帧才变成新曝光，以及一轮融合能跑多快。"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import numpy as np

RUNTIME = Path(__file__).resolve().parents[1] / "clc_mbe_runtime"
sys.path.insert(0, str(RUNTIME))

from src.lib.VimbaX_camera import VimbaCamera  # noqa: E402

SERIAL = "E0022060"


def squeeze(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    return np.ascontiguousarray(arr)


def level(image: np.ndarray) -> float:
    data = squeeze(image)
    return float(np.percentile(data[::4, ::4], 50))


class Stream:
    def __init__(self, camera: VimbaCamera) -> None:
        self.camera = camera
        self.latest = {"n": 0, "frame": None, "t": 0.0}
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.meta: dict = {}
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        self.camera.run_streaming_session(self._on_frame, self.stop, on_live=self._on_live)

    def _on_live(self) -> None:
        camera = self.camera
        for action in (
            lambda: camera.set_exposure_auto_live("Off"),
            lambda: camera.set_gain_auto_live("Off"),
            lambda: camera.set_gamma_live(1.0),
            lambda: camera.set_gain_live(0.0),
        ):
            try:
                action()
            except Exception as exc:
                print(f"lock failed: {exc}", flush=True)
        try:
            name = camera._probe_exposure_feature_name()
            low, high = getattr(camera._camera, name).get_range()
            self.meta["range"] = (float(low), float(high))
        except Exception as exc:
            self.meta["range"] = (26.0, 4_000_000.0)
            print(f"range failed: {exc}", flush=True)

    def _on_frame(self, image) -> None:
        frame = squeeze(image)
        with self.lock:
            self.latest["n"] += 1
            self.latest["frame"] = frame
            self.latest["t"] = time.perf_counter()

    def start(self) -> None:
        self.thread.start()
        if not self.camera.wait_until_live(8.0):
            raise RuntimeError("stream did not start")
        deadline = time.monotonic() + 8.0
        while time.monotonic() < deadline:
            with self.lock:
                ready = self.latest["frame"] is not None and "range" in self.meta
            if ready:
                return
            time.sleep(0.01)
        raise RuntimeError("no frame")

    def close(self) -> None:
        self.stop.set()
        self.thread.join(timeout=8.0)

    def grab(self, exposure_us: float, skip: int) -> tuple[float, np.ndarray, float]:
        """改曝光后丢掉 skip 帧，返回读回曝光、留下的帧、从改曝光到这帧的毫秒数。"""
        t_min, t_max = self.meta["range"]
        use = float(min(t_max, max(t_min, exposure_us)))
        with self.lock:
            start_n = self.latest["n"]
        t0 = time.perf_counter()
        self.camera.set_exposure_time_live(use)
        need = start_n + skip + 1
        deadline = time.monotonic() + max(3.0, use / 1e6 * 8 + 1.0)
        frame = None
        arrived = t0
        while time.monotonic() < deadline:
            with self.lock:
                count = self.latest["n"]
                frame = self.latest["frame"]
                arrived = self.latest["t"]
            if count >= need and frame is not None:
                break
            time.sleep(0.001)
        else:
            raise TimeoutError(f"no frame for {use:.0f} us")
        return float(self.camera.get_exposure_time_live()), frame.copy(), (arrived - t0) * 1000.0

    def trace(self, exposure_us: float, count: int) -> list[tuple[int, float, float]]:
        """改曝光后连续 count 帧的序号、中位码值、到达间隔。"""
        t_min, t_max = self.meta["range"]
        use = float(min(t_max, max(t_min, exposure_us)))
        with self.lock:
            start_n = self.latest["n"]
            last_t = self.latest["t"]
        self.camera.set_exposure_time_live(use)
        rows = []
        seen = start_n
        deadline = time.monotonic() + max(4.0, use / 1e6 * (count + 4))
        while len(rows) < count and time.monotonic() < deadline:
            with self.lock:
                number = self.latest["n"]
                frame = self.latest["frame"]
                stamp = self.latest["t"]
            if number > seen and frame is not None:
                rows.append((number - start_n, level(frame), (stamp - last_t) * 1000.0))
                last_t = stamp
                seen = number
            else:
                time.sleep(0.001)
        return rows


def main() -> None:
    camera = VimbaCamera()
    stream = None
    try:
        camera.init()
        devices = camera.get_all_devices_info()
        index = None
        for i, device in enumerate(devices):
            blob = " ".join(str(device.get(key, "")) for key in ("model_name", "camera_name", "camera_id", "serial_number"))
            print(f"camera {i}: {blob}", flush=True)
            if SERIAL in blob or "Manta_G-033B" in blob:
                index = i
        if index is None:
            raise RuntimeError("camera missing")
        camera.open_device(index)
        camera.setup_camera()
        stream = Stream(camera)
        stream.start()
        print(f"range {stream.meta['range']}", flush=True)

        print("settle 4000 us", flush=True)
        stream.grab(4000, skip=4)
        print("switch 4000 -> 1000, next frames:", flush=True)
        for index, median, gap in stream.trace(1000, 6):
            print(f"  frame+{index} median {median:.0f} gap {gap:.1f} ms", flush=True)
        print("switch 1000 -> 16000, next frames:", flush=True)
        for index, median, gap in stream.trace(16000, 6):
            print(f"  frame+{index} median {median:.0f} gap {gap:.1f} ms", flush=True)
        print("switch 16000 -> 4000, next frames:", flush=True)
        for index, median, gap in stream.trace(4000, 6):
            print(f"  frame+{index} median {median:.0f} gap {gap:.1f} ms", flush=True)

        bracket = [1000.0, 2000.0, 4000.0, 16000.0]
        for skip in (1, 2):
            t0 = time.perf_counter()
            cycles = 0
            levels = []
            while time.perf_counter() - t0 < 2.0:
                got = []
                for exposure in bracket:
                    _actual, frame, _ms = stream.grab(exposure, skip=skip)
                    got.append(level(frame))
                levels.append(got)
                cycles += 1
            elapsed = time.perf_counter() - t0
            med = np.median(np.asarray(levels), axis=0)
            print(
                f"skip {skip}: {cycles} cycles in {elapsed:.2f}s, {cycles / elapsed:.2f} HDR/s, medians {med}",
                flush=True,
            )
    finally:
        if stream is not None:
            stream.close()
        camera.close()


if __name__ == "__main__":
    main()
