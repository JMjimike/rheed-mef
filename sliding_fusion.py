#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""流式多曝光。每档只留最新一帧，来一帧就更新一次融合。"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np

from debevec import hat_weights
from exposure_plan import MAX_EXPOSURE_US


def squeeze(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    return np.ascontiguousarray(arr)


def frame_level(image: np.ndarray) -> float:
    """整幅图的中位码值。斑点只占很少像素，中位数跟的是背景。"""
    sample = squeeze(image)[::4, ::4]
    return float(np.median(sample))


def reference_ratio(ratios: list[float | None], slot: int) -> float | None:
    """其它档已经接受的「中位码值 / 曝光时间」。用来判断新的一帧是不是这一档。"""
    others = [
        value
        for index, value in enumerate(ratios)
        if index != slot and value is not None and value > 0
    ]
    if not others:
        return None
    return float(np.median(others))


def level_matches(
    level: float,
    exposure_us: float,
    reference: float | None,
    factor: float = 1.5,
) -> bool:
    """新帧的码值是否和曝光时间相称。差一个因子以上就当这帧还是上一档，丢掉。"""
    if level <= 0 or exposure_us <= 0:
        return False
    if reference is None or reference <= 0:
        return True
    ratio = level / float(exposure_us)
    return (reference / factor) <= ratio <= (reference * factor)


class SlidingFusion:
    """式 (6) 的增量形式。某一档换了新帧，只改这一档的分子和分母。"""

    def __init__(
        self,
        response: np.ndarray,
        n_exposures: int,
        window: tuple[float, float] | None = None,
    ) -> None:
        curve = np.asarray(response, dtype=np.float32).ravel()
        self.zmax = int(curve.size - 1)
        self.curve = curve
        self.weight = hat_weights(0, self.zmax).astype(np.float32)
        self._numer = None
        self._denom = None
        self._part_numer: list[np.ndarray | None] = [None] * n_exposures
        self._part_denom: list[np.ndarray | None] = [None] * n_exposures
        self._filled = 0
        self._shown = 0
        self._window_lock = threading.Lock()
        self._low = None
        self._high = None
        if window is not None:
            self.set_display_window(window[0], window[1])

    def push(self, index: int, frame: np.ndarray, exposure_us: float) -> np.ndarray | None:
        codes = np.clip(squeeze(frame), 0, self.zmax).astype(np.int32, copy=False)
        weight = self.weight[codes]
        part_numer = weight * (self.curve[codes] - np.float32(np.log(float(exposure_us))))
        part_denom = weight
        if self._numer is None:
            self._numer = np.zeros(codes.shape, dtype=np.float32)
            self._denom = np.zeros(codes.shape, dtype=np.float32)
        if self._part_numer[index] is None:
            self._filled += 1
        else:
            self._numer -= self._part_numer[index]
            self._denom -= self._part_denom[index]
        self._numer += part_numer
        self._denom += part_denom
        self._part_numer[index] = part_numer
        self._part_denom[index] = part_denom
        if self._filled < len(self._part_numer):
            return None
        self._shown += 1
        if self._shown % 16 == 0:
            self._numer = np.zeros_like(self._part_numer[0])
            self._denom = np.zeros_like(self._part_denom[0])
            for part_numer, part_denom in zip(self._part_numer, self._part_denom):
                self._numer += part_numer
                self._denom += part_denom
        log_e = self._numer / np.maximum(self._denom, 1e-6)
        low, high = self._display_limits(log_e)
        view = np.clip((log_e - low) / (high - low), 0.0, 1.0)
        return np.rint(view * 255.0).astype(np.uint8)

    def set_display_window(self, low: float, high: float) -> None:
        """用和辐射图相同的对数窗口。给定之后不再从视频里重估。"""
        low_value, high_value = float(low), float(high)
        if high_value <= low_value:
            high_value = low_value + 1.0
        with self._window_lock:
            self._low = low_value
            self._high = high_value

    def _display_limits(self, log_e: np.ndarray) -> tuple[float, float]:
        # 没有融合结果时，窗口只在第一张完整融合上取一次。
        # 之后灰阶跟着 log E 走，不再每几帧重拉对比度。
        with self._window_lock:
            if self._low is None or self._high is None:
                sample = log_e[::4, ::4]
                low, high = np.percentile(sample, [1.0, 99.5])
                if not np.isfinite(low) or not np.isfinite(high) or high <= low:
                    high = float(low) + 1.0
                self._low, self._high = float(low), float(high)
            return self._low, self._high


def run_sliding(
    camera,
    exposures: list[float],
    response: np.ndarray,
    on_frame,
    stop_event: threading.Event,
    seconds: float | None = None,
    on_raw=None,
    fusion: SlidingFusion | None = None,
    gain: float = 0.0,
) -> dict:
    """在已经打开的相机上循环曝光。on_frame 收到 8-bit 对数融合图。"""
    if len(exposures) < 2:
        raise ValueError("至少需要两档曝光")
    latest = {"n": 0, "frame": None}
    lock = threading.Lock()
    stop = threading.Event()

    def on_camera_frame(image) -> None:
        if image is None:
            return
        frame = squeeze(image)
        if frame.ndim < 2:
            return
        with lock:
            latest["n"] += 1
            latest["frame"] = frame

    def on_live() -> None:
        for action in (
            lambda: camera.set_exposure_auto_live("Off"),
            lambda: camera.set_gain_auto_live("Off"),
            lambda: camera.set_gamma_live(1.0),
            lambda: camera.set_gain_live(float(gain)),
            lambda: camera._camera.TriggerMode.set("Off"),
            lambda: camera.set_exposure_time_live(min(float(min(exposures)), MAX_EXPOSURE_US)),
        ):
            try:
                action()
            except Exception:
                pass

    if fusion is None:
        fusion = SlidingFusion(response, len(exposures))
    thread = threading.Thread(
        target=lambda: camera.run_streaming_session(on_camera_frame, stop, on_live=on_live),
        daemon=True,
    )
    shown = 0
    started = time.perf_counter()
    try:
        thread.start()
        if not camera.wait_until_live(8.0):
            raise RuntimeError("相机流没有启动")
        deadline = time.monotonic() + 20.0
        while time.monotonic() < deadline:
            with lock:
                ready = latest["frame"] is not None
            if ready:
                break
            time.sleep(0.01)
        else:
            raise RuntimeError("预览帧没有到达")

        ratios: list[float | None] = [None] * len(exposures)
        mismatches = [0] * len(exposures)
        rejected = 0
        index = 0

        def wait_settled(exposure: float, slot: int):
            """改曝光后，连续两帧亮度一致才收下。和其它档对不上的帧不写入融合。"""
            camera.set_exposure_time_live(exposure)
            with lock:
                start_n = latest["n"]
            timeout = max(3.0, exposure / 1e6 * 8.0 + 1.0)
            end = time.monotonic() + timeout
            previous: tuple[int, float] | None = None
            reference = reference_ratio(ratios, slot)
            while time.monotonic() < end and not stop_event.is_set():
                with lock:
                    count = latest["n"]
                    frame = None if latest["frame"] is None else latest["frame"].copy()
                if frame is None or count < start_n + 2 or count > start_n + 6:
                    if count > start_n + 6:
                        break
                    time.sleep(0.001)
                    continue
                if previous is not None and count <= previous[0]:
                    time.sleep(0.001)
                    continue
                level = frame_level(frame)
                agreed = (
                    previous is not None
                    and previous[1] > 0
                    and abs(level - previous[1]) <= 0.15 * previous[1]
                )
                if agreed and (level_matches(level, exposure, reference) or mismatches[slot] >= 1):
                    mismatches[slot] = 0
                    return frame, level
                previous = (count, level)
                time.sleep(0.001)
            mismatches[slot] += 1
            return None, None

        while not stop_event.is_set():
            if seconds is not None and (time.perf_counter() - started) >= seconds:
                break
            exposure = min(float(exposures[index]), MAX_EXPOSURE_US)
            frame, level = wait_settled(exposure, index)
            if frame is None:
                rejected += 1
                index = (index + 1) % len(exposures)
                continue
            try:
                actual = float(camera.get_exposure_time_live())
            except Exception:
                actual = exposure
            if not np.isfinite(actual) or actual <= 0:
                actual = exposure
            ratios[index] = level / actual
            if on_raw is not None:
                on_raw(frame)
            view = fusion.push(index, frame, actual)
            index = (index + 1) % len(exposures)
            if view is None:
                continue
            shown += 1
            elapsed = time.perf_counter() - started
            on_frame(view, shown / elapsed if elapsed > 0 else 0.0)
        elapsed = time.perf_counter() - started
        return {
            "frames": shown,
            "seconds": elapsed,
            "fps": shown / elapsed if elapsed > 0 else 0.0,
            "rejected": rejected,
        }
    finally:
        stop.set()
        thread.join(timeout=8.0)


class FusionRecorder:
    """把融合画面写成 mp4，并记下每帧相对第一帧的时间。"""

    def __init__(self) -> None:
        self._path: Path | None = None
        self._writer = None
        self._pending: list[np.ndarray] = []
        self._times: list[float] = []
        self._t0: float | None = None
        self._fps = 15.0
        self._written = 0

    def set_active(self, path: str | None) -> None:
        if path is None:
            return
        target = Path(path)
        if self._path != target:
            self.close()
            self._path = target
            self._t0 = None

    def push(self, view: np.ndarray, now: float) -> None:
        if self._path is None:
            return
        if self._t0 is None:
            self._t0 = now
        self._times.append(now - self._t0)
        frame = np.ascontiguousarray(view)
        if self._writer is None:
            self._pending.append(frame)
            if len(self._pending) < 8:
                return
            span = self._times[-1] - self._times[0]
            rate = (len(self._pending) - 1) / span if span > 0.05 else 15.0
            self._fps = float(min(60.0, max(1.0, rate)))
            self._open()
            for item in self._pending:
                self._write(item)
            self._pending.clear()
            return
        self._write(frame)

    def close(self) -> dict | None:
        if self._path is None and self._writer is None and not self._pending:
            return None
        path = self._path
        if self._writer is None and self._pending and path is not None:
            span = self._times[-1] - self._times[0] if len(self._times) > 1 else 0.0
            rate = (len(self._pending) - 1) / span if span > 0.05 else 15.0
            self._fps = float(min(60.0, max(1.0, rate)))
            self._open()
            for item in self._pending:
                self._write(item)
        self._pending.clear()
        if self._writer is not None:
            self._writer.release()
            self._writer = None
        time_path = None
        if path is not None and self._times:
            time_path = path.with_name(path.stem + "_time.csv")
            lines = ["frame,seconds\n"]
            lines.extend(f"{index},{stamp:.6f}\n" for index, stamp in enumerate(self._times))
            time_path.write_text("".join(lines), encoding="utf-8")
        seconds = self._times[-1] if self._times else 0.0
        info = {
            "path": path,
            "time_path": time_path,
            "frames": self._written,
            "seconds": seconds,
            "fps": self._fps,
        }
        self._path = None
        self._times.clear()
        self._t0 = None
        self._written = 0
        if path is None or info["frames"] == 0:
            return None
        return info

    def _open(self) -> None:
        import cv2

        if self._path is None or not self._pending:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        height, width = self._pending[0].shape[:2]
        self._writer = cv2.VideoWriter(
            str(self._path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            self._fps,
            (width, height),
            True,
        )

    def _write(self, frame: np.ndarray) -> None:
        import cv2

        if self._writer is None:
            return
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        self._writer.write(frame)
        self._written += 1


def _record_clip() -> None:
    import sys
    from pathlib import Path

    import cv2

    runtime = Path(__file__).resolve().parents[1] / "clc_mbe_runtime"
    sys.path.insert(0, str(runtime))
    from src.lib.VimbaX_camera import VimbaCamera

    exposures = [1000.0, 2000.0, 4000.0, 16000.0]
    response = np.load(Path(__file__).resolve().parent / "response_detail" / "g_run1.npy")
    out = Path(__file__).resolve().parent / "video_check"
    out.mkdir(parents=True, exist_ok=True)
    frames: list[np.ndarray] = []

    def on_frame(view, _fps: float) -> None:
        frames.append(view.copy())

    camera = VimbaCamera()
    index = None
    try:
        camera.init()
        for i, device in enumerate(camera.get_all_devices_info()):
            blob = " ".join(
                str(device.get(key, ""))
                for key in ("model_name", "camera_name", "camera_id", "serial_number")
            )
            if "E0022060" in blob or "Manta_G-033B" in blob:
                index = i
        if index is None:
            raise RuntimeError("没有找到 Manta")
        camera.open_device(index)
        camera.setup_camera()
        stats = run_sliding(
            camera,
            exposures,
            response,
            on_frame,
            threading.Event(),
            seconds=5.0,
        )
    finally:
        camera.close()

    if not frames:
        raise RuntimeError("没有融合出画面")
    height, width = frames[0].shape
    fps = max(stats["fps"], 1.0)
    path = out / "fusion.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height), True)
    for view in frames:
        writer.write(cv2.cvtColor(view, cv2.COLOR_GRAY2BGR))
    writer.release()
    strip = np.hstack([cv2.resize(frames[int(i)], (160, 120)) for i in np.linspace(0, len(frames) - 1, 8)])
    cv2.imwrite(str(out / "strip.png"), strip)
    means = np.array([float(view.mean()) for view in frames])
    jumps = np.abs(np.diff(means))
    print(
        f"frames {stats['frames']} seconds {stats['seconds']:.2f} fps {stats['fps']:.2f} "
        f"rejected {stats['rejected']}"
    )
    print(
        f"mean std {means.std():.2f} min {means.min():.1f} max {means.max():.1f} "
        f"jump median {np.median(jumps):.2f} max {jumps.max():.1f}"
    )
    print(path)


if __name__ == "__main__":
    _record_clip()
