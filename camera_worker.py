#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""在工作线程里调用相机封装（VimbaCamera），不阻塞界面。"""

from __future__ import annotations

import json
import queue
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

def _record_status(saved: dict) -> str:
    path = saved["path"]
    time_path = saved.get("time_path")
    clock = f"，时间在 {time_path.name}" if time_path is not None else ""
    return (
        f"已保存 {path}，{saved['frames']} 帧，{saved['seconds']:.1f} 秒，"
        f"{saved['fps']:.1f} 帧/秒{clock}"
    )


def _cap_exposure(exposure_us: float) -> float:
    """写给相机的曝光不超过曝光限制里的 5 秒。"""
    from exposure_plan import MAX_EXPOSURE_US

    return min(float(exposure_us), MAX_EXPOSURE_US)


def _bgr_to_rgb(image):
    """Vimba 彩色帧是 BGR。单色帧若带一个长度为 1 的通道维，压成二维。"""
    arr = np.asarray(image)
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[..., 0]
    if arr.ndim == 3 and arr.shape[2] == 3:
        return np.ascontiguousarray(arr[..., ::-1])
    return np.ascontiguousarray(arr)


_RUNTIME_ROOT = Path(__file__).resolve().parents[1] / "clc_mbe_runtime"
if str(_RUNTIME_ROOT) not in sys.path:
    sys.path.insert(0, str(_RUNTIME_ROOT))


class CameraWorker(QThread):
    """预览与曝光序列都在这条线程里执行，避免和相机上下文互相嵌套。"""

    devices_listed = pyqtSignal(object)
    preview_frame = pyqtSignal(object)
    bracket_done = pyqtSignal(object)
    plan_ready = pyqtSignal(object)
    calibration_done = pyqtSignal(object)
    video_frame = pyqtSignal(object)
    video_stopped = pyqtSignal()
    progress = pyqtSignal(int, int)
    status_text = pyqtSignal(str)
    error_text = pyqtSignal(str)
    connected_changed = pyqtSignal(bool)

    def __init__(self) -> None:
        super().__init__()
        self._commands: queue.Queue[tuple[str, object]] = queue.Queue()
        self._stream_stop = threading.Event()
        self._quit = False
        self._preview = False
        self._gain = 0.0
        self._camera = None
        self._last_preview = 0.0
        self._video_stop = threading.Event()
        self._video_status = 0.0
        self._record_lock = threading.Lock()
        self._record_enabled = False
        self._record_path: str | None = None
        self._video_fusion_lock = threading.Lock()
        self._video_fusion = None

    def request_list(self) -> None:
        self._commands.put(("list", None))
        self._stream_stop.set()

    def request_connect(self, index: int) -> None:
        self._commands.put(("connect", int(index)))
        self._stream_stop.set()

    def request_disconnect(self) -> None:
        self._video_stop.set()
        self._commands.put(("disconnect", None))
        self._stream_stop.set()

    def request_bracket(self, exposures_us: list[float]) -> None:
        self._commands.put(("bracket", [float(value) for value in exposures_us]))
        self._stream_stop.set()

    def request_plan(self, sat_limit: float, t_max_us: float) -> None:
        self._commands.put(("plan", (float(sat_limit), float(t_max_us))))
        self._stream_stop.set()

    def request_calibrate(self, gain_steps: int = 7) -> None:
        self._commands.put(("calibrate", int(gain_steps)))
        self._stream_stop.set()

    def request_video(
        self,
        exposures_us: list[float],
        response: np.ndarray,
        window: tuple[float, float] | None = None,
    ) -> None:
        self._video_stop.clear()
        packed = None if window is None else (float(window[0]), float(window[1]))
        self._commands.put(
            ("video", ([float(value) for value in exposures_us], np.asarray(response), packed))
        )
        self._stream_stop.set()

    def set_video_window(self, low: float, high: float) -> None:
        """视频进行中改用和辐射图相同的对数窗口。"""
        with self._video_fusion_lock:
            fusion = self._video_fusion
        if fusion is not None:
            fusion.set_display_window(low, high)

    def request_video_stop(self) -> None:
        self._video_stop.set()

    def request_record(self, path: str) -> None:
        with self._record_lock:
            self._record_path = str(path)
            self._record_enabled = True

    def request_record_stop(self) -> None:
        with self._record_lock:
            self._record_enabled = False

    def set_gain(self, gain: float) -> None:
        """流已经打开时直接写 Gain；否则等下一次开流时在 on_live 里写。"""
        self._gain = float(gain)
        camera = self._camera
        if camera is not None and camera.is_live_ready():
            try:
                camera.set_gain_live(self._gain)
            except Exception as exc:
                self.error_text.emit(f"设置增益失败: {exc}")

    def shutdown(self) -> None:
        self._quit = True
        self._preview = False
        self._video_stop.set()
        self._stream_stop.set()
        self._commands.put(("quit", None))

    def run(self) -> None:
        try:
            while not self._quit:
                self._drain_available()
                if self._quit:
                    break
                if self._preview and self._camera is not None and self._camera.is_opened:
                    self._run_preview()
                    continue
                self._wait_one()
        finally:
            self._close_camera()

    def _drain_available(self) -> None:
        while not self._quit:
            try:
                command, arg = self._commands.get_nowait()
            except queue.Empty:
                return
            self._handle(command, arg)

    def _wait_one(self) -> None:
        try:
            command, arg = self._commands.get(timeout=0.2)
        except queue.Empty:
            return
        self._handle(command, arg)

    def _handle(self, command: str, arg: object) -> None:
        if command == "quit":
            self._quit = True
            self._preview = False
        elif command == "list":
            self._do_list()
        elif command == "connect":
            self._do_connect(int(arg))
        elif command == "disconnect":
            self._do_disconnect()
        elif command == "bracket":
            self._do_bracket(list(arg))  # type: ignore[arg-type]
        elif command == "plan":
            sat_limit, t_max_us = arg  # type: ignore[misc]
            self._do_plan(float(sat_limit), float(t_max_us))
        elif command == "calibrate":
            self._do_calibrate(int(arg))
        elif command == "video":
            exposures, response, window = arg  # type: ignore[misc]
            self._do_video(list(exposures), response, window)

    def _do_list(self) -> None:
        try:
            self._ensure_camera()
            infos = self._camera.get_all_devices_info()
            self.devices_listed.emit(infos)
            self.status_text.emit(f"找到 {len(infos)} 台相机")
        except Exception as exc:
            self.error_text.emit(f"枚举相机失败: {exc}")

    def _do_connect(self, index: int) -> None:
        try:
            self._close_camera()
            from src.lib.VimbaX_camera import VimbaCamera

            camera = VimbaCamera()
            camera.init()
            camera.open_device(index)
            camera.setup_camera()
            self._camera = camera
            self._preview = True
            self.connected_changed.emit(True)
            self.status_text.emit("已连接，正在预览")
        except Exception as exc:
            self._close_camera()
            self.connected_changed.emit(False)
            self.error_text.emit(f"连接相机失败: {exc}")

    def _do_video(
        self,
        exposures: list[float],
        response: np.ndarray,
        window: tuple[float, float] | None = None,
    ) -> None:
        from sliding_fusion import FusionRecorder, SlidingFusion, run_sliding

        camera = self._camera
        if camera is None or not camera.is_opened:
            self.error_text.emit("尚未连接相机")
            self.video_stopped.emit()
            return
        recorder = FusionRecorder()
        fusion = SlidingFusion(response, len(exposures), window=window)
        with self._video_fusion_lock:
            self._video_fusion = fusion

        def on_frame(view, fps: float) -> None:
            self.video_frame.emit(view)
            with self._record_lock:
                enabled = self._record_enabled
                path = self._record_path
            saved = None
            if enabled and path:
                recorder.set_active(path)
                recorder.push(view, time.perf_counter())
            else:
                saved = recorder.close()
            now = time.monotonic()
            if saved:
                self.status_text.emit(_record_status(saved))
                self._video_status = now + 2.5
            elif now - self._video_status >= 0.5:
                self._video_status = now
                if enabled and path:
                    self.status_text.emit(f"录制 {fps:.1f} 帧/秒")
                else:
                    self.status_text.emit(f"视频 {fps:.1f} 帧/秒")

        def on_raw(frame) -> None:
            now = time.monotonic()
            if now - self._last_preview < 1.0 / 15.0:
                return
            self._last_preview = now
            self.preview_frame.emit(frame)

        try:
            stats = run_sliding(
                camera,
                [_cap_exposure(value) for value in exposures],
                response,
                on_frame,
                self._video_stop,
                on_raw=on_raw,
                fusion=fusion,
                gain=self._gain,
            )
            self.status_text.emit(f"视频结束，{stats['fps']:.1f} 帧/秒，共 {stats['frames']} 帧")
        except Exception as exc:
            self.error_text.emit(f"视频失败: {exc}")
        finally:
            with self._video_fusion_lock:
                self._video_fusion = None
            saved = recorder.close()
            with self._record_lock:
                self._record_enabled = False
                self._record_path = None
            if saved:
                self.status_text.emit(_record_status(saved))
            self.video_stopped.emit()

    def _do_disconnect(self) -> None:
        self._video_stop.set()
        self._preview = False
        self._close_camera()
        self.connected_changed.emit(False)
        self.status_text.emit("相机已断开")

    def _do_bracket(self, exposures: list[float]) -> None:
        """流不关。改曝光后丢掉正在积分的 1 帧，留下下一帧。"""
        camera = self._camera
        if camera is None or not camera.is_opened:
            self.error_text.emit("尚未连接相机")
            return
        if len(exposures) < 2:
            self.error_text.emit("至少需要两档曝光")
            return
        latest = {"n": 0, "frame": None}
        lock = threading.Lock()
        stop = threading.Event()

        def on_frame(image) -> None:
            frame = _bgr_to_rgb(image)
            with lock:
                latest["n"] += 1
                latest["frame"] = np.asarray(frame).copy()

        def on_live() -> None:
            for action in (
                lambda: camera.set_exposure_auto_live("Off"),
                lambda: camera.set_gain_auto_live("Off"),
                lambda: camera.set_gamma_live(1.0),
                lambda: camera.set_gain_live(self._gain),
            ):
                try:
                    action()
                except Exception:
                    pass

        thread = threading.Thread(
            target=lambda: camera.run_streaming_session(on_frame, stop, on_live=on_live),
            daemon=True,
        )
        try:
            thread.start()
            if not camera.wait_until_live(8.0):
                raise RuntimeError("相机流没有启动")
            deadline = time.monotonic() + 8.0
            while time.monotonic() < deadline:
                with lock:
                    ready = latest["frame"] is not None
                if ready:
                    break
                time.sleep(0.01)
            else:
                raise RuntimeError("预览帧没有到达")

            kept: list[tuple[float, object]] = []
            total = len(exposures)
            started = time.perf_counter()
            for index, exposure in enumerate(exposures, start=1):
                exposure = _cap_exposure(exposure)
                self.status_text.emit(f"采集 {index}/{total}，曝光 {exposure:.0f} μs")
                self.progress.emit(index - 1, total)
                camera.set_exposure_time_live(exposure)
                with lock:
                    start_n = latest["n"]
                # 序号 +1 仍是旧曝光，+2 才是新曝光。
                need = start_n + 2
                timeout = max(3.0, float(exposure) / 1e6 * 8.0 + 1.0)
                end = time.monotonic() + timeout
                frame = None
                while time.monotonic() < end:
                    with lock:
                        count = latest["n"]
                        frame = None if latest["frame"] is None else latest["frame"].copy()
                    if count >= need and frame is not None:
                        break
                    time.sleep(0.001)
                else:
                    raise TimeoutError(f"曝光 {exposure:.0f} μs 没有等到稳定帧")
                actual = float(camera.get_exposure_time_live())
                kept.append((actual, frame))
                self.preview_frame.emit(frame)
            elapsed = time.perf_counter() - started
            self.bracket_done.emit(kept)
            self.progress.emit(total, total)
            self.status_text.emit(f"已采集 {total} 张，用时 {elapsed:.2f} s")
        except Exception as exc:
            self.error_text.emit(f"采集失败: {exc}")
        finally:
            stop.set()
            thread.join(timeout=8.0)

    def _do_plan(self, sat_limit: float, t_max_us: float) -> None:
        from exposure_plan import MAX_EXPOSURE_US, frame_metrics, plan_bracket

        camera = self._camera
        if camera is None or not camera.is_opened:
            self.error_text.emit("尚未连接相机")
            return
        latest = {"n": 0, "frame": None}
        lock = threading.Lock()
        stop = threading.Event()
        meta: dict = {}

        def on_frame(image) -> None:
            frame = _bgr_to_rgb(image)
            with lock:
                latest["n"] += 1
                latest["frame"] = np.asarray(frame).copy()

        def on_live() -> None:
            for action in (
                lambda: camera.set_exposure_auto_live("Off"),
                lambda: camera.set_gain_auto_live("Off"),
                lambda: camera.set_gamma_live(1.0),
                lambda: camera.set_gain_live(self._gain),
            ):
                try:
                    action()
                except Exception:
                    pass
            try:
                name = camera._probe_exposure_feature_name()
                low, high = getattr(camera._camera, name).get_range()
                meta["range"] = (float(low), float(high))
            except Exception:
                meta["range"] = (26.0, MAX_EXPOSURE_US)

        thread = threading.Thread(
            target=lambda: camera.run_streaming_session(on_frame, stop, on_live=on_live),
            daemon=True,
        )
        try:
            thread.start()
            if not camera.wait_until_live(8.0):
                raise RuntimeError("相机流没有启动")
            deadline = time.monotonic() + 8.0
            while time.monotonic() < deadline:
                with lock:
                    ready = latest["frame"] is not None
                if ready and "range" in meta:
                    break
                time.sleep(0.02)
            else:
                raise RuntimeError("预览帧没有到达")

            t_min, t_cam_max = meta["range"]
            t_cap = min(float(t_cam_max), float(t_max_us), MAX_EXPOSURE_US)
            samples: dict[float, tuple[float, float, int]] = {}

            def measure(exposure: float) -> tuple[float, float, float, int]:
                use = float(min(t_cap, max(t_min, exposure)))
                camera.set_exposure_time_live(use)
                with lock:
                    start = latest["n"]
                timeout = max(4.0, (use / 1e6) * 8.0 + 1.5)
                end = time.monotonic() + timeout
                frame = None
                while time.monotonic() < end:
                    with lock:
                        count = latest["n"]
                        frame = None if latest["frame"] is None else latest["frame"].copy()
                    if count >= start + 3 and frame is not None:
                        break
                    time.sleep(0.005)
                else:
                    raise TimeoutError(f"曝光 {use:.0f} μs 没有等到稳定帧")
                actual = float(camera.get_exposure_time_live())
                scale = 255 if np.asarray(frame).dtype == np.uint8 else 4095
                saturated, peak = frame_metrics(frame, scale)
                samples[actual] = (saturated, peak, scale)
                self.preview_frame.emit(frame)
                self.status_text.emit(f"试曝光 {actual:.0f} μs，饱和 {saturated * 100:.1f}%")
                return actual, saturated, peak, scale

            current, saturated, peak, scale = measure(max(t_min, 1000.0))
            steps = 1
            while steps < 14 and current / 2.0 >= t_min and (saturated > sat_limit or peak > 0.85 * scale):
                nxt, saturated, peak, scale = measure(current / 2.0)
                steps += 1
                if nxt >= current * 0.95:
                    current = nxt
                    break
                current = nxt
            while steps < 14 and saturated <= sat_limit and current * 2.0 <= t_cap:
                nxt, saturated, peak, scale = measure(current * 2.0)
                steps += 1
                if nxt <= current * 1.05:
                    current = nxt
                    break
                current = nxt

            exposures = sorted(samples)
            saturations = [samples[item][0] for item in exposures]
            peaks = [samples[item][1] for item in exposures]
            scale = int(samples[exposures[-1]][2])
            times, note = plan_bracket(
                exposures,
                saturations,
                peaks,
                sat_limit=sat_limit,
                full_scale=scale,
                t_min=t_min,
                t_max=t_cap,
            )
            self.plan_ready.emit((times, note))
        except Exception as exc:
            self.error_text.emit(f"自动选档失败: {exc}")
        finally:
            stop.set()
            thread.join(timeout=8.0)

    def _do_calibrate(self, gain_steps: int) -> None:
        """在相机完整增益范围内逐档采两轮详细曝光并立即分析、落盘。"""
        from calibration_plan import DEFAULT_EXPOSURES_US, gain_targets
        from exposure_plan import MAX_EXPOSURE_US
        from measure_response import analyze_and_save_gain

        camera = self._camera
        if camera is None or not camera.is_opened:
            self.error_text.emit("尚未连接相机")
            return
        latest = {"n": 0, "frame": None}
        lock = threading.Lock()
        stop = threading.Event()
        meta: dict = {}

        def on_frame(image) -> None:
            if image is None:
                return
            frame = _bgr_to_rgb(image)
            with lock:
                latest["n"] += 1
                latest["frame"] = np.asarray(frame).copy()

        def on_live() -> None:
            try:
                gain_low, gain_high = camera._camera.Gain.get_range()
                try:
                    gain_increment = getattr(camera._camera.Gain, "get_increment", lambda: None)()
                except Exception:
                    gain_increment = None
                if gain_increment is not None and float(gain_increment) <= 0:
                    gain_increment = None
                name = camera._probe_exposure_feature_name()
                exposure_low, exposure_high = getattr(camera._camera, name).get_range()
                meta.update(
                    gain_range=(float(gain_low), float(gain_high)),
                    gain_increment=None if gain_increment is None else float(gain_increment),
                    exposure_range=(float(exposure_low), float(exposure_high)),
                    camera_serial=str(camera._camera.get_serial() or camera.camera_id or "unknown"),
                    pixel_format=str(camera.get_pixel_format_live()),
                )
            except Exception as exc:
                meta["error"] = str(exc)

        stream: dict = {"thread": None}

        def start_stream() -> None:
            stop.clear()
            with lock:
                latest["n"] = 0
                latest["frame"] = None
            thread = threading.Thread(
                target=lambda: camera.run_streaming_session(on_frame, stop, on_live=on_live),
                daemon=True,
            )
            stream["thread"] = thread
            thread.start()
            if not camera.wait_until_live(8.0):
                raise RuntimeError("相机流没有启动")
            deadline = time.monotonic() + 8.0
            while time.monotonic() < deadline:
                with lock:
                    ready = latest["frame"] is not None
                if ready and ("gain_range" in meta or "error" in meta):
                    return
                time.sleep(0.02)
            raise RuntimeError("预览帧没有到达")

        def stop_stream() -> None:
            stop.set()
            thread = stream["thread"]
            if thread is not None:
                thread.join(timeout=8.0)
                stream["thread"] = None

        root = Path(__file__).resolve().parent / "response_calibrations" / datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )
        manifest: dict = {
            "schema_version": 1,
            "created_at": datetime.now().astimezone().isoformat(),
            "requested_gain_steps": int(gain_steps),
            "entries": [],
        }
        try:
            start_stream()
            if "error" in meta:
                raise RuntimeError(f"读取相机标定能力失败: {meta['error']}")
            # on_live 在切到 Mono12 之前执行，这里按已经开始的采集流重新读格式。
            meta["pixel_format"] = str(camera.get_pixel_format_live())
            if meta["pixel_format"] != "Mono12":
                raise RuntimeError(f"标定要求 Mono12，当前为 {meta['pixel_format']}")
            if meta["gain_increment"] is None:
                from calibration_plan import infer_gain_increment

                def read_after_write(value: float) -> float:
                    camera.set_gain_live(float(value))
                    return float(camera.get_gain_live())

                inferred = infer_gain_increment(*meta["gain_range"], read_after_write)
                if inferred is not None:
                    meta["gain_increment"] = inferred

            gains = gain_targets(
                *meta["gain_range"],
                gain_steps,
                increment=meta["gain_increment"],
            )
            manifest.update(
                camera_serial=meta["camera_serial"],
                pixel_format=meta["pixel_format"],
                gamma=1.0,
                gain_range_db=list(meta["gain_range"]),
                gain_increment_db=meta["gain_increment"],
                requested_gain_targets_db=gains,
                exposures_requested_us=list(DEFAULT_EXPOSURES_US),
            )
            root.mkdir(parents=True, exist_ok=True)
            total = len(gains) * 2 * len(DEFAULT_EXPOSURES_US)
            completed = 0

            def set_and_verify(target_gain: float) -> float:
                camera.set_exposure_auto_live("Off")
                camera.set_gain_auto_live("Off")
                camera.set_gamma_live(1.0)
                camera.set_gain_live(target_gain)
                exposure_auto = str(camera.get_exposure_auto_live())
                gain_auto = str(camera.get_gain_auto_live())
                gamma = float(camera.get_gamma_live())
                actual_gain = float(camera.get_gain_live())
                exposure_mode = exposure_auto.rsplit(".", 1)[-1].lower()
                gain_mode = gain_auto.rsplit(".", 1)[-1].lower()
                if exposure_mode != "off" or gain_mode != "off":
                    raise RuntimeError(
                        f"自动参数未关闭: ExposureAuto={exposure_auto}, GainAuto={gain_auto}"
                    )
                if not np.isclose(gamma, 1.0, atol=1e-6):
                    raise RuntimeError(f"Gamma 写入 1 后读回 {gamma}")
                if not meta["gain_range"][0] - 1e-6 <= actual_gain <= meta["gain_range"][1] + 1e-6:
                    raise RuntimeError(f"增益读回值 {actual_gain} 超出范围")
                gain_tolerance = (
                    max(1e-6, float(meta["gain_increment"]) * 0.51)
                    if meta["gain_increment"] is not None
                    else 1e-6
                )
                if not np.isclose(actual_gain, target_gain, atol=gain_tolerance):
                    raise RuntimeError(
                        f"增益写入 {target_gain} dB 后读回 {actual_gain} dB"
                    )
                return actual_gain

            def one_ladder(gain_index: int, round_index: int) -> tuple[list[np.ndarray], np.ndarray]:
                nonlocal completed
                frames: list[np.ndarray] = []
                actual_times: list[float] = []
                low, high = meta["exposure_range"]
                cap = min(float(high), MAX_EXPOSURE_US)
                requested = [value for value in DEFAULT_EXPOSURES_US if low <= value <= cap]
                if len(requested) < 4:
                    raise RuntimeError(f"相机曝光范围 {low}–{cap} μs 无法形成详细序列")
                for exposure in requested:
                    for attempt in range(2):
                        camera.set_exposure_time_live(float(exposure))
                        with lock:
                            start = latest["n"]
                        end = time.monotonic() + max(6.0, exposure / 1e6 * 8.0 + 2.0)
                        frame = None
                        while time.monotonic() < end:
                            with lock:
                                count = latest["n"]
                                frame = (
                                    None if latest["frame"] is None else latest["frame"].copy()
                                )
                            if count >= start + 3 and frame is not None:
                                break
                            time.sleep(0.005)
                        else:
                            if attempt == 0:
                                self.status_text.emit("采集中断，重新开流")
                                stop_stream()
                                start_stream()
                                continue
                            raise TimeoutError(f"曝光 {exposure:.0f} μs 没有等到稳定帧")
                        break
                    actual = float(camera.get_exposure_time_live())
                    if not np.isfinite(actual) or actual <= 0:
                        raise RuntimeError(f"曝光读回非法: {actual}")
                    if frame.dtype != np.uint16 or int(np.max(frame)) > 4095:
                        raise RuntimeError(
                            f"帧不是有效 Mono12: dtype={frame.dtype}, max={int(np.max(frame))}"
                        )
                    frames.append(frame)
                    actual_times.append(actual)
                    completed += 1
                    self.progress.emit(completed, total)
                    self.preview_frame.emit(frame)
                    self.status_text.emit(
                        f"增益 {gain_index + 1}/{len(gains)}，第 {round_index} 轮，"
                        f"曝光 {actual / 1000:.3g} ms"
                    )
                return frames, np.asarray(actual_times, dtype=np.float64)

            successful: list[dict] = []
            for gain_index, target in enumerate(gains):
                entry: dict = {
                    "index": gain_index,
                    "requested_gain_db": float(target),
                    "status": "failed",
                }
                try:
                    actual_gain = set_and_verify(float(target))
                    entry["actual_gain_db"] = actual_gain
                    frames_a, times_a = one_ladder(gain_index, 1)
                    frames_b, times_b = one_ladder(gain_index, 2)
                    subdir = f"gain_{gain_index:02d}_{actual_gain:+.3f}dB"
                    # 求解会长时间占住解释器。先停流，避免 GigE 缓冲耗尽后不再出帧。
                    stop_stream()
                    result = analyze_and_save_gain(
                        root / subdir,
                        frames_a,
                        times_a,
                        frames_b,
                        times_b,
                        {
                            "camera_serial": meta["camera_serial"],
                            "pixel_format": meta["pixel_format"],
                            "gamma": 1.0,
                            "requested_gain_db": float(target),
                            "actual_gain_db": actual_gain,
                        },
                    )
                    paths = {name: f"{subdir}/{path}" for name, path in result["paths"].items()}
                    entry.update(status="ok", quality=result["quality"], paths=paths)
                    successful.append({**entry, "curve": result["curve"]})
                except Exception as exc:
                    entry["error"] = str(exc)
                    self.status_text.emit(
                        f"增益 {gain_index + 1}/{len(gains)} 失败，继续下一档: {exc}"
                    )
                finally:
                    stop_stream()
                    start_stream()
                manifest["entries"].append(entry)
                (root / "manifest.json").write_text(
                    json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
                )

            manifest["completed_gain_steps"] = len(successful)
            manifest["failed_gain_steps"] = len(gains) - len(successful)
            passed = [item for item in successful if item["quality"]["passed"]]
            choices = passed or successful
            if choices:
                best = min(
                    choices,
                    key=lambda item: (
                        item["quality"]["repeat_rmse_ln"],
                        item["quality"]["held_out_max_rmse_ln"],
                    ),
                )
                manifest["recommended_gain_index"] = int(best["index"])
            (root / "manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            if not successful:
                raise RuntimeError(f"全部 {len(gains)} 档增益标定失败，详见 {root / 'manifest.json'}")
            self._gain = float(best["actual_gain_db"])
            manifest["recommended_gain_db"] = self._gain
            try:
                camera.set_gain_live(self._gain)
                restored_gain = float(camera.get_gain_live())
                restore_tolerance = (
                    max(1e-6, float(meta["gain_increment"]) * 0.51)
                    if meta["gain_increment"] is not None
                    else 1e-6
                )
                if not np.isclose(restored_gain, self._gain, atol=restore_tolerance):
                    raise RuntimeError(
                        f"目标 {self._gain} dB，读回 {restored_gain} dB"
                    )
            except Exception as exc:
                manifest["restore_recommended_gain"] = {
                    "status": "failed",
                    "error": str(exc),
                }
                (root / "manifest.json").write_text(
                    json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                raise RuntimeError(f"标定后恢复推荐增益失败: {exc}") from exc
            self._gain = restored_gain
            manifest["restore_recommended_gain"] = {
                "status": "ok",
                "actual_gain_db": restored_gain,
            }
            (root / "manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            self.calibration_done.emit(
                {
                    "curve": best["curve"],
                    "gain": self._gain,
                    "quality": best["quality"],
                    "manifest": str(root / "manifest.json"),
                    "successful": len(successful),
                    "failed": len(gains) - len(successful),
                }
            )
        except Exception as exc:
            self.error_text.emit(f"标定失败: {exc}")
        finally:
            stop_stream()

    def _run_preview(self) -> None:
        camera = self._camera
        if camera is None:
            return
        self._stream_stop.clear()
        started = time.monotonic()
        try:
            camera.run_streaming_session(
                self._on_frame,
                self._stream_stop,
                on_live=self._on_live,
            )
        except Exception as exc:
            self.error_text.emit(f"预览中断: {exc}")
            self._preview = False
            return
        if (not self._stream_stop.is_set()) and (time.monotonic() - started) < 0.3:
            self.error_text.emit("预览未能启动")
            self._preview = False

    def _on_live(self) -> None:
        camera = self._camera
        if camera is None:
            return
        actions = (
            ("自动曝光", lambda: camera.set_exposure_auto_live("Off")),
            ("自动增益", lambda: camera.set_gain_auto_live("Off")),
            ("Gamma", lambda: camera.set_gamma_live(1.0)),
            ("增益", lambda: camera.set_gain_live(self._gain)),
        )
        for label, action in actions:
            try:
                action()
            except Exception as exc:
                self.status_text.emit(f"{label} 未能锁定: {exc}")

    def _on_frame(self, image) -> None:
        if image is None:
            return
        image = _bgr_to_rgb(image)
        now = time.monotonic()
        if now - self._last_preview < 1.0 / 15.0:
            return
        self._last_preview = now
        self.preview_frame.emit(image)

    def _ensure_camera(self) -> None:
        if self._camera is None:
            from src.lib.VimbaX_camera import VimbaCamera

            self._camera = VimbaCamera()
            self._camera.init()

    def _close_camera(self) -> None:
        camera = self._camera
        self._camera = None
        if camera is None:
            return
        try:
            camera.close()
        except Exception:
            pass
