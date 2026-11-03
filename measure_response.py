#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""按 Debevec & Malik 1997 实测相机响应曲线。

固定增益和 Gamma，连拍两轮已知曝光。曲线用正文式 (3) 求解。
可靠性看三件事：式 (2) 的加权残差、没参与拟合的曝光能否落在同一条
曲线上、两轮独立采集的 g(Z) 是否一致。原始帧会留下。
"""

from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from debevec import hat_weights, recover

RUNTIME = Path(__file__).resolve().parents[1] / "clc_mbe_runtime"
if str(RUNTIME) not in sys.path:
    sys.path.insert(0, str(RUNTIME))

from src.lib.VimbaX_camera import VimbaCamera  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "response_detail"
SERIAL = "E0022060"
# 1 档一档，覆盖上次实拍里从近黑到饱和的区间，并向两端各延伸一档。
REQUESTED_US = np.array(
    [500, 1_000, 2_000, 4_000, 8_000, 16_000, 32_000, 64_000, 128_000, 256_000, 512_000],
    dtype=np.float64,
)
LAM = 10.0
N_PIXELS = 4000
ZMIN = 0
ZMAX = 4095


def _try_read(action):
    try:
        return action()
    except Exception as exc:
        return f"读取失败: {exc}"


def squeeze(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    return np.ascontiguousarray(arr)


def weighted_rmse(residual: np.ndarray, weight: np.ndarray, mask: np.ndarray) -> float:
    """式 (3) 里数据项用 w*r，所以按 w^2 加权。单位是 ln E。"""
    ww = weight[mask] ** 2
    total = float(np.sum(ww))
    if total <= 0:
        return float("nan")
    return float(np.sqrt(np.sum(ww * residual[mask] ** 2) / total))


def fit_metrics(images: list[np.ndarray], times: np.ndarray, result) -> dict:
    g = np.asarray(result.g, dtype=np.float64).ravel()
    wtab = hat_weights(result.zmin, result.zmax)
    log_t = np.log(np.asarray(times, dtype=np.float64))
    den = np.zeros(images[0].shape, dtype=np.float64)
    num = np.zeros(images[0].shape, dtype=np.float64)
    per_stop = []
    for image, lt in zip(images, log_t):
        codes = np.clip(image.astype(np.int32), result.zmin, result.zmax)
        weight = wtab[codes - result.zmin]
        predicted = g[codes - result.zmin] - lt
        den += weight
        num += weight * predicted
    ln_e = np.log(np.maximum(result.radiance.astype(np.float64), 1e-30))
    sum_w2_r2 = 0.0
    sum_w2 = 0.0
    pixel_var = np.zeros(images[0].shape, dtype=np.float64)
    for image, lt, exposure in zip(images, log_t, times):
        codes = np.clip(image.astype(np.int32), result.zmin, result.zmax)
        weight = wtab[codes - result.zmin]
        residual = g[codes - result.zmin] - lt - ln_e
        mask = (weight > 0) & (den > 0)
        ww = weight[mask] ** 2
        sum_w2_r2 += float(np.sum(ww * residual[mask] ** 2))
        sum_w2 += float(np.sum(ww))
        pixel_var += np.where(mask, (weight ** 2) * (residual ** 2), 0.0)
        per_stop.append(
            {
                "requested_note": float(exposure),
                "median_code": float(np.median(codes)),
                "p99_code": float(np.percentile(codes, 99)),
                "frac_black": float(np.mean(codes <= 1)),
                "frac_saturated": float(np.mean(codes >= ZMAX - 1)),
                "rmse_ln": weighted_rmse(residual, weight, mask),
            }
        )
    pixel_den = np.zeros(images[0].shape, dtype=np.float64)
    for image in images:
        codes = np.clip(image.astype(np.int32), result.zmin, result.zmax)
        weight = wtab[codes - result.zmin]
        pixel_den += weight ** 2
    good = pixel_den > 0
    pixel_rms = np.sqrt(pixel_var[good] / pixel_den[good])
    rmse = float(np.sqrt(sum_w2_r2 / sum_w2)) if sum_w2 > 0 else float("nan")
    return {
        "rmse_ln": rmse,
        "rmse_stops": rmse / np.log(2.0),
        "rmse_rel": float(np.expm1(rmse)),
        "pixel_rms_median": float(np.median(pixel_rms)),
        "pixel_rms_p90": float(np.percentile(pixel_rms, 90)),
        "per_stop": per_stop,
        "n_used": result.n_used,
        "n_required": result.n_required,
    }


def coverage(images: list[np.ndarray]) -> dict:
    hist = np.zeros(ZMAX + 1, dtype=np.int64)
    stack = np.stack([np.clip(im.astype(np.int32), 0, ZMAX) for im in images], axis=0)
    for plane in stack:
        hist += np.bincount(plane.ravel(), minlength=ZMAX + 1)
    observed = hist > 0
    interior = observed[1:ZMAX]
    gaps = []
    run = 0
    for seen in interior:
        if seen:
            if run:
                gaps.append(run)
            run = 0
        else:
            run += 1
    if run:
        gaps.append(run)
    span = stack.max(axis=0) - stack.min(axis=0)
    return {
        "hist": hist,
        "distinct": int(observed.sum()),
        "interior_frac": float(interior.mean()),
        "frac_count_ge_100": float(np.mean(hist[1:ZMAX] >= 100)),
        "max_gap": int(max(gaps) if gaps else 0),
        "pixels_span_1000": int(np.sum(span >= 1000)),
        "pixels_span_2000": int(np.sum(span >= 2000)),
        "pixels_span_3000": int(np.sum(span >= 3000)),
        "median_span": float(np.median(span)),
    }


def held_out(images: list[np.ndarray], times: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray) -> float:
    train_images = [images[int(i)] for i in train_idx]
    result = recover(train_images, times[train_idx], lam=LAM, n_pixels=N_PIXELS)
    g = np.asarray(result.g, dtype=np.float64).ravel()
    wtab = hat_weights(result.zmin, result.zmax)
    den = np.zeros(images[0].shape, dtype=np.float64)
    for image in train_images:
        codes = np.clip(image.astype(np.int32), result.zmin, result.zmax)
        den += wtab[codes - result.zmin]
    ln_e = np.log(np.maximum(result.radiance.astype(np.float64), 1e-30))
    sum_w2_r2 = 0.0
    sum_w2 = 0.0
    for index in test_idx:
        codes = np.clip(images[int(index)].astype(np.int32), result.zmin, result.zmax)
        weight = wtab[codes - result.zmin]
        residual = g[codes - result.zmin] - np.log(float(times[int(index)])) - ln_e
        mask = (weight > 0) & (den > 0)
        ww = weight[mask] ** 2
        sum_w2_r2 += float(np.sum(ww * residual[mask] ** 2))
        sum_w2 += float(np.sum(ww))
    if sum_w2 <= 0:
        return float("nan")
    return float(np.sqrt(sum_w2_r2 / sum_w2))


def curve_difference(g_a: np.ndarray, g_b: np.ndarray, hist: np.ndarray) -> dict:
    delta = np.asarray(g_a, dtype=np.float64).ravel() - np.asarray(g_b, dtype=np.float64).ravel()
    measured = hist[1:ZMAX] >= 100
    body = delta[1:ZMAX][measured]
    if body.size == 0:
        body = delta
    return {
        "rmse_measured": float(np.sqrt(np.mean(body ** 2))),
        "max_measured": float(np.max(np.abs(body))),
        "rmse_all": float(np.sqrt(np.mean(delta ** 2))),
    }


def validate_mono12_pair(
    frames_a: list[np.ndarray],
    times_a: np.ndarray,
    frames_b: list[np.ndarray],
    times_b: np.ndarray,
) -> None:
    """拒绝非法曝光、非整数帧、越界码值和不一致的两轮序列。"""
    if len(frames_a) < 4 or len(frames_a) != len(frames_b):
        raise ValueError("两轮标定必须有相同且不少于 4 档的曝光")
    reference_shape = np.asarray(frames_a[0]).shape
    for run, frames, times in ((1, frames_a, times_a), (2, frames_b, times_b)):
        exposure = np.asarray(times, dtype=np.float64).reshape(-1)
        if exposure.size != len(frames) or np.any(~np.isfinite(exposure)) or np.any(exposure <= 0):
            raise ValueError(f"第 {run} 轮实际曝光含非法值")
        if np.any(np.diff(exposure) <= 0):
            raise ValueError(f"第 {run} 轮实际曝光没有严格递增")
        for index, image in enumerate(frames, start=1):
            data = np.asarray(image)
            if data.shape != reference_shape or data.ndim != 2:
                raise ValueError(f"第 {run} 轮第 {index} 帧尺寸或通道不一致")
            if not np.issubdtype(data.dtype, np.integer):
                raise ValueError(f"第 {run} 轮第 {index} 帧不是整数 Mono12")
            if data.size == 0 or int(np.min(data)) < 0 or int(np.max(data)) > ZMAX:
                raise ValueError(f"第 {run} 轮第 {index} 帧码值超出 Mono12 0–{ZMAX}")


def analyze_pair(images: list[np.ndarray], times: np.ndarray) -> tuple[dict, object]:
    result = recover(images, times, lam=LAM, n_pixels=N_PIXELS)
    metrics = fit_metrics(images, times, result)
    cover = coverage(images)
    indices = np.arange(len(images))
    even = indices[0::2]
    odd = indices[1::2]
    metrics["held_even_train_odd_test_ln"] = held_out(images, times, even, odd)
    metrics["held_odd_train_even_test_ln"] = held_out(images, times, odd, even)
    metrics["coverage"] = cover
    minimum = recover(images, times, lam=LAM, n_pixels=result.n_required)
    metrics["sample_stability"] = curve_difference(result.g, minimum.g, cover["hist"])
    return metrics, result


def analyze_and_save_gain(
    directory: Path,
    frames_a: list[np.ndarray],
    times_a: np.ndarray,
    frames_b: list[np.ndarray],
    times_b: np.ndarray,
    metadata: dict,
) -> dict:
    """分析一档增益并保存曲线、原始帧、实际曝光、图和机器可读报告。"""
    validate_mono12_pair(frames_a, times_a, frames_b, times_b)
    directory.mkdir(parents=True, exist_ok=True)
    metrics_a, result_a = analyze_pair(frames_a, np.asarray(times_a, dtype=np.float64))
    metrics_b, result_b = analyze_pair(frames_b, np.asarray(times_b, dtype=np.float64))
    repeat = curve_difference(result_a.g, result_b.g, metrics_a["coverage"]["hist"])
    held_values = (
        metrics_a["held_even_train_odd_test_ln"],
        metrics_a["held_odd_train_even_test_ln"],
        metrics_b["held_even_train_odd_test_ln"],
        metrics_b["held_odd_train_even_test_ln"],
    )
    held_max = float(np.nanmax(held_values))
    coverage_fraction = float(metrics_a["coverage"]["frac_count_ge_100"])
    quality = {
        "passed": bool(
            np.isfinite(repeat["rmse_measured"])
            and repeat["rmse_measured"] <= 0.10
            and np.isfinite(held_max)
            and held_max <= 0.25
            and coverage_fraction >= 0.05
        ),
        "repeat_rmse_ln": float(repeat["rmse_measured"]),
        "repeat_max_ln": float(repeat["max_measured"]),
        "held_out_max_rmse_ln": held_max,
        "coverage_codes_ge_100_fraction": coverage_fraction,
        "fit_run1_rmse_ln": float(metrics_a["rmse_ln"]),
        "fit_run2_rmse_ln": float(metrics_b["rmse_ln"]),
    }
    np.save(directory / "frames_run1.npy", np.stack(frames_a).astype(np.uint16))
    np.save(directory / "frames_run2.npy", np.stack(frames_b).astype(np.uint16))
    np.save(directory / "times_run1_us.npy", np.asarray(times_a, dtype=np.float64))
    np.save(directory / "times_run2_us.npy", np.asarray(times_b, dtype=np.float64))
    np.save(directory / "g_run1.npy", np.asarray(result_a.g, dtype=np.float64))
    np.save(directory / "g_run2.npy", np.asarray(result_b.g, dtype=np.float64))
    curve = 0.5 * (
        np.asarray(result_a.g, dtype=np.float64).ravel()
        + np.asarray(result_b.g, dtype=np.float64).ravel()
    )
    if curve.size != ZMAX + 1 or np.any(~np.isfinite(curve)):
        raise RuntimeError("响应曲线不是有效的 Mono12 4096 点曲线")
    np.save(directory / "response.npy", curve)
    np.savez(
        directory / "response.npz",
        g=curve,
        g_run1=np.asarray(result_a.g),
        g_run2=np.asarray(result_b.g),
        times_run1_us=np.asarray(times_a),
        times_run2_us=np.asarray(times_b),
        zmin=np.int32(ZMIN),
        zmax=np.int32(ZMAX),
        gain_db=np.float64(metadata["actual_gain_db"]),
        gamma=np.float64(metadata["gamma"]),
    )
    even = np.arange(len(frames_a))[0::2]
    odd = np.arange(len(frames_a))[1::2]
    held_z, held_y = held_out_points(frames_a, np.asarray(times_a), even, odd)
    draw_curve(directory / "response_curve.png", [result_a.g, result_b.g], held_z, held_y)
    contact_sheet(frames_a, np.asarray(times_a), directory / "frames_run1.png")
    report = {
        "camera_serial": str(metadata["camera_serial"]),
        "pixel_format": str(metadata["pixel_format"]),
        "gamma": float(metadata["gamma"]),
        "actual_gain_db": float(metadata["actual_gain_db"]),
        "requested_gain_db": float(metadata["requested_gain_db"]),
        "actual_exposure_run1_us": np.asarray(times_a).tolist(),
        "actual_exposure_run2_us": np.asarray(times_b).tolist(),
        "quality": quality,
        "fit_run1": _json_metrics(metrics_a),
        "fit_run2": _json_metrics(metrics_b),
    }
    (directory / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (directory / "report.txt").write_text(
        "\n".join(
            [
                f"相机序列号: {report['camera_serial']}",
                f"像素格式: {report['pixel_format']}",
                f"Gamma: {report['gamma']}",
                f"实际增益: {report['actual_gain_db']:.6g} dB",
                f"重复性 RMSE: {quality['repeat_rmse_ln']:.6g} ln",
                f"留出验证最大 RMSE: {quality['held_out_max_rmse_ln']:.6g} ln",
                f"覆盖率: {quality['coverage_codes_ge_100_fraction']:.2%}",
                f"质量通过: {quality['passed']}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        **report,
        "paths": {
            "curve": "response.npy",
            "response": "response.npz",
            "frames_run1": "frames_run1.npy",
            "frames_run2": "frames_run2.npy",
            "exposures_run1": "times_run1_us.npy",
            "exposures_run2": "times_run2_us.npy",
            "curve_plot": "response_curve.png",
            "frame_sheet": "frames_run1.png",
            "report": "report.json",
        },
        "curve": curve,
    }


def _json_metrics(metrics: dict) -> dict:
    """去掉大直方图，把 NumPy 标量转成 JSON 可写的质量摘要。"""
    return {
        "rmse_ln": float(metrics["rmse_ln"]),
        "held_even_train_odd_test_ln": float(metrics["held_even_train_odd_test_ln"]),
        "held_odd_train_even_test_ln": float(metrics["held_odd_train_even_test_ln"]),
        "n_used": int(metrics["n_used"]),
        "n_required": int(metrics["n_required"]),
        "coverage": {
            key: (float(value) if isinstance(value, (float, np.floating)) else int(value))
            for key, value in metrics["coverage"].items()
            if key != "hist"
        },
    }


def draw_curve(
    path: Path,
    g_runs: list[np.ndarray],
    held_z: np.ndarray,
    held_y: np.ndarray,
) -> None:
    width, height = 1200, 760
    canvas = np.full((height, width, 3), 255, np.uint8)
    left, right, top, bottom = 90, 30, 30, 70
    curves = [np.asarray(g, dtype=np.float64).ravel() for g in g_runs]
    y_values = [curve for curve in curves]
    if held_y.size:
        y_values.append(held_y)
    y_min = min(float(np.min(values)) for values in y_values) - 0.3
    y_max = max(float(np.max(values)) for values in y_values) + 0.3

    def x_of(code: float) -> int:
        return int(left + (width - left - right) * (code / ZMAX))

    def y_of(value: float) -> int:
        return int(top + (height - top - bottom) * ((y_max - value) / (y_max - y_min)))

    for tick in range(0, ZMAX + 1, 512):
        x = x_of(tick)
        cv2.line(canvas, (x, top), (x, height - bottom), (230, 230, 230), 1)
        cv2.putText(canvas, str(tick), (x - 18, height - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (40, 40, 40), 1)
    for step in range(int(np.floor(y_min)), int(np.ceil(y_max)) + 1):
        y = y_of(step)
        cv2.line(canvas, (left, y), (width - right, y), (230, 230, 230), 1)
        cv2.putText(canvas, str(step), (20, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (40, 40, 40), 1)
    if held_z.size:
        take = np.linspace(0, held_z.size - 1, min(2500, held_z.size)).astype(np.int32)
        for code, value in zip(held_z[take], held_y[take]):
            cv2.circle(canvas, (x_of(float(code)), y_of(float(value))), 1, (180, 180, 180), -1)
    colors = [(180, 80, 20), (40, 90, 200)]
    for curve, color in zip(curves, colors):
        pts = np.array([[x_of(i), y_of(float(v))] for i, v in enumerate(curve)], dtype=np.int32)
        cv2.polylines(canvas, [pts], False, color, 2, cv2.LINE_AA)
    cv2.putText(canvas, "Z", (width // 2, height - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    cv2.putText(canvas, "g(Z)", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    cv2.putText(canvas, "run1", (width - 220, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, colors[0], 2)
    cv2.putText(canvas, "run2", (width - 220, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55, colors[1], 2)
    cv2.putText(canvas, "ln E + ln t", (width - 280, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1)
    cv2.imwrite(str(path), canvas)


def held_out_points(images, times, train_idx, test_idx) -> tuple[np.ndarray, np.ndarray]:
    train_images = [images[int(i)] for i in train_idx]
    result = recover(train_images, times[train_idx], lam=LAM, n_pixels=N_PIXELS)
    ln_e = np.log(np.maximum(result.radiance.astype(np.float64), 1e-30))
    # 重新按空间均匀抽一点，避免只看拟合样本。
    height, width = ln_e.shape
    ys = np.linspace(8, height - 9, 18).astype(np.int32)
    xs = np.linspace(8, width - 9, 22).astype(np.int32)
    grid_y, grid_x = np.meshgrid(ys, xs, indexing="ij")
    codes = []
    values = []
    test_images = [images[int(i)] for i in test_idx]
    test_times = times[test_idx]
    for image, exposure in zip(test_images, test_times):
        z = image[grid_y, grid_x].astype(np.float64)
        y = ln_e[grid_y, grid_x] + np.log(float(exposure))
        keep = (z > 0) & (z < ZMAX)
        codes.append(z[keep])
        values.append(y[keep])
    if not codes:
        return np.array([]), np.array([])
    return np.concatenate(codes), np.concatenate(values)


def contact_sheet(images: list[np.ndarray], times: np.ndarray, path: Path) -> None:
    thumbs = []
    for image, exposure in zip(images, times):
        view = np.clip(image.astype(np.float32) * (255.0 / ZMAX), 0, 255).astype(np.uint8)
        thumb = cv2.resize(view, (180, 136), interpolation=cv2.INTER_AREA)
        tile = cv2.cvtColor(thumb, cv2.COLOR_GRAY2BGR)
        cv2.putText(
            tile,
            f"{exposure / 1000:.3g} ms",
            (6, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 255),
            1,
        )
        thumbs.append(tile)
    sheet = np.hstack(thumbs)
    cv2.imwrite(str(path), sheet)


def self_check() -> None:
    rng = np.random.default_rng(0)
    height, width = 96, 128
    y, x = np.mgrid[0:height, 0:width]
    radiance = np.exp((x / width) * 5.5 - 2.5) * (0.4 + 0.6 * y / height)
    radiance *= 1.0 + 0.01 * rng.normal(size=radiance.shape)
    t_ref = 64.0
    gamma = 0.45
    times = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256], dtype=np.float64)
    images = []
    for exposure in times:
        level = np.clip(radiance * exposure / t_ref, 0, 1) ** gamma
        images.append(np.rint(ZMAX * level).astype(np.uint16))
    result = recover(images, times, lam=LAM, n_pixels=1500)
    metrics = fit_metrics(images, times, result)
    if not np.isfinite(metrics["rmse_ln"]) or metrics["rmse_ln"] > 0.08:
        raise RuntimeError(f"合成数据自检未通过，拟合 RMSE = {metrics['rmse_ln']}")
    print(f"自检通过，合成数据拟合 RMSE = {metrics['rmse_ln']:.4f} ln", flush=True)


def capture_two(camera: VimbaCamera) -> tuple[list[np.ndarray], np.ndarray, list[np.ndarray], np.ndarray, dict]:
    meta: dict = {}
    latest = {"n": 0, "frame": None}
    lock = threading.Lock()
    stop = threading.Event()

    def on_frame(image) -> None:
        frame = squeeze(image)
        with lock:
            latest["n"] += 1
            latest["frame"] = frame.copy()

    def on_live() -> None:
        locked = (
            ("自动曝光", lambda: camera.set_exposure_auto_live("Off")),
            ("自动增益", lambda: camera.set_gain_auto_live("Off")),
            ("Gamma", lambda: camera.set_gamma_live(1.0)),
            ("增益", lambda: camera.set_gain_live(0.0)),
        )
        for label, action in locked:
            try:
                action()
            except Exception as exc:
                meta[f"{label}_error"] = str(exc)
                print(f"{label} 未能锁定: {exc}", flush=True)
        meta["gain"] = _try_read(camera.get_gain_live)
        meta["gamma"] = _try_read(camera.get_gamma_live)
        meta["exposure_auto"] = _try_read(camera.get_exposure_auto_live)
        meta["gain_auto"] = _try_read(camera.get_gain_auto_live)
        try:
            name = camera._probe_exposure_feature_name()
            feature = getattr(camera._camera, name)
            low, high = feature.get_range()
            meta["exposure_range_us"] = (float(low), float(high))
        except Exception as exc:
            meta["exposure_range_us"] = None
            meta["exposure_range_error"] = str(exc)

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
            if ready and "gain" in meta:
                break
            time.sleep(0.02)
        else:
            raise RuntimeError("预览帧没有到达")

        def one_ladder() -> tuple[list[np.ndarray], np.ndarray]:
            kept: list[np.ndarray] = []
            actual = []
            for exposure in REQUESTED_US:
                low, high = meta.get("exposure_range_us") or (0.0, 1.0e12)
                use = float(np.clip(exposure, low, high))
                camera.set_exposure_time_live(use)
                with lock:
                    start = latest["n"]
                timeout = max(6.0, (use / 1e6) * 8.0 + 2.0)
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
                kept.append(frame)
                actual.append(float(camera.get_exposure_time_live()))
                print(
                    f"  曝光请求 {exposure:.0f} μs，读回 {actual[-1]:.1f} μs，"
                    f"中位码值 {np.median(frame):.0f}，最大 {np.max(frame)}",
                    flush=True,
                )
            return kept, np.asarray(actual, dtype=np.float64)

        print("第 1 轮", flush=True)
        frames_a, times_a = one_ladder()
        print("第 2 轮", flush=True)
        frames_b, times_b = one_ladder()
        return frames_a, times_a, frames_b, times_b, meta
    finally:
        stop.set()
        thread.join(timeout=8)


def rel_text(rmse_ln: float) -> str:
    return f"{rmse_ln:.4f} ln（约 {np.expm1(rmse_ln) * 100:.2f}% ，{rmse_ln / np.log(2):.3f} 档）"


def write_report(path: Path, meta: dict, times_a, times_b, metrics_a, metrics_b, repeat) -> None:
    cover = metrics_a["coverage"]
    lines = [
        "Debevec & Malik 1997 响应曲线实测",
        "",
        f"相机 Manta G-033B，序列号 {SERIAL}，Mono12，码值 0–4095。",
        f"增益 {meta.get('gain')} dB，Gamma {meta.get('gamma')}，"
        f"自动曝光 {meta.get('exposure_auto')}，自动增益 {meta.get('gain_auto')}。",
        f"曝光范围 {meta.get('exposure_range_us')} μs。",
        f"λ = {LAM}。12 位平滑系数按码值个数放大，实际系数是 λ × 4096/256 = {LAM * 4096 / 256:.0f} × w(Z)。",
        f"每轮 {len(REQUESTED_US)} 档，相邻档曝光比 2。采样 {metrics_a['n_used']} 个像素，"
        f"论文下界是 {metrics_a['n_required']}。",
        "",
        "第 1 轮实际曝光（μs）：" + ", ".join(f"{t:.1f}" for t in times_a),
        "第 2 轮实际曝光（μs）：" + ", ".join(f"{t:.1f}" for t in times_b),
        "",
        "式 (2) 拟合残差，用参与融合的全部像素、按帽子权重：",
        f"  第 1 轮 {rel_text(metrics_a['rmse_ln'])}",
        f"  第 2 轮 {rel_text(metrics_b['rmse_ln'])}",
        f"  第 1 轮单像素残差中位 {metrics_a['pixel_rms_median']:.4f} ln，"
        f"90 分位 {metrics_a['pixel_rms_p90']:.4f} ln",
        "",
        "留出一半曝光再预测（同一轮里奇数档、偶数档互测）：",
        f"  用偶数档拟合，预测奇数档 {rel_text(metrics_a['held_even_train_odd_test_ln'])}",
        f"  用奇数档拟合，预测偶数档 {rel_text(metrics_a['held_odd_train_even_test_ln'])}",
        "",
        "两轮 g(Z) 之差，只统计至少出现 100 次的码值：",
        f"  RMSE {repeat['rmse_measured']:.4f} ln，最大偏差 {repeat['max_measured']:.4f} ln",
        f"  整条曲线 RMSE {repeat['rmse_all']:.4f} ln（含两端外推）",
        "",
        "把采样数收到论文下界再解一次，和第 1 轮完整采样相比：",
        f"  有数据的码值上 RMSE {metrics_a['sample_stability']['rmse_measured']:.4f} ln，"
        f"最大偏差 {metrics_a['sample_stability']['max_measured']:.4f} ln",
        "",
        "码值覆盖（第 1 轮，11 张合在一起）：",
        f"  出现过的码值 {cover['distinct']} / 4096",
        f"  1 到 4094 里至少出现 100 次的比例 {cover['frac_count_ge_100'] * 100:.1f}%",
        f"  中间最长的空档 {cover['max_gap']} 个码值",
        f"  同一像素在各档之间的码值跨度：中位 {cover['median_span']:.0f}，"
        f"跨度 ≥1000 的像素 {cover['pixels_span_1000']}，"
        f"≥2000 的 {cover['pixels_span_2000']}，≥3000 的 {cover['pixels_span_3000']}",
        "",
        "第 1 轮各档：",
    ]
    for exposure, row in zip(times_a, metrics_a["per_stop"]):
        lines.append(
            f"  {exposure:.0f} μs  中位码值 {row['median_code']:.0f}  "
            f"99 分位 {row['p99_code']:.0f}  "
            f"近黑 {row['frac_black'] * 100:.1f}%  "
            f"饱和 {row['frac_saturated'] * 100:.1f}%  "
            f"该档残差 {row['rmse_ln']:.4f} ln"
        )
    lines.extend(
        [
            "",
            "g(Z) 存在 g_run1.npy、g_run2.npy。原始帧在 frames_run1.npy、frames_run2.npy。",
            "0 和 4095 的帽子权重是 0，两端不由数据决定。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    self_check()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    camera = VimbaCamera()
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
            raise RuntimeError(f"没有找到序列号 {SERIAL}")
        camera.open_device(index)
        camera.setup_camera()
        frames_a, times_a, frames_b, times_b, meta = capture_two(camera)
    finally:
        camera.close()

    for run, frames in ((1, frames_a), (2, frames_b)):
        peak = max(int(np.max(frame)) for frame in frames)
        if peak > ZMAX:
            raise RuntimeError(f"第 {run} 轮最大码值 {peak}，不是 Mono12")
        if frames[0].dtype.kind not in "ui":
            raise RuntimeError(f"第 {run} 轮像素类型 {frames[0].dtype}")

    print("求解第 1 轮", flush=True)
    metrics_a, result_a = analyze_pair(frames_a, times_a)
    print("求解第 2 轮", flush=True)
    metrics_b, result_b = analyze_pair(frames_b, times_b)
    repeat = curve_difference(result_a.g, result_b.g, metrics_a["coverage"]["hist"])

    np.save(OUT_DIR / "frames_run1.npy", np.stack(frames_a).astype(np.uint16))
    np.save(OUT_DIR / "frames_run2.npy", np.stack(frames_b).astype(np.uint16))
    np.save(OUT_DIR / "times_run1_us.npy", times_a)
    np.save(OUT_DIR / "times_run2_us.npy", times_b)
    np.save(OUT_DIR / "g_run1.npy", np.asarray(result_a.g))
    np.save(OUT_DIR / "g_run2.npy", np.asarray(result_b.g))
    np.savez(
        OUT_DIR / "response.npz",
        g_run1=np.asarray(result_a.g),
        g_run2=np.asarray(result_b.g),
        times_run1_us=times_a,
        times_run2_us=times_b,
        lam=LAM,
        zmin=ZMIN,
        zmax=ZMAX,
    )
    contact_sheet(frames_a, times_a, OUT_DIR / "bracket_run1.png")
    even = np.arange(len(frames_a))[0::2]
    odd = np.arange(len(frames_a))[1::2]
    held_z, held_y = held_out_points(frames_a, times_a, even, odd)
    draw_curve(OUT_DIR / "response_curve.png", [result_a.g, result_b.g], held_z, held_y)
    write_report(
        OUT_DIR / "report.txt",
        meta,
        times_a,
        times_b,
        metrics_a,
        metrics_b,
        repeat,
    )
    print(f"结果写在 {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
