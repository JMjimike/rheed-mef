#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Debevec 与 Malik, SIGGRAPH 1997, 辐射图恢复。

实现正文式 (4) 的帽子权重、式 (3) 的加权最小二乘（附录 gsolve，
右端按正文取 w * ln t_j），以及式 (6) 的全图融合。
码值范围按图像位深取：Mono8 为 0..255，Mono12 为 0..4095。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import uniform_filter
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import spsolve


def hat_weights(zmin: int, zmax: int) -> np.ndarray:
    """式 (4)。返回长度 zmax - zmin + 1 的权重，下标 0 对应 zmin。"""
    if zmax <= zmin:
        raise ValueError(f"非法码值范围 [{zmin}, {zmax}]")
    z = np.arange(zmin, zmax + 1, dtype=np.float64)
    mid = 0.5 * (zmin + zmax)
    return np.where(z <= mid, z - zmin, zmax - z).astype(np.float64)


def zmid_of(zmin: int, zmax: int) -> int:
    """g(Zmid) = 0 的码值。与附录一致，取中间码（.5 向偶数取整）。"""
    return int(np.round(0.5 * (zmin + zmax)))


def minimum_sample_count(zmin: int, zmax: int, n_exposures: int) -> int:
    """论文条件 N(P - 1) > Zmax - Zmin 的最小整数 N。"""
    span = zmax - zmin
    denom = max(int(n_exposures) - 1, 1)
    return span // denom + 1


# 详细标定相对 ln(Z/Zmid) 的均方根差不超过这个值，就当作线性。
# 这台 Manta 在增益 0 dB、gamma 1 时大约是 0.036 ln，两轮重复测量大约 0.017 ln。
LINEAR_RMSE_MAX = 0.05
CALIBRATION_LAM = 10.0
CALIBRATION_PIXELS = 4000


def linear_response(zmin: int, zmax: int) -> np.ndarray:
    """g(Z) = ln(Z / Zmid)，g(Zmid) = 0。码值 0 沿用码值 1，避免对数发散。"""
    zmid = zmid_of(zmin, zmax)
    anchor = max(int(zmid), int(zmin) + 1)
    codes = np.arange(int(zmin), int(zmax) + 1, dtype=np.float64)
    positive = np.maximum(codes, float(zmin) + 1.0)
    return np.log(positive / float(anchor))


def code_counts(images: list[np.ndarray], zmin: int, zmax: int) -> np.ndarray:
    """各码值在整组标定图里出现的次数。下标 0 对应 zmin。"""
    counts = np.zeros(int(zmax) - int(zmin) + 1, dtype=np.int64)
    for image in images:
        data = np.asarray(image)
        if data.ndim == 3:
            data = data[..., 0]
        codes = np.clip(data.ravel(), zmin, zmax).astype(np.int64, copy=False)
        counts += np.bincount(codes - int(zmin), minlength=counts.size)
    return counts


def response_linearity(
    curve: np.ndarray,
    zmin: int,
    counts: np.ndarray,
    min_count: int = 100,
) -> dict:
    """测得的 g 相对直线的均方根差。两端和出现次数太少的码值不参加。"""
    measured = np.asarray(curve, dtype=np.float64).ravel()
    zmax = int(zmin) + int(measured.size) - 1
    ideal = linear_response(zmin, zmax)
    seen = np.asarray(counts, dtype=np.int64).ravel()
    if seen.size != measured.size:
        raise ValueError("码值计数长度与响应曲线不一致")
    index = np.arange(measured.size)
    usable = (seen >= int(min_count)) & (index > 8) & (index < measured.size - 1)
    n_codes = int(np.count_nonzero(usable))
    needed = max(32, int(measured.size) // 8)
    if n_codes < needed:
        return {"rmse": float("inf"), "n_codes": n_codes, "usable": False}
    delta = measured[usable] - ideal[usable]
    return {
        "rmse": float(np.sqrt(np.mean(delta * delta))),
        "n_codes": n_codes,
        "usable": True,
    }


def choose_response(
    measured: np.ndarray,
    zmin: int,
    counts: np.ndarray,
) -> tuple[np.ndarray, bool, float]:
    """接近线性就改用直线，否则保留测得的 g。返回曲线、是否线性、RMSE。"""
    stats = response_linearity(measured, zmin, counts)
    measured = np.asarray(measured, dtype=np.float64).ravel()
    if stats["usable"] and stats["rmse"] <= LINEAR_RMSE_MAX:
        return linear_response(zmin, zmin + measured.size - 1), True, float(stats["rmse"])
    rmse = float(stats["rmse"])
    return measured, False, rmse


def infer_code_range(images: list[np.ndarray]) -> tuple[int, int]:
    """Mono8 -> 255；像素值不超过 4095 的更高位深按 Mono12；否则按实际峰值。"""
    if not images:
        raise ValueError("没有图像，无法判断码值范围")
    sample = images[0]
    if not np.issubdtype(np.asarray(sample).dtype, np.integer):
        raise ValueError("响应标定只接受整数像素码值")
    if sample.dtype == np.uint8:
        return 0, 255
    peak = 0
    for image in images:
        data = np.asarray(image)
        if data.size == 0 or int(np.min(data)) < 0:
            raise ValueError("图像为空或含负码值")
        peak = max(peak, int(np.max(data)))
    if peak <= 4095:
        return 0, 4095
    if np.issubdtype(sample.dtype, np.integer):
        return 0, int(np.iinfo(sample.dtype).max)
    return 0, peak


def _as_code_image(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if np.issubdtype(arr.dtype, np.floating):
        if not np.all(np.isfinite(arr)):
            raise ValueError("图像含有非有限值")
        return np.rint(arr).astype(np.int32)
    if not np.issubdtype(arr.dtype, np.integer):
        raise ValueError(f"无法把 dtype={arr.dtype} 当作像素码值")
    return arr


def _local_variance(image: np.ndarray) -> np.ndarray:
    values = image.astype(np.float64)
    mean = uniform_filter(values, size=5, mode="reflect")
    mean_sq = uniform_filter(values * values, size=5, mode="reflect")
    var = mean_sq - mean * mean
    return np.maximum(var, 0.0)


def sample_pixels(
    images: list[np.ndarray],
    zmin: int,
    zmax: int,
    n_pixels: int,
) -> tuple[np.ndarray, dict]:
    """按码值均匀、空间上散开地采样，并避开局部方差高的像素。

    返回 Z，形状 (N, P)，以及采样信息。N 至少抬到论文给出的下界；
    图像里不够时用全部合格像素，并在信息里标记 short。
    """
    if len(images) < 2:
        raise ValueError("至少需要两张曝光")
    stack = np.stack(
        [np.clip(np.asarray(im, dtype=np.int32), zmin, zmax) for im in images],
        axis=0,
    )
    n_exp, height, width = stack.shape
    weights = hat_weights(zmin, zmax)[stack - zmin]
    best_exposure = np.argmax(weights, axis=0)
    z_star = np.take_along_axis(stack, best_exposure[None, ...], axis=0)[0]
    w_star = np.take_along_axis(weights, best_exposure[None, ...], axis=0)[0]

    variance = _local_variance(z_star)
    variance[:2, :] = np.inf
    variance[-2:, :] = np.inf
    variance[:, :2] = np.inf
    variance[:, -2:] = np.inf

    finite = variance[np.isfinite(variance)]
    if finite.size == 0:
        threshold = np.inf
    else:
        threshold = float(np.percentile(finite, 60))
    mask = (w_star > 0) & (variance <= threshold)
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        ys, xs = np.nonzero(w_star > 0)
    if ys.size == 0:
        raise ValueError("没有可用像素：各曝光的权重都是 0")

    required = minimum_sample_count(zmin, zmax, n_exp)
    target = max(int(n_pixels), required)
    target = min(target, int(ys.size))

    zvals = z_star[ys, xs]
    order = np.argsort(zvals, kind="mergesort")
    ranks = np.linspace(0, order.size - 1, target)
    ranks = np.unique(np.rint(ranks).astype(np.int32))
    used: set[int] = set()
    chosen: list[int] = []
    grid = max(int(np.ceil(np.sqrt(target))), 1)
    for i, rank in enumerate(ranks):
        target_y = ((i % grid) + 0.5) / grid * height
        target_x = ((i // grid) + 0.5) / grid * width
        lo = max(0, int(rank) - 40)
        hi = min(order.size, int(rank) + 41)
        best = -1
        best_dist = None
        for pos in range(lo, hi):
            idx = int(order[pos])
            if idx in used:
                continue
            dy = float(ys[idx]) - target_y
            dx = float(xs[idx]) - target_x
            dist = dy * dy + dx * dx
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best = idx
        if best < 0 or best in used:
            for idx in order:
                idx = int(idx)
                if idx not in used:
                    best = idx
                    break
        if best < 0 or best in used:
            continue
        used.add(best)
        chosen.append(best)

    if not chosen:
        raise ValueError("采样失败")
    chosen_arr = np.asarray(chosen, dtype=np.int32)
    yy = ys[chosen_arr]
    xx = xs[chosen_arr]
    samples = np.ascontiguousarray(stack[:, yy, xx].T)
    info = {
        "n_requested": int(n_pixels),
        "n_required": int(required),
        "n_used": int(samples.shape[0]),
        "raised": int(samples.shape[0]) > int(n_pixels),
        "short": int(samples.shape[0]) < int(required),
    }
    return samples, info


def gsolve(
    samples: np.ndarray,
    times: np.ndarray,
    lam: float = 10.0,
    zmin: int = 0,
    zmax: int = 255,
) -> tuple[np.ndarray, np.ndarray, int]:
    """求解 g(Z) 与采样像素的 ln E。

    samples 形状 (N, P)。返回 g（下标 0 对应 zmin）、ln E、以及 2（正规方程已解）。
    平滑项与附录一样乘在二阶差分上，不是 sqrt(λ)。码值变密以后，
    同一条对数曲线的离散二阶差分按约 1/z^2 变小，因此实际系数是
    λ * (码值个数 / 256) * w(z)。这样界面上的 λ=10 与 8-bit 附录处在同一工作点。
    g(Zmid) = 0 用大权重写入，把尺度固定下来。
    未知量按列归一化后解正规方程，避免 12-bit 时稠密 SVD 的开销。
    """
    codes = np.asarray(samples, dtype=np.int32)
    exposure = np.asarray(times, dtype=np.float64)
    if codes.ndim != 2:
        raise ValueError("samples 必须是 (N, P)")
    if exposure.shape != (codes.shape[1],):
        raise ValueError("曝光时间数量与样本列数不一致")
    if np.any(~np.isfinite(exposure)) or np.any(exposure <= 0):
        raise ValueError("曝光时间必须为有限正数")
    if lam < 0:
        raise ValueError("λ 不能为负")

    n_pix, n_exp = codes.shape
    n_codes = zmax - zmin + 1
    wtab = hat_weights(zmin, zmax)
    clipped = np.clip(codes, zmin, zmax)
    wij = wtab[clipped - zmin]
    valid = wij > 0
    pix_i, exp_j = np.nonzero(valid)
    n_data = int(pix_i.size)
    if n_data < n_pix:
        raise ValueError("有效样本方程少于像素数，无法求解响应曲线")

    levels = np.arange(zmin + 1, zmax)
    smooth_w = wtab[levels - zmin]
    keep = smooth_w > 0
    levels = levels[keep]
    smooth_w = smooth_w[keep]
    n_smooth = int(levels.size)

    n_eq = n_data + 1 + n_smooth
    n_unknown = n_codes + n_pix
    n_entries = n_data * 2 + 1 + n_smooth * 3
    rows = np.empty(n_entries, dtype=np.int32)
    cols = np.empty(n_entries, dtype=np.int32)
    vals = np.empty(n_entries, dtype=np.float64)
    rhs = np.zeros(n_eq, dtype=np.float64)

    eq = np.arange(n_data, dtype=np.int32)
    weight = wij[pix_i, exp_j]
    rows[:n_data] = eq
    cols[:n_data] = clipped[pix_i, exp_j] - zmin
    vals[:n_data] = weight
    rows[n_data : 2 * n_data] = eq
    cols[n_data : 2 * n_data] = n_codes + pix_i
    vals[n_data : 2 * n_data] = -weight
    rhs[:n_data] = weight * np.log(exposure[exp_j])

    cursor = 2 * n_data
    anchor = n_data
    rows[cursor] = anchor
    cols[cursor] = zmid_of(zmin, zmax) - zmin
    vals[cursor] = 1.0e6
    cursor += 1

    # 256 是附录里 8-bit 的码值个数。见 gsolve 的说明。
    coef = float(lam) * (n_codes / 256.0) * smooth_w
    smooth_eq = anchor + 1 + np.arange(n_smooth, dtype=np.int32)
    center = levels - zmin
    rows[cursor : cursor + n_smooth] = smooth_eq
    cols[cursor : cursor + n_smooth] = center - 1
    vals[cursor : cursor + n_smooth] = coef
    cursor += n_smooth
    rows[cursor : cursor + n_smooth] = smooth_eq
    cols[cursor : cursor + n_smooth] = center
    vals[cursor : cursor + n_smooth] = -2.0 * coef
    cursor += n_smooth
    rows[cursor : cursor + n_smooth] = smooth_eq
    cols[cursor : cursor + n_smooth] = center + 1
    vals[cursor : cursor + n_smooth] = coef
    cursor += n_smooth
    if cursor != n_entries:
        raise RuntimeError("线性系统装配长度不一致")

    matrix = coo_matrix((vals, (rows, cols)), shape=(n_eq, n_unknown)).tocsc()
    column_norm = np.sqrt(np.asarray(matrix.multiply(matrix).sum(axis=0))).ravel()
    column_norm[column_norm == 0] = 1.0
    scaled = matrix @ diags(1.0 / column_norm)
    normal = (scaled.T @ scaled).tocsc()
    solution = spsolve(normal, scaled.T @ rhs) / column_norm
    solution = np.asarray(solution, dtype=np.float64).ravel()
    if not np.all(np.isfinite(solution)):
        raise RuntimeError("响应曲线最小二乘没有得到有限解")
    response = np.asarray(solution[:n_codes], dtype=np.float64)
    log_irradiance = np.asarray(solution[n_codes:], dtype=np.float64)
    return response, log_irradiance, 2


def merge_radiance(
    images: list[np.ndarray],
    times: np.ndarray,
    response: np.ndarray,
    zmin: int,
    zmax: int,
) -> np.ndarray:
    """式 (6)。权重全为 0 的像素改用最接近中灰的那一档。"""
    exposure = np.asarray(times, dtype=np.float64)
    if exposure.ndim != 1 or exposure.size != len(images):
        raise ValueError("曝光时间数量与图像数量不一致")
    if np.any(~np.isfinite(exposure)) or np.any(exposure <= 0):
        raise ValueError("曝光时间必须为有限正数")
    curve = np.asarray(response, dtype=np.float64)
    if curve.shape != (int(zmax) - int(zmin) + 1,) or np.any(~np.isfinite(curve)):
        raise ValueError("响应曲线长度、维度或有限性与码值范围不一致")
    log_t = np.log(exposure)
    wtab = hat_weights(zmin, zmax)
    shape = images[0].shape
    numerator = np.zeros(shape, dtype=np.float64)
    denominator = np.zeros(shape, dtype=np.float64)
    closest = None
    closest_dist = None
    mid = 0.5 * (zmin + zmax)
    for image, lt in zip(images, log_t):
        codes = np.clip(np.asarray(image, dtype=np.int32), zmin, zmax)
        weight = wtab[codes - zmin]
        predicted = curve[codes - zmin] - lt
        numerator += weight * predicted
        denominator += weight
        distance = np.abs(codes.astype(np.float64) - mid)
        if closest is None:
            closest = predicted
            closest_dist = distance
        else:
            take = distance < closest_dist
            closest = np.where(take, predicted, closest)
            closest_dist = np.where(take, distance, closest_dist)

    log_e = np.empty(shape, dtype=np.float64)
    good = denominator > 0
    log_e[good] = numerator[good] / denominator[good]
    log_e[~good] = closest[~good]
    radiance = np.exp(log_e).astype(np.float32)
    radiance[~np.isfinite(radiance)] = 0
    return radiance


@dataclass
class DebevecResult:
    radiance: np.ndarray
    g: np.ndarray
    zmin: int
    zmax: int
    times: np.ndarray
    lam: float
    n_requested: int
    n_required: int
    n_used: int
    raised: bool
    short: bool
    istop: int


def recover(
    images: list[np.ndarray],
    times: np.ndarray,
    lam: float = 10.0,
    n_pixels: int = 200,
    zmin: int | None = None,
    zmax: int | None = None,
) -> DebevecResult:
    """由一组已知曝光时间的图恢复响应曲线和相对辐照度。

    单通道返回 g 形状 (n,)，三通道逐通道求解，g 形状 (3, n)。
    times 与图像一一对应，单位只要整组一致即可（程序里用微秒）。
    """
    if len(images) < 2:
        raise ValueError("至少需要两张不同曝光的图像")
    frames = [_as_code_image(im) for im in images]
    shape = frames[0].shape
    for frame in frames[1:]:
        if frame.shape != shape:
            raise ValueError("各曝光图像尺寸不一致")
    exposure = np.asarray(times, dtype=np.float64).reshape(-1)
    if exposure.shape != (len(frames),):
        raise ValueError("曝光时间数量与图像数量不一致")
    if np.any(~np.isfinite(exposure)) or np.any(exposure <= 0):
        raise ValueError("曝光时间必须为有限正数")
    if (zmin is None) != (zmax is None):
        raise ValueError("zmin 和 zmax 必须同时提供")
    if zmin is not None and int(zmax) <= int(zmin):
        raise ValueError("非法码值范围")

    if frames[0].ndim == 2:
        channel_groups = [frames]
    elif frames[0].ndim == 3 and frames[0].shape[2] in (1, 3):
        channel_groups = [
            [frame[..., channel] for frame in frames]
            for channel in range(frames[0].shape[2])
        ]
    else:
        raise ValueError(f"不支持的图像形状 {shape}")

    curves = []
    planes = []
    sample_info = None
    istop = 0
    used_zmin = 0
    used_zmax = 0
    for group in channel_groups:
        if zmin is None or zmax is None:
            used_zmin, used_zmax = infer_code_range(group)
        else:
            used_zmin, used_zmax = int(zmin), int(zmax)
        samples, sample_info = sample_pixels(group, used_zmin, used_zmax, n_pixels)
        curve, _log_e, istop = gsolve(samples, exposure, lam, used_zmin, used_zmax)
        curves.append(curve)
        planes.append(merge_radiance(group, exposure, curve, used_zmin, used_zmax))

    if len(planes) == 1 and frames[0].ndim == 2:
        radiance = planes[0]
        response = curves[0]
    elif len(planes) == 1:
        radiance = planes[0][..., None]
        response = curves[0]
    else:
        radiance = np.stack(planes, axis=-1)
        response = np.stack(curves, axis=0)

    assert sample_info is not None
    return DebevecResult(
        radiance=radiance,
        g=response,
        zmin=used_zmin,
        zmax=used_zmax,
        times=exposure.copy(),
        lam=float(lam),
        n_requested=int(sample_info["n_requested"]),
        n_required=int(sample_info["n_required"]),
        n_used=int(sample_info["n_used"]),
        raised=bool(sample_info["raised"]),
        short=bool(sample_info["short"]),
        istop=istop,
    )


def merge_with_response(
    images: list[np.ndarray],
    times: np.ndarray,
    curve: np.ndarray,
    zmin: int,
    zmax: int,
) -> DebevecResult:
    """用已经定好的 g 做式 (6)。不再从这一组曝光重解响应曲线。"""
    if len(images) < 2:
        raise ValueError("至少需要两张不同曝光的图像")
    frames = [_as_code_image(im) for im in images]
    shape = frames[0].shape
    for frame in frames[1:]:
        if frame.shape != shape:
            raise ValueError("各曝光图像尺寸不一致")
    exposure = np.asarray(times, dtype=np.float64).reshape(-1)
    if exposure.shape != (len(frames),):
        raise ValueError("曝光时间数量与图像数量不一致")
    if np.any(~np.isfinite(exposure)) or np.any(exposure <= 0):
        raise ValueError("曝光时间必须为有限正数")
    response = np.asarray(curve, dtype=np.float64)
    if response.ndim == 2:
        response = response[0]
    if frames[0].ndim == 3 and frames[0].shape[-1] == 1:
        frames = [frame[..., 0] for frame in frames]
    if frames[0].ndim != 2:
        raise ValueError(f"不支持的图像形状 {shape}")
    if (
        response.ndim != 1
        or response.size != int(zmax) - int(zmin) + 1
        or np.any(~np.isfinite(response))
    ):
        raise ValueError("响应曲线长度、维度或有限性与码值范围不一致")
    radiance = merge_radiance(frames, exposure, response, int(zmin), int(zmax))
    return DebevecResult(
        radiance=radiance,
        g=response,
        zmin=int(zmin),
        zmax=int(zmax),
        times=exposure.copy(),
        lam=0.0,
        n_requested=0,
        n_required=0,
        n_used=0,
        raised=False,
        short=False,
        istop=0,
    )


def radiance_log(radiance: np.ndarray) -> np.ndarray:
    """相对辐照度的对数。彩色图先对通道取平均，只用于显示。"""
    values = np.asarray(radiance, dtype=np.float64)
    if values.ndim == 3:
        values = np.mean(values, axis=2)
    return np.log(np.maximum(values, 1e-20))


def frame_log_irradiance(
    image: np.ndarray,
    exposure_us: float,
    curve: np.ndarray,
    zmin: int = 0,
) -> np.ndarray:
    """单档的对数辐照度 g(Z) − ln t，和式 (6) 里每一档的项相同。"""
    data = np.asarray(image)
    if data.ndim == 3 and data.shape[-1] == 1:
        data = data[..., 0]
    if data.ndim == 3:
        data = np.mean(data, axis=2)
    response = np.asarray(curve, dtype=np.float64).ravel()
    hi = zmin + response.size - 1
    codes = np.clip(np.rint(data), zmin, hi).astype(np.int32)
    return response[codes - zmin] - np.log(max(float(exposure_us), 1e-6))


def log_limits(
    log_v: np.ndarray,
    low_percentile: float,
    high_percentile: float,
) -> tuple[float, float]:
    finite = np.isfinite(log_v)
    if not np.any(finite):
        return 0.0, 1.0
    lo, hi = np.percentile(log_v[finite], [low_percentile, high_percentile])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        hi = float(lo) + 1.0
    return float(lo), float(hi)


def paint_log(log_v: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """把对数辐照度按给定窗口画成 8-bit。窗口外的值贴到黑或白。"""
    if hi <= lo:
        hi = lo + 1.0
    scaled = (np.asarray(log_v, dtype=np.float64) - lo) / (hi - lo)
    scaled = np.where(np.isfinite(scaled), scaled, 0.0)
    return np.rint(np.clip(scaled, 0.0, 1.0) * 255.0).astype(np.uint8)


def log_display(
    radiance: np.ndarray,
    low_percentile: float = 1.0,
    high_percentile: float = 99.5,
) -> np.ndarray:
    """把相对辐照度变成 8-bit 对数显示图。百分位只影响显示。"""
    log_v = radiance_log(radiance)
    lo, hi = log_limits(log_v, low_percentile, high_percentile)
    return paint_log(log_v, lo, hi)
