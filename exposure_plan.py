#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""按饱和比例自动排曝光，短曝光更密。

最长一档停在饱和不超过上限的地方，默认 10%。斑点先顶满、整幅还远没到
10% 时，用最亮约 0.01% 像素决定斑点档，再往短曝光加两档。斑点档以上
按 4 倍拉开，只补暗部，不再堆高曝光。
"""

from __future__ import annotations

import numpy as np

SAT_LIMIT = 0.10
# 写给相机的曝光硬上限。界面上的「最长曝光」不能超过这一档。
MAX_EXPOSURE_US = 5_000_000.0
PEAK_FRACTION = 0.85
PEAK_PERCENTILE = 99.99
SHORT_RATIO = 2.0
LONG_RATIO = 4.0
SHORT_STEPS = 2


def full_scale_of(image: np.ndarray) -> int:
    """Mono8 为 255，其余按这台相机的 Mono12。"""
    if np.asarray(image).dtype == np.uint8:
        return 255
    return 4095


def frame_metrics(image: np.ndarray, full_scale: int | None = None) -> tuple[float, float]:
    """饱和像素比例，以及最亮约 0.01% 像素的码值。"""
    data = np.asarray(image)
    scale = full_scale_of(data) if full_scale is None else int(full_scale)
    saturated = float(np.mean(data >= scale))
    peak = float(np.percentile(data, PEAK_PERCENTILE))
    return saturated, peak


def plan_bracket(
    exposures_us: list[float],
    saturations: list[float],
    peaks: list[float],
    sat_limit: float = SAT_LIMIT,
    full_scale: int = 4095,
    t_min: float = 1.0,
    t_max: float | None = None,
) -> tuple[list[float], str]:
    """由探测结果生成曝光表。返回微秒列表和一句说明。"""
    if sat_limit <= 0 or sat_limit >= 0.5:
        raise ValueError("饱和上限要大于 0 且小于 50%")
    if len(exposures_us) == 0 or len(exposures_us) != len(saturations) or len(exposures_us) != len(peaks):
        raise ValueError("探测结果不完整")
    rows = sorted(
        (
            (float(exposure), float(saturated), float(peak))
            for exposure, saturated, peak in zip(exposures_us, saturations, peaks)
        ),
        key=lambda row: row[0],
    )
    usable = [row for row in rows if row[1] <= sat_limit]
    if not usable:
        raise ValueError("连最短曝光的饱和都超过上限，先把增益降下来")
    t_hi = usable[-1][0]
    if t_max is not None:
        t_hi = min(t_hi, float(t_max), MAX_EXPOSURE_US)
    else:
        t_hi = min(t_hi, MAX_EXPOSURE_US)
    ceiling = PEAK_FRACTION * float(full_scale)
    linear = [row for row in usable if row[0] <= t_hi and row[2] <= ceiling]
    t_spot = linear[-1][0] if linear else min(usable[0][0], t_hi)
    t_spot = min(t_spot, t_hi)
    t_lo = max(float(t_min), t_spot / (SHORT_RATIO ** SHORT_STEPS))
    if t_hi <= t_lo:
        t_lo = max(float(t_min), t_hi / SHORT_RATIO)
    if t_hi <= t_lo:
        raise ValueError("曝光范围太窄，排不出两档")

    if t_spot <= t_lo * 1.01:
        short = [t_lo]
    else:
        short = _geometric(t_lo, t_spot, SHORT_RATIO)
    if t_hi <= t_spot * 1.01:
        long: list[float] = []
    else:
        long = _geometric(t_spot, t_hi, LONG_RATIO)[1:]
    times = _dedupe(short + long)
    if len(times) < 2:
        raise ValueError("曝光范围太窄，排不出两档")
    hi_sat = usable[-1][1] if abs(usable[-1][0] - t_hi) / t_hi < 0.02 else 0.0
    note = (
        f"最长 { _fmt(t_hi) }，饱和 {hi_sat * 100:.1f}%，上限 {sat_limit * 100:.1f}%。"
        f"斑点档 { _fmt(t_spot) } 及更短按 2 倍，更长按 4 倍。"
    )
    return times, note


def _geometric(start: float, stop: float, ratio: float) -> list[float]:
    if stop <= start * 1.01:
        return [stop]
    values = [start]
    current = start
    while current * ratio < stop * 0.99:
        current *= ratio
        values.append(current)
    if values[-1] < stop * 0.99:
        values.append(stop)
    return values


def _dedupe(values: list[float]) -> list[float]:
    ordered: list[float] = []
    for value in values:
        rounded = float(max(1, int(round(value))))
        if not ordered or rounded > ordered[-1] * 1.05:
            ordered.append(rounded)
    return ordered


def _fmt(exposure_us: float) -> str:
    if exposure_us >= 1_000_000:
        return f"{exposure_us / 1_000_000:.3g} s"
    if exposure_us >= 1_000:
        return f"{exposure_us / 1_000:.3g} ms"
    return f"{exposure_us:.3g} μs"
