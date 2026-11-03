#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""多增益响应标定的纯计算函数。"""

from __future__ import annotations

import math

DEFAULT_GAIN_STEPS = 7
MIN_GAIN_STEPS = 2
MAX_GAIN_STEPS = 32
DEFAULT_EXPOSURES_US = (
    500.0,
    1_000.0,
    2_000.0,
    4_000.0,
    8_000.0,
    16_000.0,
    32_000.0,
    64_000.0,
    128_000.0,
    256_000.0,
    512_000.0,
)


def gain_targets(
    low: float,
    high: float,
    count: int,
    increment: float | None = None,
) -> list[float]:
    """生成包含端点的等间距可写增益；提供步进时取最近的离散档。"""
    low = float(low)
    high = float(high)
    count = int(count)
    if not math.isfinite(low) or not math.isfinite(high) or high < low:
        raise ValueError("非法增益范围")
    if not MIN_GAIN_STEPS <= count <= MAX_GAIN_STEPS:
        raise ValueError(f"增益档数必须在 {MIN_GAIN_STEPS}–{MAX_GAIN_STEPS} 之间")
    if high == low:
        raise ValueError("增益范围没有跨度，无法生成多档增益")
    if increment is not None:
        increment = float(increment)
        if not math.isfinite(increment) or increment <= 0:
            raise ValueError("非法增益步进")
        writable_steps = int(math.floor((high - low) / increment + 1e-9))
        if writable_steps + 1 < count:
            raise ValueError("相机可写增益档少于请求档数")
        indices = [round(index * writable_steps / (count - 1)) for index in range(count)]
        values = [low + int(index) * increment for index in indices]
        values[0], values[-1] = low, high
        return values
    step = (high - low) / (count - 1)
    values = [low + step * index for index in range(count)]
    values[0], values[-1] = low, high
    return values


def infer_gain_increment(low: float, high: float, read_after_write) -> float | None:
    """相机不报告步进时，用一次中间值和二分查找读回差。

    读回与写入一致时视为连续增益，返回 None。
    """
    low = float(low)
    high = float(high)
    span = high - low
    if span <= 0:
        return None
    midpoint = low + span / 2.0
    if abs(float(read_after_write(midpoint)) - midpoint) <= 1e-4:
        return None
    base = float(read_after_write(low))
    lo = 0.0
    hi = span
    step: float | None = None
    for _ in range(20):
        mid = (lo + hi) / 2.0
        if mid <= 1e-6:
            break
        delta = abs(float(read_after_write(low + mid)) - base)
        if delta > 1e-4:
            step = delta
            hi = mid
        else:
            lo = mid
        if hi - lo <= max(1e-4, span * 1e-6):
            break
    return None if step is None else float(step)


def estimate_calibration_seconds(
    gain_count: int,
    exposures_us: tuple[float, ...] | list[float] = DEFAULT_EXPOSURES_US,
    rounds: int = 2,
    settle_frames: int = 3,
    frame_overhead_s: float = 0.12,
    gain_overhead_s: float = 2.0,
) -> float:
    """按曝光积分、稳定帧和每档切换开销估算批量标定时长。"""
    if not MIN_GAIN_STEPS <= int(gain_count) <= MAX_GAIN_STEPS:
        raise ValueError(f"增益档数必须在 {MIN_GAIN_STEPS}–{MAX_GAIN_STEPS} 之间")
    if int(rounds) < 1 or int(settle_frames) < 1:
        raise ValueError("轮数和稳定帧数必须为正")
    values = [float(value) for value in exposures_us]
    if len(values) < 2 or any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("曝光序列必须包含至少两个有限正数")
    per_round = sum((value / 1_000_000.0) * settle_frames + frame_overhead_s for value in values)
    return int(gain_count) * (int(rounds) * per_round + float(gain_overhead_s))


def format_duration(seconds: float) -> str:
    total = max(0, int(math.ceil(float(seconds))))
    minutes, remain = divmod(total, 60)
    if minutes:
        return f"约 {minutes} 分 {remain} 秒"
    return f"约 {remain} 秒"
