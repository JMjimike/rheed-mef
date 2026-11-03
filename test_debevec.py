#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""用已知响应合成曝光，检查 Debevec 恢复。"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from debevec import (  # noqa: E402
    choose_response,
    frame_log_irradiance,
    hat_weights,
    linear_response,
    log_display,
    log_limits,
    merge_radiance,
    merge_with_response,
    paint_log,
    radiance_log,
    recover,
    zmid_of,
)
from exposure_plan import MAX_EXPOSURE_US, plan_bracket  # noqa: E402
from calibration_plan import (  # noqa: E402
    estimate_calibration_seconds,
    gain_targets,
    infer_gain_increment,
)
from sliding_fusion import FusionRecorder, SlidingFusion, level_matches  # noqa: E402


def _smooth_scene(height: int, width: int, rng: np.random.Generator) -> np.ndarray:
    yy, xx = np.mgrid[0:height, 0:width]
    field = 0.55 + 0.35 * np.sin(xx / width * 2.2 * np.pi) * np.cos(yy / height * 1.7 * np.pi)
    field += 0.08 * np.sin((xx + yy) / max(width, height) * 3.0 * np.pi)
    field += 0.02 * rng.standard_normal(field.shape)
    return np.clip(field, 0.02, 1.0)


def _synthesize(scene, times, zmax, gamma, t_ref):
    images = []
    dtype = np.uint8 if zmax <= 255 else np.uint16
    for shutter in times:
        ratio = np.clip(scene * shutter / t_ref, 0.0, 1.0)
        codes = np.rint(zmax * np.power(ratio, gamma))
        images.append(np.clip(codes, 0, zmax).astype(dtype))
    return images


def _true_g(zmax, gamma, t_ref):
    z = np.arange(0, zmax + 1, dtype=np.float64)
    curve = np.full(z.shape, np.nan)
    positive = z > 0
    curve[positive] = np.log(t_ref) + (1.0 / gamma) * np.log(z[positive] / zmax)
    return curve


class DebevecRecoveryTest(unittest.TestCase):
    def test_hat_weight_is_zero_at_ends_and_peaks_in_the_middle(self):
        w = hat_weights(0, 255)
        self.assertEqual(w[0], 0.0)
        self.assertEqual(w[255], 0.0)
        self.assertGreater(w[128], w[10])
        self.assertEqual(w[127], w[128])

    def test_recovers_gamma_response_and_radiance_8bit(self):
        self._assert_recovers(zmax=255, height=96, width=96, n_pixels=80)

    def test_recovers_gamma_response_and_radiance_12bit(self):
        self._assert_recovers(zmax=4095, height=80, width=80, n_pixels=200)

    def test_color_channels_are_recovered_separately(self):
        rng = np.random.default_rng(7)
        scene = _smooth_scene(48, 48, rng)
        times = np.array([1.0, 4.0, 16.0, 64.0, 256.0])
        gamma = 0.6
        t_ref = 256.0
        planes = []
        for scale in (1.0, 0.7, 0.4):
            images = _synthesize(np.clip(scene * scale, 0.02, 1.0), times, 255, gamma, t_ref)
            planes.append(images)
        color = [np.stack([planes[c][j] for c in range(3)], axis=-1) for j in range(len(times))]
        result = recover(color, times, lam=10.0, n_pixels=80)
        self.assertEqual(result.g.shape[0], 3)
        self.assertEqual(result.radiance.shape[-1], 3)
        for channel in range(3):
            self.assertGreater(result.g[channel, 200] , result.g[channel, 40])

    def test_saturated_pixel_falls_back_to_shortest_exposure(self):
        zmax = 255
        g = np.log(np.arange(1, zmax + 2))
        g = g - g[zmid_of(0, zmax)]
        images = [
            np.array([[255, 0]], dtype=np.uint8),
            np.array([[255, 0]], dtype=np.uint8),
        ]
        times = np.array([1.0, 8.0])
        radiance = merge_radiance(images, times, g, 0, zmax)
        # 两档都在权重为 0 的端点。最接近中灰的是更靠近 127.5 的码值，
        # 这里 0 和 255 等距，第一档先写入，距离不小于时保持第一档。
        expected_dark = np.exp(g[0] - np.log(times[0]))
        self.assertAlmostEqual(float(radiance[0, 1]), float(expected_dark), places=4)

    def test_short_exposure_stays_visible_on_the_fusion_window(self):
        zmax = 255
        curve = _true_g(zmax, 1.0, 1.0)
        curve[0] = curve[1]
        scene = _smooth_scene(40, 40, np.random.default_rng(0))
        times = np.array([1.0, 4.0])
        images = _synthesize(scene, times, zmax, 1.0, 4.0)
        fused = merge_radiance(images, times, curve, 0, zmax)
        lo, hi = log_limits(radiance_log(fused), 1.0, 99.5)
        fused_view = paint_log(radiance_log(fused), lo, hi)
        short_view = paint_log(frame_log_irradiance(images[0], times[0], curve), lo, hi)
        usable = (images[0] > 0) & (images[0] < zmax)
        difference = np.abs(short_view.astype(np.int16) - fused_view.astype(np.int16))
        self.assertLess(float(difference[usable].mean()), 2.0)
        raw = images[0].astype(np.float32) * (255.0 / zmax)
        self.assertGreater(float(short_view.mean()), float(raw.mean()) + 20.0)

    def test_log_display_uses_percentiles_only_for_viewing(self):
        radiance = np.array([[1.0, 10.0], [100.0, 1000.0]], dtype=np.float32)
        view = log_display(radiance, 0.0, 100.0)
        self.assertEqual(view.dtype, np.uint8)
        self.assertEqual(view.shape, (2, 2))
        self.assertLess(view[0, 0], view[1, 1])

    def test_rejects_nonfinite_exposure_and_curve(self) -> None:
        images = [
            np.full((8, 8), 40, dtype=np.uint8),
            np.full((8, 8), 120, dtype=np.uint8),
        ]
        curve = linear_response(0, 255)
        with self.assertRaises(ValueError):
            merge_radiance(images, np.array([1.0, np.nan]), curve, 0, 255)
        curve[100] = np.nan
        with self.assertRaises(ValueError):
            merge_with_response(images, np.array([1.0, 2.0]), curve, 0, 255)

    def test_straight_response_is_replaced_by_the_linear_model(self) -> None:
        ideal = linear_response(0, 255)
        counts = np.full(ideal.size, 400, dtype=np.int64)
        chosen, use_linear, rmse = choose_response(ideal, 0, counts)
        self.assertTrue(use_linear)
        self.assertLess(rmse, 1e-9)
        self.assertAlmostEqual(float(chosen[128]), 0.0, places=6)
        self.assertAlmostEqual(float(chosen[64]), float(np.log(64 / 128)), places=6)

    def test_bent_response_is_kept(self) -> None:
        bent = linear_response(0, 255) * 2.2
        counts = np.full(bent.size, 400, dtype=np.int64)
        chosen, use_linear, rmse = choose_response(bent, 0, counts)
        self.assertFalse(use_linear)
        self.assertGreater(rmse, 0.05)
        self.assertAlmostEqual(float(chosen[40]), float(bent[40]), places=6)

    def test_too_few_codes_does_not_claim_a_straight_line(self) -> None:
        ideal = linear_response(0, 255)
        counts = np.zeros(ideal.size, dtype=np.int64)
        counts[40:60] = 500
        _chosen, use_linear, rmse = choose_response(ideal, 0, counts)
        self.assertFalse(use_linear)
        self.assertFalse(np.isfinite(rmse))

    def _assert_recovers(self, zmax, height, width, n_pixels):
        rng = np.random.default_rng(1)
        scene = _smooth_scene(height, width, rng)
        times = np.array([1.0, 4.0, 16.0, 64.0, 256.0])
        gamma = 0.5
        t_ref = 256.0
        images = _synthesize(scene, times, zmax, gamma, t_ref)
        result = recover(images, times, lam=10.0, n_pixels=n_pixels, zmin=0, zmax=zmax)
        self.assertFalse(result.short)
        self.assertGreaterEqual(result.n_used, result.n_required)

        truth = _true_g(zmax, gamma, t_ref)
        mid = zmid_of(0, zmax)
        aligned = truth - truth[mid]
        lo = int(zmax * 0.15)
        hi = int(zmax * 0.85)
        err = result.g[lo:hi] - aligned[lo:hi]
        rmse = float(np.sqrt(np.mean(err ** 2)))
        self.assertLess(rmse, 0.08, f"g RMSE={rmse:.4f} zmax={zmax}")
        self.assertGreater(float(result.g[hi] - result.g[lo]), 0.5)

        ratio = result.radiance.astype(np.float64) / scene
        interior = (images[2] > zmax * 0.05) & (images[2] < zmax * 0.95)
        self.assertGreater(int(np.count_nonzero(interior)), 100)
        scale = float(np.median(ratio[interior]))
        rel = np.abs(result.radiance[interior] - scale * scene[interior]) / (scale * scene[interior])
        self.assertLess(float(np.median(rel)), 0.05, f"radiance median rel={np.median(rel):.4f}")


class StackFileTest(unittest.TestCase):
    def test_filename_units_and_roundtrip_files(self):
        import tempfile

        import cv2

        from main_window import exposure_us_from_name, load_folder, write_result
        from debevec import DebevecResult

        self.assertEqual(exposure_us_from_name("spot_1000us.tif"), 1000.0)
        self.assertEqual(exposure_us_from_name("spot_4ms.png"), 4000.0)
        self.assertEqual(exposure_us_from_name("spot_0.1s.tif"), 100_000.0)

        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            for name, value in (("a_1000us.png", 10), ("b_4ms.png", 40)):
                image = np.full((8, 8), value, dtype=np.uint8)
                self.assertTrue(cv2.imwrite(str(folder / name), image))
            frames, times = load_folder(folder)
            self.assertEqual(list(times), [1000.0, 4000.0])
            self.assertEqual(int(frames[0][0, 0]), 10)
            self.assertEqual(int(frames[1][0, 0]), 40)

            result = DebevecResult(
                radiance=np.ones((8, 8), dtype=np.float32),
                g=np.zeros(256, dtype=np.float64),
                zmin=0,
                zmax=255,
                times=times,
                lam=10.0,
                n_requested=20,
                n_required=20,
                n_used=20,
                raised=False,
                short=False,
                istop=2,
            )
            out = folder / "out"
            written = write_result(out, result, 1.0, 99.0)
            names = {path.name for path in written}
            self.assertIn("radiance.npy", names)
            self.assertIn("response.npz", names)
            self.assertIn("display.png", names)
            loaded = np.load(out / "radiance.npy")
            self.assertEqual(loaded.shape, (8, 8))
            with np.load(out / "response.npz") as curve:
                self.assertEqual(int(curve["zmax"]), 255)


class ExposurePlanTest(unittest.TestCase):
    def test_stops_before_ten_percent_and_packs_short_exposures(self) -> None:
        # 和实拍同一形状：16 ms 饱和 0.9%，32 ms 已经 45%；8 ms 最亮像素顶满。
        exposures = [500, 1000, 2000, 4000, 8000, 16000, 32000]
        saturations = [0.0, 0.0, 0.0, 0.0, 0.0, 0.009, 0.45]
        peaks = [251, 487, 980, 1945, 3898, 4095, 4095]
        times, note = plan_bracket(exposures, saturations, peaks, t_min=26)
        self.assertEqual(times, [1000.0, 2000.0, 4000.0, 16000.0])
        self.assertIn("16 ms", note)
        self.assertNotIn("32", note)

    def test_dim_scene_still_adds_short_stops(self) -> None:
        times, _note = plan_bracket(
            [1000, 2000, 4000],
            [0.0, 0.0, 0.0],
            [100, 200, 400],
            t_min=26,
        )
        self.assertEqual(times, [1000.0, 2000.0, 4000.0])

    def test_rejects_a_frame_that_is_already_half_saturated(self) -> None:
        with self.assertRaises(ValueError):
            plan_bracket([1000], [0.55], [4095], t_min=26)

    def test_longest_stop_stays_within_five_seconds(self) -> None:
        times, note = plan_bracket(
            [1_000_000, 5_000_000, 10_000_000, 20_000_000],
            [0.0, 0.0, 0.0, 0.0],
            [200, 800, 2000, 4000],
            t_min=26,
        )
        self.assertLessEqual(max(times), MAX_EXPOSURE_US)
        self.assertEqual(times[-1], MAX_EXPOSURE_US)
        self.assertIn("5 s", note)


class CalibrationPlanTest(unittest.TestCase):
    def test_gain_targets_include_endpoints_and_are_evenly_spaced(self) -> None:
        values = gain_targets(0.0, 24.0, 7)
        self.assertEqual(values[0], 0.0)
        self.assertEqual(values[-1], 24.0)
        np.testing.assert_allclose(np.diff(values), np.full(6, 4.0))
        quantized = gain_targets(0.0, 10.0, 4, increment=1.0)
        self.assertEqual(quantized, [0.0, 3.0, 7.0, 10.0])

    def test_hidden_integer_gain_step_is_inferred(self) -> None:
        def read_after_write(value: float) -> float:
            return float(int(float(value) + 0.5))

        step = infer_gain_increment(0.0, 35.0, read_after_write)
        self.assertEqual(step, 1.0)
        self.assertEqual(gain_targets(0.0, 35.0, 7, increment=step)[1], 6.0)
        self.assertIsNone(infer_gain_increment(0.0, 35.0, lambda value: float(value)))

    def test_bad_gain_counts_and_flat_range_fail(self) -> None:
        with self.assertRaises(ValueError):
            gain_targets(3.5, 18.0, 1)
        with self.assertRaises(ValueError):
            gain_targets(0.0, 24.0, 0)
        with self.assertRaises(ValueError):
            gain_targets(0.0, 24.0, 33)
        with self.assertRaises(ValueError):
            gain_targets(3.5, 3.5, 7)

    def test_estimated_time_scales_with_gain_count_and_exposure(self) -> None:
        two = estimate_calibration_seconds(
            2, [1_000.0, 2_000.0], rounds=2, settle_frames=3, frame_overhead_s=0.1
        )
        six = estimate_calibration_seconds(
            6, [1_000.0, 2_000.0], rounds=2, settle_frames=3, frame_overhead_s=0.1
        )
        self.assertAlmostEqual(six, two * 3)
        self.assertGreater(
            estimate_calibration_seconds(2, [1_000.0, 4_000.0]),
            estimate_calibration_seconds(2, [1_000.0, 2_000.0]),
        )


class SlidingDisplayTest(unittest.TestCase):
    def test_wrong_exposure_does_not_match_the_other_stops(self) -> None:
        self.assertTrue(level_matches(74.0, 1000.0, 0.074))
        self.assertFalse(level_matches(74.0, 16000.0, 0.074))
        self.assertFalse(level_matches(0.0, 1000.0, 0.074))
        self.assertTrue(level_matches(80.0, 1000.0, None))

    def test_display_scale_stays_fixed_when_the_scene_changes(self) -> None:
        curve = np.log(np.maximum(np.arange(256), 1) / 128.0)
        fusion = SlidingFusion(curve, 2)
        dim = np.full((32, 32), 30, dtype=np.uint8)
        for _ in range(2):
            fusion.push(0, dim, 1000.0)
            fusion.push(1, dim, 2000.0)
        bright = np.full((32, 32), 200, dtype=np.uint8)
        means = []
        for _ in range(10):
            fusion.push(0, bright, 1000.0)
            view = fusion.push(1, bright, 2000.0)
            self.assertIsNotNone(view)
            means.append(float(view.mean()))
        self.assertGreater(means[0], 10.0)
        self.assertLess(max(means) - min(means), 0.51)

    def test_video_uses_the_still_fusion_window(self) -> None:
        curve = np.log(np.maximum(np.arange(256), 1.0) / 128.0)
        curve[0] = curve[1]
        images = [np.full((16, 16), 40, np.uint8), np.full((16, 16), 160, np.uint8)]
        times = np.array([1000.0, 4000.0])
        radiance = merge_radiance(images, times, curve, 0, 255)
        low, high = log_limits(radiance_log(radiance), 1.0, 99.5)
        still = paint_log(radiance_log(radiance), low, high)
        fusion = SlidingFusion(curve, 2, window=(low, high))
        fusion.push(0, images[0], times[0])
        view = fusion.push(1, images[1], times[1])
        self.assertIsNotNone(view)
        self.assertLess(abs(int(view[0, 0]) - int(still[0, 0])), 2)
        fusion.set_display_window(low - 5.0, high + 5.0)
        widened = fusion.push(1, images[1], times[1])
        expected = paint_log(radiance_log(radiance), low - 5.0, high + 5.0)
        self.assertLess(abs(int(widened[0, 0]) - int(expected[0, 0])), 2)

    def test_recorder_writes_frames_and_timestamps(self) -> None:
        import cv2

        recorder = FusionRecorder()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "clip.mp4"
            recorder.set_active(str(path))
            for index in range(10):
                frame = np.full((32, 48), 20 + index, dtype=np.uint8)
                recorder.push(frame, now=100.0 + index * 0.1)
            info = recorder.close()
            self.assertIsNotNone(info)
            assert info is not None
            self.assertEqual(info["frames"], 10)
            self.assertTrue(path.is_file())
            self.assertTrue(info["time_path"].is_file())
            text = info["time_path"].read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(text), 11)
            self.assertAlmostEqual(float(text[-1].split(",")[1]), 0.9, places=3)
            capture = cv2.VideoCapture(str(path))
            count = 0
            while capture.read()[0]:
                count += 1
            capture.release()
            self.assertEqual(count, 10)
            self.assertIsNone(recorder.close())


if __name__ == "__main__":
    unittest.main()
