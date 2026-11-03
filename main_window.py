#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Debevec 辐射定标融合的桌面界面。"""

from __future__ import annotations

import re
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import QEvent, QPoint, Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont, QImage, QPixmap
from PyQt6.QtWidgets import (
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QStatusBar,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
    QComboBox,
)

from camera_worker import CameraWorker
from calibration_plan import (
    DEFAULT_GAIN_STEPS,
    MAX_GAIN_STEPS,
    MIN_GAIN_STEPS,
    estimate_calibration_seconds,
    format_duration,
)
from debevec import (
    CALIBRATION_LAM,
    CALIBRATION_PIXELS,
    DebevecResult,
    choose_response,
    code_counts,
    frame_log_irradiance,
    infer_code_range,
    linear_response,
    log_display,
    log_limits,
    merge_with_response,
    paint_log,
    radiance_log,
    recover,
)
from exposure_plan import MAX_EXPOSURE_US

_IMAGE_SUFFIXES = {".png", ".tif", ".tiff", ".bmp", ".jpg", ".jpeg", ".npy"}
_EXPOSURE_RE = re.compile(
    r"(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>us|µs|μs|ms|s)\b",
    re.IGNORECASE,
)
_DEFAULT_EXPOSURES_US = (1_000.0, 4_000.0, 16_000.0, 64_000.0, 256_000.0)
_CURVE_PATH = Path(__file__).resolve().parent / "response_detail" / "g_use.npy"
_CURVE_NOTE_PATH = _CURVE_PATH.with_name("calibration.txt")


def _load_saved_curve() -> tuple[np.ndarray, str, bool]:
    """上次标定的结论。没有文件时按 12 位线性响应。"""
    if _CURVE_PATH.is_file():
        curve = np.asarray(np.load(_CURVE_PATH), dtype=np.float64).ravel()
        note = "使用已保存的响应曲线。"
        linear = False
        if _CURVE_NOTE_PATH.is_file():
            lines = _CURVE_NOTE_PATH.read_text(encoding="utf-8").splitlines()
            if lines and lines[0] in ("linear", "measured"):
                linear = lines[0] == "linear"
                note = lines[1] if len(lines) > 1 and lines[1].strip() else note
            elif lines:
                note = lines[0]
        return curve, note, linear
    return linear_response(0, 4095), "尚未标定，按线性 g(Z)=ln(Z/2048) 融合。", True


def _save_curve(curve: np.ndarray, linear: bool, note: str) -> None:
    _CURVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.save(_CURVE_PATH, np.asarray(curve, dtype=np.float64))
    kind = "linear" if linear else "measured"
    _CURVE_NOTE_PATH.write_text(f"{kind}\n{note}\n", encoding="utf-8")


def _runtime_on_path() -> None:
    root = Path(__file__).resolve().parents[1] / "clc_mbe_runtime"
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def exposure_us_from_name(name: str) -> float | None:
    match = None
    for match in _EXPOSURE_RE.finditer(name):
        pass
    if match is None:
        return None
    value = float(match.group("value"))
    unit = match.group("unit").lower().replace("µ", "u").replace("μ", "u")
    if unit == "ms":
        return value * 1_000.0
    if unit == "s":
        return value * 1_000_000.0
    return value


def read_image(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        image = np.load(path)
    else:
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"无法读取 {path.name}")
        if image.ndim == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if image.ndim == 3 and image.shape[2] == 1:
        image = image[..., 0]
    return image


def load_folder(folder: Path) -> tuple[list[np.ndarray], np.ndarray]:
    """从文件夹读曝光序列。优先 exposures.txt，否则从文件名里的 us/ms/s 解析。"""
    images_on_disk = sorted(
        path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES
    )
    if len(images_on_disk) < 2:
        raise ValueError("文件夹里至少要有两张图")
    by_name = {path.name: path for path in images_on_disk}
    sidecar = next(
        (folder / name for name in ("exposures.txt", "exposure_us.txt") if (folder / name).is_file()),
        None,
    )
    paired: list[tuple[Path, float]] = []
    if sidecar is not None:
        numbered: list[float] = []
        named: list[tuple[Path, float]] = []
        for raw in sidecar.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            exposure = float(parts[-1])
            if len(parts) == 1:
                numbered.append(exposure)
            else:
                filename = parts[0]
                if filename not in by_name:
                    raise ValueError(f"{sidecar.name} 里的文件不存在: {filename}")
                named.append((by_name[filename], exposure))
        if named and numbered:
            raise ValueError(f"{sidecar.name} 请统一用“文件名 微秒”或每行一个微秒数")
        if named:
            paired = named
        else:
            if len(numbered) != len(images_on_disk):
                raise ValueError(
                    f"{sidecar.name} 有 {len(numbered)} 行，图像有 {len(images_on_disk)} 张"
                )
            paired = list(zip(images_on_disk, numbered))
    else:
        for path in images_on_disk:
            exposure = exposure_us_from_name(path.name)
            if exposure is None:
                raise ValueError(
                    f"无法从文件名读取曝光: {path.name}。请写成 1000us / 4ms / 0.1s，"
                    "或提供 exposures.txt"
                )
            paired.append((path, exposure))
    paired.sort(key=lambda item: item[1])
    frames = [read_image(path) for path, _exposure in paired]
    times = np.asarray([exposure for _path, exposure in paired], dtype=np.float64)
    return frames, times


def full_scale_peak(image: np.ndarray) -> int:
    if image.dtype == np.uint8:
        return 255
    if np.issubdtype(image.dtype, np.integer) and image.dtype.itemsize >= 2:
        if int(np.max(image)) <= 4095:
            return 4095
        return int(np.iinfo(image.dtype).max)
    return max(int(np.max(image)) if image.size else 1, 1)


def preview_u8(image: np.ndarray) -> np.ndarray:
    """按满量程缩到 8-bit，饱和处保持白色，便于看曝光是否顶格。"""
    peak = full_scale_peak(image)
    scaled = np.clip(np.asarray(image, dtype=np.float32) * (255.0 / peak), 0, 255)
    return np.ascontiguousarray(scaled.astype(np.uint8))


def ndarray_to_qimage(image: np.ndarray) -> QImage:
    image = np.asarray(image)
    if image.ndim == 3 and image.shape[2] == 1:
        image = image[..., 0]
    view = preview_u8(image)
    if view.ndim == 2:
        height, width = view.shape
        qimage = QImage(view.data, width, height, width, QImage.Format.Format_Grayscale8)
        return qimage.copy()
    height, width, channels = view.shape
    if channels != 3:
        raise ValueError(f"无法显示形状 {view.shape}")
    qimage = QImage(view.data, width, height, width * 3, QImage.Format.Format_RGB888)
    return qimage.copy()


class _ExposureThumb(QLabel):
    def __init__(self, owner: "MainWindow", index: int) -> None:
        super().__init__()
        self._owner = owner
        self._index = index

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        if event.button() == Qt.MouseButton.LeftButton:
            self._owner._open_exposure(self._index)
        super().mousePressEvent(event)


class ZoomDialog(QDialog):
    """单张曝光图。滚轮缩放，拖拽平移，双击回到适应窗口。"""

    def __init__(self, parent: QWidget, title: str, image: np.ndarray) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(960, 720)
        self._source = QPixmap.fromImage(ndarray_to_qimage(image))
        self._scale = 1.0
        self._drag: QPoint | None = None
        layout = QVBoxLayout(self)
        hint = QLabel("滚轮缩放，拖拽平移，双击适应窗口")
        layout.addWidget(hint)
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(False)
        self._scroll.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._picture = QLabel()
        self._picture.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._scroll.setWidget(self._picture)
        layout.addWidget(self._scroll, 1)
        self._scroll.viewport().installEventFilter(self)
        self._scroll.viewport().setCursor(Qt.CursorShape.OpenHandCursor)

    def showEvent(self, event) -> None:  # type: ignore[override]
        super().showEvent(event)
        self._fit()

    def _fit(self) -> None:
        viewport = self._scroll.viewport().size()
        width = max(self._source.width(), 1)
        height = max(self._source.height(), 1)
        if viewport.width() < 2 or viewport.height() < 2:
            self._scale = 1.0
        else:
            self._scale = min(viewport.width() / width, viewport.height() / height)
        self._apply()

    def _apply(self) -> None:
        width = max(1, int(round(self._source.width() * self._scale)))
        height = max(1, int(round(self._source.height() * self._scale)))
        mode = (
            Qt.TransformationMode.FastTransformation
            if self._scale >= 2.0
            else Qt.TransformationMode.SmoothTransformation
        )
        self._picture.setPixmap(
            self._source.scaled(width, height, Qt.AspectRatioMode.KeepAspectRatio, mode)
        )
        self._picture.resize(width, height)

    def _zoom(self, factor: float, anchor: QPoint) -> None:
        updated = min(16.0, max(0.1, self._scale * factor))
        if abs(updated - self._scale) < 1e-6:
            return
        horizontal = self._scroll.horizontalScrollBar()
        vertical = self._scroll.verticalScrollBar()
        content_x = horizontal.value() + anchor.x()
        content_y = vertical.value() + anchor.y()
        old_width = max(self._picture.width(), 1)
        old_height = max(self._picture.height(), 1)
        self._scale = updated
        self._apply()
        horizontal.setValue(int(content_x / old_width * self._picture.width() - anchor.x()))
        vertical.setValue(int(content_y / old_height * self._picture.height() - anchor.y()))

    def eventFilter(self, watched, event) -> bool:  # type: ignore[override]
        if watched is not self._scroll.viewport():
            return super().eventFilter(watched, event)
        if event.type() == QEvent.Type.Wheel:
            step = 1.25 if event.angleDelta().y() > 0 else 0.8
            self._zoom(step, event.position().toPoint())
            return True
        if event.type() == QEvent.Type.MouseButtonDblClick:
            self._fit()
            return True
        if event.type() == QEvent.Type.MouseButtonPress and event.button() == Qt.MouseButton.LeftButton:
            self._drag = event.position().toPoint()
            self._scroll.viewport().setCursor(Qt.CursorShape.ClosedHandCursor)
            return True
        if event.type() == QEvent.Type.MouseMove and self._drag is not None:
            point = event.position().toPoint()
            delta = point - self._drag
            self._drag = point
            horizontal = self._scroll.horizontalScrollBar()
            vertical = self._scroll.verticalScrollBar()
            horizontal.setValue(horizontal.value() - delta.x())
            vertical.setValue(vertical.value() - delta.y())
            return True
        if event.type() == QEvent.Type.MouseButtonRelease:
            self._drag = None
            self._scroll.viewport().setCursor(Qt.CursorShape.OpenHandCursor)
            return True
        return super().eventFilter(watched, event)


class FusionThread(QThread):
    succeeded = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(
        self,
        images: list[np.ndarray],
        times: np.ndarray,
        curve: np.ndarray,
        zmin: int,
        zmax: int,
    ) -> None:
        super().__init__()
        self._images = images
        self._times = np.asarray(times, dtype=np.float64).copy()
        self._curve = np.asarray(curve, dtype=np.float64).copy()
        self._zmin = int(zmin)
        self._zmax = int(zmax)

    def run(self) -> None:
        try:
            self.succeeded.emit(
                merge_with_response(self._images, self._times, self._curve, self._zmin, self._zmax)
            )
        except Exception as exc:
            self.failed.emit(str(exc))


class CalibrationThread(QThread):
    succeeded = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, frames: list, gain: float) -> None:
        super().__init__()
        self._frames = frames
        self._gain = float(gain)

    def run(self) -> None:
        try:
            images = [np.asarray(image) for _exposure, image in self._frames]
            times = np.asarray([float(exposure) for exposure, _image in self._frames], dtype=np.float64)
            zmin, zmax = infer_code_range(images)
            solved = recover(
                images,
                times,
                lam=CALIBRATION_LAM,
                n_pixels=CALIBRATION_PIXELS,
                zmin=zmin,
                zmax=zmax,
            )
            curve = np.asarray(solved.g, dtype=np.float64)
            if curve.ndim == 2:
                curve = curve[0]
            counts = code_counts(images, zmin, zmax)
            chosen, linear, rmse = choose_response(curve, zmin, counts)
            self.succeeded.emit(
                {
                    "curve": chosen,
                    "linear": linear,
                    "rmse": rmse,
                    "zmin": zmin,
                    "zmax": zmax,
                    "gain": self._gain,
                    "times": times,
                    "n_used": int(solved.n_used),
                }
            )
        except Exception as exc:
            self.failed.emit(str(exc))


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        _runtime_on_path()
        self.setWindowTitle("Debevec 辐射定标融合")
        self.resize(1280, 820)

        self._stack: list[tuple[float, np.ndarray]] = []
        self._result: DebevecResult | None = None
        self._video_on = False
        self._recording = False
        self._last_preview: np.ndarray | None = None
        self._fusion: FusionThread | None = None
        self._calibration: CalibrationThread | None = None
        self._connected = False
        self._curve, self._curve_note, self._curve_linear = _load_saved_curve()

        self._camera = CameraWorker()
        self._camera.devices_listed.connect(self._on_devices)
        self._camera.preview_frame.connect(self._on_preview)
        self._camera.bracket_done.connect(self._on_bracket)
        self._camera.plan_ready.connect(self._on_plan)
        self._camera.calibration_done.connect(self._on_calibration_frames)
        self._camera.video_frame.connect(self._on_video_frame)
        self._camera.video_stopped.connect(self._on_video_stopped)
        self._camera.progress.connect(self._on_progress)
        self._camera.status_text.connect(self._set_status)
        self._camera.error_text.connect(self._on_error)
        self._camera.connected_changed.connect(self._on_connected)
        self._camera.start()

        self._build_ui()
        self._reset_exposures()
        self._plot_response(self._curve, 0, int(self._curve.size) - 1, compare_linear=not self._curve_linear)
        self._camera.request_list()

    def _build_ui(self) -> None:
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._build_controls())
        splitter.addWidget(self._build_views())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([430, 850])
        self.setCentralWidget(splitter)
        status = QStatusBar()
        self.setStatusBar(status)
        self._progress = QProgressBar()
        self._progress.setMaximumWidth(180)
        self._progress.setTextVisible(True)
        self._progress.setRange(0, 1)
        self._progress.setValue(0)
        status.addPermanentWidget(self._progress)

    def _build_controls(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)

        camera_box = QGroupBox("相机")
        camera_layout = QVBoxLayout(camera_box)
        self._device_combo = QComboBox()
        self._device_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        row = QHBoxLayout()
        refresh = QPushButton("刷新")
        refresh.clicked.connect(self._camera.request_list)
        self._connect_button = QPushButton("连接")
        self._connect_button.clicked.connect(self._toggle_connection)
        row.addWidget(self._device_combo, 1)
        row.addWidget(refresh)
        row.addWidget(self._connect_button)
        camera_layout.addLayout(row)

        self._live = QLabel("尚未连接")
        self._live.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._live.setMinimumSize(360, 240)
        self._live.setStyleSheet("background: #111; color: #ddd;")
        camera_layout.addWidget(self._live)

        gain_row = QFormLayout()
        self._gain = QDoubleSpinBox()
        self._gain.setRange(0.0, 30.0)
        self._gain.setDecimals(2)
        self._gain.setSuffix(" dB")
        self._gain.setToolTip("连接后写入相机。自动增益会被关掉。")
        self._gain.valueChanged.connect(self._camera.set_gain)
        gain_row.addRow("增益", self._gain)
        camera_layout.addLayout(gain_row)
        note = QLabel("预览开始后会关闭自动曝光、自动增益，并把 Gamma 设为 1。括号里的时间就是融合用的 t。")
        note.setWordWrap(True)
        camera_layout.addWidget(note)
        layout.addWidget(camera_box)

        exposure_box = QGroupBox("曝光序列")
        exposure_layout = QVBoxLayout(exposure_box)
        self._table = QTableWidget(0, 1)
        self._table.setHorizontalHeaderLabels(["曝光 (μs)"])
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.verticalHeader().setVisible(False)
        exposure_layout.addWidget(self._table)
        buttons = QHBoxLayout()
        add_button = QPushButton("加一档")
        add_button.clicked.connect(self._add_exposure)
        remove_button = QPushButton("删除")
        remove_button.clicked.connect(self._remove_exposure)
        reset_button = QPushButton("恢复默认")
        reset_button.clicked.connect(self._reset_exposures)
        self._plan_button = QPushButton("自动选档")
        self._plan_button.setEnabled(False)
        self._plan_button.clicked.connect(self._start_plan)
        buttons.addWidget(add_button)
        buttons.addWidget(remove_button)
        buttons.addWidget(reset_button)
        buttons.addWidget(self._plan_button)
        exposure_layout.addLayout(buttons)
        limit_row = QHBoxLayout()
        limit_row.addWidget(QLabel("饱和上限"))
        self._sat_limit = QDoubleSpinBox()
        self._sat_limit.setRange(0.1, 20.0)
        self._sat_limit.setDecimals(1)
        self._sat_limit.setSingleStep(0.1)
        self._sat_limit.setValue(10.0)
        self._sat_limit.setSuffix(" %")
        self._sat_limit.setToolTip(
            "最长一档的饱和像素不超过这个比例。斑点先顶满时，短曝光按 2 倍多采；"
            "再往上按 4 倍补暗部。整幅饱和到 50% 的画面不再使用。"
        )
        limit_row.addWidget(self._sat_limit)
        limit_row.addWidget(QLabel("最长曝光"))
        self._max_exposure = QDoubleSpinBox()
        self._max_exposure.setRange(0.001, MAX_EXPOSURE_US / 1_000_000.0)
        self._max_exposure.setDecimals(3)
        self._max_exposure.setSingleStep(0.1)
        self._max_exposure.setValue(MAX_EXPOSURE_US / 1_000_000.0)
        self._max_exposure.setSuffix(" s")
        self._max_exposure.setToolTip("写到相机上的曝光不超过这个时间，硬上限 5 秒。")
        limit_row.addWidget(self._max_exposure)
        limit_row.addStretch(1)
        exposure_layout.addLayout(limit_row)
        layout.addWidget(exposure_box)

        action_row = QHBoxLayout()
        self._calibrate_button = QPushButton("标定响应")
        self._calibrate_button.setEnabled(False)
        self._calibrate_button.setToolTip(
            "覆盖相机完整增益范围，每档采两轮详细曝光；分别保存响应曲线和质量报告，"
            "失败档会记录后继续，完成后采用质量最好的增益与曲线。"
        )
        self._calibrate_button.clicked.connect(self._start_calibration)
        self._capture_button = QPushButton("采集并融合")
        self._capture_button.setEnabled(False)
        self._capture_button.clicked.connect(self._start_capture)
        self._video_button = QPushButton("视频")
        self._video_button.setEnabled(False)
        self._video_button.clicked.connect(self._toggle_video)
        self._record_button = QPushButton("录制")
        self._record_button.setEnabled(False)
        self._record_button.setToolTip("把融合画面写成 mp4，并保存每帧时间。可以在视频播放中途开始或停下。")
        self._record_button.clicked.connect(self._toggle_record)
        import_button = QPushButton("导入文件夹…")
        import_button.clicked.connect(self._import_folder)
        action_row.addWidget(self._calibrate_button)
        action_row.addWidget(self._capture_button)
        action_row.addWidget(self._video_button)
        action_row.addWidget(self._record_button)
        action_row.addWidget(import_button)
        layout.addLayout(action_row)

        calibration_row = QHBoxLayout()
        calibration_row.addWidget(QLabel("标定增益档数"))
        self._calibration_gain_steps = QSpinBox()
        self._calibration_gain_steps.setRange(MIN_GAIN_STEPS, MAX_GAIN_STEPS)
        self._calibration_gain_steps.setValue(DEFAULT_GAIN_STEPS)
        self._calibration_gain_steps.setToolTip("覆盖相机完整增益范围，包含两个端点。每档采两轮详细曝光。")
        self._calibration_gain_steps.valueChanged.connect(self._update_calibration_estimate)
        calibration_row.addWidget(self._calibration_gain_steps)
        self._calibration_estimate = QLabel()
        calibration_row.addWidget(self._calibration_estimate)
        calibration_row.addStretch(1)
        layout.addLayout(calibration_row)
        self._update_calibration_estimate()

        second_row = QHBoxLayout()
        self._fuse_button = QPushButton("用当前序列融合")
        self._fuse_button.setEnabled(False)
        self._fuse_button.clicked.connect(self._fuse_current)
        self._save_button = QPushButton("保存结果…")
        self._save_button.setEnabled(False)
        self._save_button.clicked.connect(self._save_result)
        second_row.addWidget(self._fuse_button)
        second_row.addWidget(self._save_button)
        layout.addLayout(second_row)

        self._summary = QLabel(self._curve_note + " 相对辐照度 E 由式 (6) 得到。右侧辐射图是对数显示，不写进保存的 E。")
        self._summary.setWordWrap(True)
        layout.addWidget(self._summary)
        layout.addStretch(1)
        return panel

    def _build_views(self) -> QWidget:
        tabs = QTabWidget()
        self._thumbs_host = QWidget()
        self._thumbs_grid = QGridLayout(self._thumbs_host)
        self._thumbs_grid.setAlignment(Qt.AlignmentFlag.AlignTop)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self._thumbs_host)
        sequence_page = QWidget()
        sequence_layout = QVBoxLayout(sequence_page)
        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("显示"))
        self._sequence_mode = QComboBox()
        self._sequence_mode.addItem("拉开，与合成图同一窗口", "stretch")
        self._sequence_mode.addItem("原图，按满量程", "raw")
        self._sequence_mode.currentIndexChanged.connect(self._on_sequence_mode)
        mode_row.addWidget(self._sequence_mode)
        mode_row.addStretch(1)
        sequence_layout.addLayout(mode_row)
        self._sequence_note = QLabel()
        self._sequence_note.setWordWrap(True)
        sequence_layout.addWidget(self._sequence_note)
        sequence_layout.addWidget(scroll, 1)
        self._apply_sequence_note()
        tabs.addTab(sequence_page, "曝光序列")

        self._plot = pg.PlotWidget()
        self._plot.setBackground("w")
        self._plot.showGrid(x=True, y=True, alpha=0.25)
        self._plot.setLabel("bottom", "ln Z")
        self._plot.setLabel("left", "g(Z) = ln 曝光")
        self._plot.addLegend()
        tabs.addTab(self._plot, "响应曲线")

        radiance_page = QWidget()
        radiance_layout = QVBoxLayout(radiance_page)
        percent = QHBoxLayout()
        percent.addWidget(QLabel("显示低百分位"))
        self._low_percent = QDoubleSpinBox()
        self._low_percent.setRange(0.0, 20.0)
        self._low_percent.setValue(1.0)
        self._low_percent.setSuffix(" %")
        self._high_percent = QDoubleSpinBox()
        self._high_percent.setRange(80.0, 100.0)
        self._high_percent.setValue(99.5)
        self._high_percent.setSuffix(" %")
        self._low_percent.valueChanged.connect(self._refresh_radiance_view)
        self._high_percent.valueChanged.connect(self._refresh_radiance_view)
        percent.addWidget(self._low_percent)
        percent.addWidget(QLabel("高百分位"))
        percent.addWidget(self._high_percent)
        percent.addStretch(1)
        radiance_layout.addLayout(percent)
        self._radiance_view = QLabel("融合完成后在这里显示对数辐射图")
        self._radiance_view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._radiance_view.setMinimumSize(480, 360)
        self._radiance_view.setStyleSheet("background: #111; color: #ddd;")
        radiance_layout.addWidget(self._radiance_view, 1)
        tabs.addTab(radiance_page, "辐射图")
        self._tabs = tabs
        return tabs

    def _reset_exposures(self) -> None:
        self._table.setRowCount(0)
        for exposure in _DEFAULT_EXPOSURES_US:
            self._append_exposure(exposure)

    def _append_exposure(self, exposure_us: float) -> None:
        row = self._table.rowCount()
        self._table.insertRow(row)
        item = QTableWidgetItem(f"{exposure_us:.0f}")
        self._table.setItem(row, 0, item)

    def _add_exposure(self) -> None:
        try:
            last = self._read_exposures()[-1]
        except ValueError:
            last = 1_000.0
        cap = self._exposure_cap_us()
        nxt = min(last * 4.0, cap)
        if nxt <= last * 1.01:
            self._set_status(f"已到最长曝光 {self._max_exposure.value():.3g} s")
            return
        self._append_exposure(nxt)

    def _remove_exposure(self) -> None:
        if self._table.rowCount() <= 2:
            self._set_status("至少保留两档曝光")
            return
        row = self._table.currentRow()
        if row < 0:
            row = self._table.rowCount() - 1
        self._table.removeRow(row)

    def _exposure_cap_us(self) -> float:
        return min(self._max_exposure.value() * 1_000_000.0, MAX_EXPOSURE_US)

    def _read_exposures(self) -> list[float]:
        values: list[float] = []
        for row in range(self._table.rowCount()):
            item = self._table.item(row, 0)
            text = "" if item is None else item.text().strip()
            try:
                value = float(text)
            except ValueError as exc:
                raise ValueError(f"第 {row + 1} 档曝光不是数字") from exc
            if value <= 0:
                raise ValueError(f"第 {row + 1} 档曝光必须为正")
            cap = self._exposure_cap_us()
            if value > cap + 0.5:
                raise ValueError(
                    f"第 {row + 1} 档曝光超过最长限制 {self._max_exposure.value():.3g} s"
                )
            values.append(value)
        if len(values) < 2:
            raise ValueError("至少需要两档曝光")
        return values

    def _toggle_connection(self) -> None:
        if self._connected:
            self._connect_button.setEnabled(False)
            self._camera.request_disconnect()
            return
        if self._device_combo.count() == 0:
            self._on_error("没有可连接的相机")
            return
        self._connect_button.setEnabled(False)
        self._camera.request_connect(self._device_combo.currentIndex())

    def _start_plan(self) -> None:
        if not self._connected:
            self._on_error("尚未连接相机")
            return
        self._set_busy(True)
        self._camera.request_plan(self._sat_limit.value() / 100.0, self._exposure_cap_us())

    def _on_plan(self, payload: object) -> None:
        times, note = payload
        self._table.setRowCount(0)
        for exposure in times:
            self._append_exposure(float(exposure))
        self._set_busy(False)
        self._set_status(str(note) + " 表已填好，可以采集并融合。")

    def _response_for_video(self) -> np.ndarray:
        return np.asarray(self._curve, dtype=np.float64).ravel()

    def _curve_for_range(self, zmin: int, zmax: int) -> np.ndarray:
        curve = np.asarray(self._curve, dtype=np.float64).ravel()
        if curve.size == int(zmax) - int(zmin) + 1:
            return curve
        return linear_response(zmin, zmax)

    def _toggle_video(self) -> None:
        if self._video_on:
            self._camera.request_video_stop()
            self._video_button.setEnabled(False)
            return
        if not self._connected:
            self._on_error("尚未连接相机")
            return
        try:
            exposures = self._read_exposures()
        except ValueError as exc:
            self._on_error(str(exc))
            return
        self._video_on = True
        self._video_button.setText("停止视频")
        self._set_busy(True)
        self._video_button.setEnabled(True)
        self._tabs.setCurrentIndex(2)
        window = self._radiance_window()
        self._camera.request_video(exposures, self._response_for_video(), window)
        if window is None:
            self._set_status("视频使用当前响应曲线。融合一次之后，显示窗口会和辐射图锁定。")
        else:
            self._set_status("视频使用当前响应曲线和辐射图窗口。")

    def _toggle_record(self) -> None:
        if self._recording:
            self._recording = False
            self._record_button.setText("录制")
            self._camera.request_record_stop()
            return
        if not self._connected:
            self._on_error("尚未连接相机")
            return
        if not self._video_on:
            try:
                self._read_exposures()
            except ValueError as exc:
                self._on_error(str(exc))
                return
        folder = Path(__file__).resolve().parent / "video_check"
        folder.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        chosen, _filter = QFileDialog.getSaveFileName(
            self,
            "保存融合视频",
            str(folder / f"fusion_{stamp}.mp4"),
            "MP4 (*.mp4)",
        )
        if not chosen:
            return
        path = Path(chosen)
        if path.suffix.lower() != ".mp4":
            path = path.with_suffix(".mp4")
        self._recording = True
        self._record_button.setText("停止录制")
        self._camera.request_record(str(path))
        self._tabs.setCurrentIndex(2)
        if not self._video_on:
            self._toggle_video()

    def _on_video_frame(self, image: np.ndarray) -> None:
        self._show_on_label(self._radiance_view, image)

    def _on_video_stopped(self) -> None:
        self._video_on = False
        self._recording = False
        self._video_button.setText("视频")
        self._record_button.setText("录制")
        self._set_busy(False)

    def _start_calibration(self) -> None:
        if not self._connected:
            self._on_error("尚未连接相机")
            return
        self._set_busy(True)
        count = self._calibration_gain_steps.value()
        self._set_status(
            f"正在标定 {count} 档增益，每档采两轮详细曝光；单档失败会继续。"
        )
        self._camera.request_calibrate(count)

    def _on_calibration_frames(self, payload: object) -> None:
        curve = np.asarray(payload["curve"], dtype=np.float64).ravel()
        gain = float(payload["gain"])
        quality = payload["quality"]
        successful = int(payload["successful"])
        failed = int(payload["failed"])
        manifest = str(payload["manifest"])
        note = (
            f"多增益标定完成：成功 {successful} 档，失败 {failed} 档。"
            f"当前采用 {gain:.2f} dB；重复性 RMSE {quality['repeat_rmse_ln']:.3f} ln，"
            f"留出验证 {quality['held_out_max_rmse_ln']:.3f} ln，"
            f"覆盖率 {quality['coverage_codes_ge_100_fraction']:.1%}。"
        )
        self._curve = curve
        self._curve_linear = False
        self._curve_note = note
        self._gain.blockSignals(True)
        self._gain.setValue(gain)
        self._gain.blockSignals(False)
        _save_curve(curve, False, note)
        self._summary.setText(note + f" 清单：{manifest}")
        self._plot_response(curve, 0, 4095, compare_linear=True)
        self._set_busy(False)
        self._set_status(note + f" 清单已写入 {manifest}")

    def _update_calibration_estimate(self) -> None:
        seconds = estimate_calibration_seconds(self._calibration_gain_steps.value())
        self._calibration_estimate.setText("预计用时 " + format_duration(seconds).removeprefix("约 "))

    def _on_calibrated(self, payload: object) -> None:
        curve = np.asarray(payload["curve"], dtype=np.float64).ravel()
        linear = bool(payload["linear"])
        rmse = float(payload["rmse"])
        gain = float(payload["gain"])
        n_stops = int(np.asarray(payload["times"]).size)
        if not np.isfinite(rmse):
            note = (
                f"码值覆盖不够，不能判断是否线性，融合改用这次测得的曲线。"
                f"增益 {gain:.2f} dB，{n_stops} 档。"
            )
        elif linear:
            note = (
                f"按线性 g(Z)=ln(Z/Zmid) 融合。详细标定相对直线的 RMSE 为 {rmse:.3f} ln，"
                f"未超过 0.05 ln。增益 {gain:.2f} dB，{n_stops} 档。"
            )
        else:
            note = (
                f"使用测得的响应曲线。相对直线的 RMSE 为 {rmse:.3f} ln，超过 0.05 ln。"
                f"增益 {gain:.2f} dB，{n_stops} 档。"
            )
        self._curve = curve
        self._curve_linear = linear and np.isfinite(rmse)
        self._curve_note = note
        _save_curve(curve, self._curve_linear, note)
        self._summary.setText(note + " 相对辐照度 E 由式 (6) 得到。右侧辐射图是对数显示，不写进保存的 E。")
        self._plot_response(curve, int(payload["zmin"]), int(payload["zmax"]), compare_linear=not self._curve_linear)
        self._set_status(note)

    def _calibration_finished(self) -> None:
        self._set_busy(False)

    def _start_capture(self) -> None:
        try:
            exposures = self._read_exposures()
        except ValueError as exc:
            self._on_error(str(exc))
            return
        self._set_busy(True)
        self._camera.request_bracket(exposures)

    def _import_folder(self) -> None:
        chosen = QFileDialog.getExistingDirectory(self, "选择曝光序列文件夹")
        if not chosen:
            return
        try:
            frames, times = load_folder(Path(chosen))
        except Exception as exc:
            self._on_error(str(exc))
            return
        self._stack = [(float(t), frame) for t, frame in zip(times, frames)]
        self._show_stack()
        self._fuse_button.setEnabled(True)
        self._fuse_current()

    def _on_bracket(self, frames: list) -> None:
        self._stack = [(float(exposure), np.asarray(image)) for exposure, image in frames]
        self._show_stack()
        self._fuse_button.setEnabled(True)
        self._fuse_current()

    def _fuse_current(self) -> None:
        if len(self._stack) < 2:
            self._on_error("还没有曝光序列")
            return
        if self._fusion is not None and self._fusion.isRunning():
            self._set_status("正在融合，请稍候")
            return
        images = [image for _exposure, image in self._stack]
        times = np.asarray([exposure for exposure, _image in self._stack], dtype=np.float64)
        zmin, zmax = infer_code_range(images)
        curve = self._curve_for_range(zmin, zmax)
        self._set_busy(True)
        self._set_status("正在用已定的响应曲线融合…")
        thread = FusionThread(images, times, curve, zmin, zmax)
        thread.succeeded.connect(self._on_fused)
        thread.failed.connect(self._on_fuse_failed)
        thread.finished.connect(self._fusion_finished)
        self._fusion = thread
        thread.start()

    def _on_fused(self, result: DebevecResult) -> None:
        self._result = result
        self._save_button.setEnabled(True)
        self._show_curve(result)
        self._refresh_radiance_view()
        self._summary.setText(
            self._curve_note + f" 码值 {result.zmin}–{result.zmax}。保存的是相对辐照度 E，不是对数显示图。"
        )
        self._set_status("融合完成")

    def _on_fuse_failed(self, message: str) -> None:
        self._on_error(message)

    def _fusion_finished(self) -> None:
        self._set_busy(False)

    def _show_stack(self) -> None:
        while self._thumbs_grid.count():
            item = self._thumbs_grid.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        columns = 3
        shown = self._sequence_views()
        for index, ((exposure, image), view) in enumerate(zip(self._stack, shown)):
            holder = QWidget()
            column = QVBoxLayout(holder)
            column.setContentsMargins(4, 4, 4, 4)
            picture = _ExposureThumb(self, index)
            picture.setAlignment(Qt.AlignmentFlag.AlignCenter)
            picture.setCursor(Qt.CursorShape.PointingHandCursor)
            picture.setToolTip("点击放大。" + ("灰阶与辐射图同一窗口。" if self._sequence_stretched() else "满量程原图。"))
            pixmap = QPixmap.fromImage(ndarray_to_qimage(view))
            picture.setPixmap(
                pixmap.scaled(240, 180, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            )
            peak = full_scale_peak(image)
            saturated = float(np.mean(np.asarray(image) >= peak)) if peak else 0.0
            caption = QLabel(f"{self._format_exposure(exposure)}    饱和 {100.0 * saturated:.1f}%")
            caption.setAlignment(Qt.AlignmentFlag.AlignCenter)
            column.addWidget(picture)
            column.addWidget(caption)
            self._thumbs_grid.addWidget(holder, index // columns, index % columns)

    def _open_exposure(self, index: int) -> None:
        if not 0 <= index < len(self._stack):
            return
        exposure, _image = self._stack[index]
        view = self._sequence_views()[index]
        dialog = ZoomDialog(self, f"曝光 {self._format_exposure(exposure)}", view)
        dialog.exec()

    def _show_curve(self, result: DebevecResult) -> None:
        curve = np.asarray(result.g)
        if curve.ndim == 2:
            self._plot.clear()
            legend = self._plot.plotItem.legend
            if legend is not None:
                legend.clear()
            codes = np.arange(result.zmin, result.zmax + 1)
            positive = codes > 0
            log_codes = np.log(codes[positive].astype(np.float64))
            names = ("R", "G", "B")
            colors = ("#1565c0", "#2e7d32", "#c62828")
            for index, channel in enumerate(curve):
                self._plot.plot(
                    log_codes,
                    np.asarray(channel)[positive],
                    pen=pg.mkPen(colors[index % len(colors)], width=2),
                    name=names[index],
                )
            return
        self._plot_response(
            curve,
            result.zmin,
            result.zmax,
            compare_linear=not self._curve_linear,
        )

    def _plot_response(self, curve: np.ndarray, zmin: int, zmax: int, compare_linear: bool) -> None:
        self._plot.clear()
        legend = self._plot.plotItem.legend
        if legend is not None:
            legend.clear()
        codes = np.arange(int(zmin), int(zmax) + 1)
        positive = codes > 0
        log_codes = np.log(codes[positive].astype(np.float64))
        values = np.asarray(curve, dtype=np.float64).ravel()
        self._plot.plot(
            log_codes,
            values[positive],
            pen=pg.mkPen("#1565c0", width=2),
            name="g(Z)",
        )
        if compare_linear:
            ideal = linear_response(zmin, zmax)
            self._plot.plot(
                log_codes,
                ideal[positive],
                pen=pg.mkPen("#9e9e9e", width=1, style=Qt.PenStyle.DashLine),
                name="ln(Z/Zmid)",
            )
        mid = int(round(0.5 * (zmin + zmax)))
        if mid > 0:
            self._plot.addItem(
                pg.InfiniteLine(float(np.log(mid)), angle=90, pen=pg.mkPen("#9e9e9e", style=Qt.PenStyle.DashLine))
            )

    def _sequence_stretched(self) -> bool:
        return self._sequence_mode.currentData() != "raw"

    def _apply_sequence_note(self) -> None:
        if self._sequence_stretched():
            text = (
                "各档按 g(Z)−ln t 拉开。融合前窗口取自整组序列，融合后与辐射图锁定同一窗口。"
                "短曝光也能看清，灰阶可以直接和合成图比；饱和处会和合成图不一样。"
            )
        else:
            text = "按传感器满量程显示原图，饱和处为白色。短曝光码值低，画面会很暗。"
        self._sequence_note.setText(text)

    def _on_sequence_mode(self) -> None:
        self._apply_sequence_note()
        if self._stack:
            self._show_stack()

    def _sequence_views(self) -> list[np.ndarray]:
        """拉开时与辐射图同一对数窗口；原图时按满量程。"""
        if not self._stack:
            return []
        if not self._sequence_stretched():
            return [preview_u8(image) for _exposure, image in self._stack]
        low = self._low_percent.value()
        high = self._high_percent.value()
        if high <= low:
            high = low + 0.1
        curve = self._response_for_video()
        zmin = int(self._result.zmin) if self._result is not None else 0
        logs = [frame_log_irradiance(image, exposure, curve, zmin) for exposure, image in self._stack]
        window = self._radiance_window()
        if window is None:
            sample = np.concatenate([item[::4, ::4].ravel() for item in logs])
            lo, hi = log_limits(sample, low, high)
        else:
            lo, hi = window
        return [paint_log(item, lo, hi) for item in logs]

    def _radiance_window(self) -> tuple[float, float] | None:
        """辐射图当前的对数窗口。视频和拉开的曝光序列都用这一对端点。"""
        if self._result is None:
            return None
        low = self._low_percent.value()
        high = self._high_percent.value()
        if high <= low:
            return None
        return log_limits(radiance_log(self._result.radiance), low, high)

    def _refresh_radiance_view(self) -> None:
        window = self._radiance_window()
        if window is not None:
            self._camera.set_video_window(*window)
        if self._stack and self._sequence_stretched():
            self._show_stack()
        if self._video_on or self._result is None:
            return
        view = log_display(self._result.radiance, self._low_percent.value(), self._high_percent.value())
        self._show_on_label(self._radiance_view, view)

    def _on_preview(self, image: np.ndarray) -> None:
        self._last_preview = np.asarray(image)
        self._show_on_label(self._live, self._last_preview)

    def _show_on_label(self, label: QLabel, image: np.ndarray) -> None:
        pixmap = QPixmap.fromImage(ndarray_to_qimage(image))
        target = label.size()
        if target.width() < 10 or target.height() < 10:
            label.setPixmap(pixmap)
            return
        label.setPixmap(
            pixmap.scaled(target, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        )

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        if self._last_preview is not None:
            self._show_on_label(self._live, self._last_preview)
        self._refresh_radiance_view()

    def _on_devices(self, devices: list) -> None:
        current = self._device_combo.currentIndex()
        self._device_combo.clear()
        for info in devices:
            model = info.get("model_name") or info.get("camera_name") or "camera"
            serial = info.get("serial_number") or ""
            self._device_combo.addItem(f"{model}  {serial}".strip(), info)
        self._device_combo.setEnabled(bool(devices) and not self._connected)
        if devices and 0 <= current < self._device_combo.count():
            self._device_combo.setCurrentIndex(current)

    def _on_connected(self, connected: bool) -> None:
        self._connected = connected
        self._connect_button.setEnabled(True)
        self._connect_button.setText("断开" if connected else "连接")
        self._capture_button.setEnabled(connected)
        self._calibrate_button.setEnabled(connected)
        self._plan_button.setEnabled(connected)
        self._video_button.setEnabled(connected and not self._video_on)
        self._record_button.setEnabled(connected and not self._recording)
        if not connected:
            self._recording = False
            self._record_button.setText("录制")
        self._device_combo.setEnabled(not connected and self._device_combo.count() > 0)
        if not connected:
            self._live.setPixmap(QPixmap())
            self._live.setText("尚未连接")
            self._last_preview = None

    def _on_progress(self, done: int, total: int) -> None:
        self._progress.setRange(0, max(total, 1))
        self._progress.setValue(done)

    def _set_busy(self, busy: bool) -> None:
        self._capture_button.setEnabled(self._connected and not busy and not self._video_on)
        self._calibrate_button.setEnabled(self._connected and not busy and not self._video_on)
        self._plan_button.setEnabled(self._connected and not busy and not self._video_on)
        self._video_button.setEnabled(self._connected and (not busy or self._video_on))
        self._record_button.setEnabled(self._connected and (not busy or self._video_on or self._recording))
        self._fuse_button.setEnabled(len(self._stack) >= 2 and not busy)
        self._save_button.setEnabled(self._result is not None and not busy)

    def _save_result(self) -> None:
        if self._result is None:
            return
        chosen = QFileDialog.getExistingDirectory(self, "选择保存目录")
        if not chosen:
            return
        directory = Path(chosen)
        targets = [
            directory / "radiance.npy",
            directory / "radiance.hdr",
            directory / "response.npz",
            directory / "display.png",
        ]
        if any(path.exists() for path in targets):
            answer = QMessageBox.question(
                self,
                "覆盖文件",
                "该目录里已有融合结果，要覆盖吗？",
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        try:
            written = write_result(directory, self._result, self._low_percent.value(), self._high_percent.value())
        except Exception as exc:
            self._on_error(f"保存失败: {exc}")
            return
        self._set_status("已保存 " + "、".join(path.name for path in written))

    def _on_error(self, message: str) -> None:
        self._video_on = False
        self._recording = False
        self._video_button.setText("视频")
        self._record_button.setText("录制")
        self._set_busy(False)
        self._set_status(message)
        QMessageBox.warning(self, "Debevec 辐射定标融合", message)

    def _set_status(self, message: str) -> None:
        self.statusBar().showMessage(message, 10000)

    @staticmethod
    def _format_exposure(exposure_us: float) -> str:
        if exposure_us >= 1_000_000:
            return f"{exposure_us / 1_000_000:.3g} s"
        if exposure_us >= 1_000:
            return f"{exposure_us / 1_000:.3g} ms"
        return f"{exposure_us:.3g} μs"

    def closeEvent(self, event) -> None:  # type: ignore[override]
        self._camera.shutdown()
        self._camera.wait(4000)
        if self._fusion is not None and self._fusion.isRunning():
            self._fusion.wait(4000)
        super().closeEvent(event)


def write_result(
    directory: Path,
    result: DebevecResult,
    low_percentile: float,
    high_percentile: float,
) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    radiance_path = directory / "radiance.npy"
    np.save(radiance_path, result.radiance)
    written.append(radiance_path)

    response_path = directory / "response.npz"
    np.savez(
        response_path,
        g=result.g,
        zmin=np.int32(result.zmin),
        zmax=np.int32(result.zmax),
        times_us=result.times,
        lam=np.float64(result.lam),
    )
    written.append(response_path)

    hdr = result.radiance
    if hdr.ndim == 2:
        hdr = np.repeat(hdr[..., None], 3, axis=2)
    hdr_bgr = np.ascontiguousarray(hdr[..., ::-1], dtype=np.float32)
    hdr_path = directory / "radiance.hdr"
    if cv2.imwrite(str(hdr_path), hdr_bgr):
        written.append(hdr_path)

    display = log_display(result.radiance, low_percentile, high_percentile)
    display_path = directory / "display.png"
    if not cv2.imwrite(str(display_path), display):
        raise RuntimeError("对数显示 PNG 没有写成功")
    written.append(display_path)
    return written
