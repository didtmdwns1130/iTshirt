#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional

import cv2
import numpy as np


@dataclass
class RedParams:
    h1_low: int = 0
    h1_high: int = 12
    h2_low: int = 165
    h2_high: int = 179
    s_low: int = 70
    v_low: int = 60
    min_area: int = 120
    min_fill: float = 0.08
    open_iter: int = 1
    close_iter: int = 1


def find_red_regions(bgr: np.ndarray, p: RedParams) -> List[Tuple[int, int, int, int, float]]:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    m1 = (H >= p.h1_low) & (H <= p.h1_high) & (S >= p.s_low) & (V >= p.v_low)
    m2 = (H >= p.h2_low) & (H <= p.h2_high) & (S >= p.s_low) & (V >= p.v_low)
    mask = (m1 | m2).astype(np.uint8) * 255
    k = np.ones((3, 3), np.uint8)
    if p.open_iter > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=p.open_iter)
    if p.close_iter > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=p.close_iter)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    Himg, Wimg = bgr.shape[:2]
    out: List[Tuple[int, int, int, int, float]] = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w * h < p.min_area:
            continue
        roi = mask[y:y+h, x:x+w]
        fill = float((roi > 0).sum()) / float(max(1, roi.size))
        if fill < p.min_fill:
            continue
        x1, y1, x2, y2 = x, y, x + w, y + h
        x1 = max(0, min(Wimg - 1, x1)); y1 = max(0, min(Himg - 1, y1))
        x2 = max(0, min(Wimg - 1, x2)); y2 = max(0, min(Himg - 1, y2))
        if x2 <= x1 or y2 <= y1:
            continue
        out.append((x1, y1, x2, y2, fill))
    return out


def find_general_regions(
    bgr: np.ndarray,
    min_area: int = 300,
    canny1: int = 80,
    canny2: int = 160,
    dilate_iter: int = 1,
    edge_density_min: float = 0.004,
) -> List[Tuple[int, int, int, int, float]]:
    """High-contrast region proposals for "safe" green boxes.
    Returns: list of (x1,y1,x2,y2,score) where score=edge density.
    """
    if bgr is None or bgr.size == 0:
        return []
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(gray, canny1, canny2)
    if dilate_iter > 0:
        k = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, k, iterations=dilate_iter)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    H, W = bgr.shape[:2]
    out: List[Tuple[int, int, int, int, float]] = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w * h < min_area:
            continue
        roi = edges[y:y+h, x:x+w]
        edge_density = float((roi > 0).sum()) / float(max(1, roi.size))
        if edge_density < edge_density_min:
            continue
        x1, y1, x2, y2 = x, y, x + w, y + h
        x1 = max(0, min(W - 1, x1)); y1 = max(0, min(H - 1, y1))
        x2 = max(0, min(W - 1, x2)); y2 = max(0, min(H - 1, y2))
        if x2 <= x1 or y2 <= y1:
            continue
        out.append((x1, y1, x2, y2, edge_density))
    return out


# ---------------- Flame-like heuristic (no model) ----------------
@dataclass
class FireHeuParams:
    # color/brightness gates
    yellow_h_low: int = 12
    yellow_h_high: int = 32
    yellow_v_low: int = 180
    yellow_ratio_min: float = 0.05
    white_v_low: int = 220
    white_s_high: int = 90
    white_ratio_min: float = 0.008
    bright_v_min: int = 180
    bright_ratio_min: float = 0.08
    # variation/texture
    sat_mean_min: float = 28.0
    h_std_min: float = 4.0
    v_std_min: float = 9.0
    lap_var_min: float = 20.0
    # hotspot
    hotspot_v_min: int = 190
    hotspot_min_area: int = 8
    hotspots_min: int = 0
    # box aspect gate
    min_aspect: float = 0.12
    # large ROI rules
    large_roi_frac: float = 0.20  # if ROI area >= this fraction of frame area → stronger hotspot gate
    hotspots_large_min: int = 2
    # temporal flicker
    flicker_gate: bool = True
    flicker_abs_thr: int = 12
    flicker_min_change: int = 900
    flicker_ratio_min: float = 0.03


def _roi_features_from_hsv(roiH: np.ndarray, roiS: np.ndarray, roiV: np.ndarray, P: FireHeuParams, prev_roiV: Optional[np.ndarray] = None):
    area = max(1, roiH.size)
    yellow = ((roiH >= P.yellow_h_low) & (roiH <= P.yellow_h_high) & (roiV >= P.yellow_v_low))
    white = ((roiV >= P.white_v_low) & (roiS <= P.white_s_high))
    bright = (roiV >= P.bright_v_min)
    yr = float(yellow.sum()) / float(area)
    wr = float(white.sum()) / float(area)
    br = float(bright.sum()) / float(area)
    sat_mean = float(roiS.mean()) if area else 0.0
    h_std = float(roiH.astype('float32').std())
    v_std = float(roiV.astype('float32').std())
    bm = (roiV >= max(P.white_v_low, P.hotspot_v_min)).astype('uint8') * 255
    cnts, _ = cv2.findContours(bm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hs = 0
    for c in cnts:
        if cv2.contourArea(c) >= P.hotspot_min_area:
            hs += 1
    lap = cv2.Laplacian(roiV, cv2.CV_64F)
    lap_var = float(lap.var())
    flicker_ratio = 0.0
    flicker_count = 0
    if prev_roiV is not None and prev_roiV.size == roiV.size and prev_roiV.shape == roiV.shape:
        d = cv2.absdiff(roiV, prev_roiV)
        m = (d >= P.flicker_abs_thr).astype('uint8')
        flicker_count = int(m.sum())
        flicker_ratio = float(flicker_count) / float(max(1, roiV.size))
    return {
        'yellow_ratio': yr,
        'white_ratio': wr,
        'bright_ratio': br,
        'sat_mean': sat_mean,
        'h_std': h_std,
        'v_std': v_std,
        'hotspots': hs,
        'lap_var': lap_var,
        'flicker_count': flicker_count,
        'flicker_ratio': flicker_ratio,
    }


def filter_fire_like(bgr: np.ndarray,
                     boxes: List[Tuple[int, int, int, int, float]],
                     P: FireHeuParams | None = None,
                     prev_V: Optional[np.ndarray] = None) -> List[Tuple[int, int, int, int, float]]:
    """Return subset of boxes that look like flames.
    Input boxes are (x1,y1,x2,y2,score) from red regions.
    """
    if P is None:
        P = FireHeuParams()
    if bgr is None or bgr.size == 0:
        return []
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    out: List[Tuple[int, int, int, int, float]] = []
    for x1, y1, x2, y2, fill in boxes:
        w = max(1, x2 - x1); h = max(1, y2 - y1)
        if min(w / h, h / w) < P.min_aspect:
            continue
        roiH = H[y1:y2, x1:x2]
        roiS = S[y1:y2, x1:x2]
        roiV = V[y1:y2, x1:x2]
        prev_roiV = None
        if prev_V is not None and prev_V.ndim == 2:
            prev_roiV = prev_V[y1:y2, x1:x2]
        f = _roi_features_from_hsv(roiH, roiS, roiV, P, prev_roiV)
        if f['sat_mean'] < P.sat_mean_min:
            continue
        if f['h_std'] < P.h_std_min or f['v_std'] < P.v_std_min:
            continue
        # core color + brightness
        if f['bright_ratio'] < P.bright_ratio_min:
            continue
        strict = (f['yellow_ratio'] >= P.yellow_ratio_min and f['white_ratio'] >= P.white_ratio_min)
        loose = (
            f['yellow_ratio'] >= max(0.8 * P.yellow_ratio_min, 0.04) and
            f['bright_ratio'] >= max(0.8 * P.bright_ratio_min, 0.10) and
            (f['white_ratio'] >= 0.5 * P.white_ratio_min or f['h_std'] >= P.h_std_min + 1.0) and
            f['lap_var'] >= P.lap_var_min
        )
        if not (strict or loose):
            continue
        # require hotspots (more for larger areas)
        min_hs = P.hotspots_min
        # If ROI is large fraction of frame, require more hotspots
        if (w * h) >= int(P.large_roi_frac * (bgr.shape[0] * bgr.shape[1])):
            min_hs = max(min_hs, P.hotspots_large_min)
        if f['hotspots'] < min_hs:
            continue
        # temporal flicker gate to suppress static red objects (e.g., apples)
        if P.flicker_gate and prev_V is not None:
            if (f['flicker_count'] < P.flicker_min_change and f['flicker_ratio'] < P.flicker_ratio_min):
                continue
        # score: mix of yellow, bright, white, var
        score = 0.55 * min(1.0, f['yellow_ratio'] / max(0.12, P.yellow_ratio_min * 2))
        score += 0.20 * min(1.0, f['bright_ratio'] / 0.30)
        score += 0.15 * min(1.0, f['white_ratio'] / max(0.05, P.white_ratio_min * 2))
        score += 0.10 * min(1.0, max(0.0, (min(f['h_std'], 12.0) - 4.0) / 8.0))
        out.append((x1, y1, x2, y2, float(max(0.0, min(1.0, score)))))
    return out
