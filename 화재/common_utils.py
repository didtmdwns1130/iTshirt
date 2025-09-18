#!/usr/bin/env python3
from __future__ import annotations

"""공통 카메라/도구 함수 (완벽 프로젝트).
 - 외장(USB) 웹캠 자동 선택 또는 명시 경로/인덱스 지정
 - YOLO 가중치 자동 탐색
 - GPU 비활성화(안정성)
"""

import glob
import os
import platform
from typing import List

import cv2

# OpenCV 호환 레이어 (Pylance 경고 회피 + 플랫폼마다 상수 보정)
VW_FOURCC = getattr(cv2, "VideoWriter_fourcc", None)
CAP_PROP_FPS         = getattr(cv2, "CAP_PROP_FPS", 5)
CAP_PROP_FRAME_WIDTH = getattr(cv2, "CAP_PROP_FRAME_WIDTH", 3)
CAP_PROP_FRAME_HEIGHT= getattr(cv2, "CAP_PROP_FRAME_HEIGHT", 4)
CAP_PROP_FOURCC      = getattr(cv2, "CAP_PROP_FOURCC", 6)


# ===== 카메라 튜닝/워밍업 공통 로직 =====
_PREFERRED_FOURCCS = ('MJPG', 'YUYV', 'YUY2', '')

def _tune_and_warmup(cap, width: int, height: int, fps: int) -> bool:
    """포맷/해상도/FPS 적용 후 첫 프레임 읽어 성공 여부 반환."""
    for fmt in _PREFERRED_FOURCCS:
        fcc = int(VW_FOURCC(*fmt)) if (fmt and VW_FOURCC) else 0
        if fcc:
            cap.set(CAP_PROP_FOURCC, fcc)
        cap.set(CAP_PROP_FPS, fps)
        cap.set(CAP_PROP_FRAME_WIDTH,  width)
        cap.set(CAP_PROP_FRAME_HEIGHT, height)

        ok, _ = cap.read()  # 드라이버 적용 대기 겸 1프레임 워밍업
        if ok:
            return True
    return False


def force_cpu() -> None:
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')


class _NullCap:
    def isOpened(self) -> bool: return False
    def read(self): return False, None
    def set(self, *_a, **_k): return False
    def get(self, *_a, **_k): return 0.0
    def release(self) -> None: return None


def _video_indices() -> List[int]:
    idx: List[int] = []
    try:
        for p in glob.glob('/dev/video*'):
            b = os.path.basename(p)
            if b.startswith('video') and b[5:].isdigit():
                idx.append(int(b[5:]))
    except Exception:
        pass
    return sorted(set(idx))


def _video_by_id_paths() -> List[str]:
    ps: List[str] = []
    try:
        for p in glob.glob('/dev/v4l/by-id/*'):
            if os.path.islink(p):
                ps.append(p)
    except Exception:
        return []
    ps.sort(key=lambda s: (0 if 'usb' in os.path.basename(s).lower() else 1, s))
    return ps


def _camera_sysinfo(idx: int) -> tuple[str, str]:
    name = ''
    devpath = ''
    try:
        p = f'/sys/class/video4linux/video{idx}/name'
        if os.path.isfile(p):
            name = open(p, 'r', encoding='utf-8', errors='ignore').read().strip()
    except Exception:
        pass
    try:
        link = f'/sys/class/video4linux/video{idx}/device'
        if os.path.exists(link):
            devpath = os.path.realpath(link)
    except Exception:
        pass
    return name, devpath


def _score_external(idx: int) -> int:
    name, devpath = _camera_sysinfo(idx)
    nl = name.lower()
    score = 0
    if '/usb' in devpath: score += 3
    if any(k in nl for k in ('logitech', 'webcam', 'uvc', 'razer', 'elgato', 'c920', 'c922', 'c930', 'brio')):
        score += 3
    if any(k in nl for k in ('integrated', 'easycamera', 'chicony', 'bison', 'realtek', 'sonix', 'lenovo', 'hp hd')):
        score -= 2
    if idx > 0: score += 1
    return score


def _try_open(src):
    sysname = platform.system().lower()
    if 'linux' in sysname:
        backends = [getattr(cv2, 'CAP_V4L2', 200), 0]
    elif 'windows' in sysname:
        backends = [getattr(cv2, 'CAP_DSHOW', 700), getattr(cv2, 'CAP_MSMF', 1400), 0]
    elif 'darwin' in sysname or 'mac' in sysname:
        backends = [getattr(cv2, 'CAP_AVFOUNDATION', 1200), 0]
    else:
        backends = [0]
    for be in backends:
        try:
            cap = cv2.VideoCapture(src, be)
            if cap.isOpened():
                return cap
            cap.release()
        except Exception:
            try: cap.release()
            except Exception: pass
    return _NullCap()


def open_camera(index: int, width: int, height: int, fps: int = 30):
    # 1) 환경변수 우선 (명시 경로/인덱스)
    dev = os.environ.get('FIRECAM_DEVICE', '').strip()
    env_idx = os.environ.get('FIRECAM_CAM_INDEX', '').strip()

    if dev and (os.path.exists(dev) or dev.startswith('/dev/v4l/by-id/')):
        cap = _try_open(dev)
        if cap.isOpened() and _tune_and_warmup(cap, width, height, fps):
            print(f"[INFO] Camera opened via FIRECAM_DEVICE: {dev}")
            return cap
        try: cap.release()
        except Exception: pass

    if env_idx.isdigit():
        cap = _try_open(int(env_idx))
        if cap.isOpened() and _tune_and_warmup(cap, width, height, fps):
            print(f"[INFO] Camera opened via FIRECAM_CAM_INDEX: {env_idx}")
            return cap
        try: cap.release()
        except Exception: pass

    # 2) 자동 선택 (index < 0)
    if int(index) < 0:
        # by-id 우선 (USB 친화적)
        for path in _video_by_id_paths():
            cap = _try_open(path)
            if cap.isOpened():
                if _tune_and_warmup(cap, width, height, fps):
                    print(f"[INFO] Camera opened: {path} (by-id)")
                    return cap
                cap.release()
        # 점수 기반 인덱스 선택
        cand = _video_indices()
        cand.sort(key=_score_external, reverse=True)
        for i in cand or [1, 0, 2, 3]:
            cap = _try_open(i)
            if not cap.isOpened():
                try: cap.release()
                except Exception: pass
                continue
            if _tune_and_warmup(cap, width, height, fps):
                name, _ = _camera_sysinfo(i)
                if name:
                    print(f"[INFO] Camera opened: /dev/video{i} (V4L2) name='{name}'")
                else:
                    print(f"[INFO] Camera opened: /dev/video{i} (V4L2)")
                return cap
            cap.release()
        return _NullCap()

    # 3) 명시적 인덱스
    cap = _try_open(int(index))
    if cap.isOpened() and _tune_and_warmup(cap, width, height, fps):
        return cap
    try: cap.release()
    except Exception: pass
    return _NullCap()


def find_yolo_weights(base_dir: str) -> str:
    """YOLO 가중치(.pt) 자동 탐색.
    우선순위: base_dir/best.pt → base_dir/yolov*.pt → base_dir/*detect*.pt → 상위 runs/detect/best.pt
    MobileNet 등 분류기 가중치는 제외.
    """
    cands: List[str] = []
    def _add(pat: str):
        for p in glob.glob(pat, recursive=True):
            n = os.path.basename(p).lower()
            if any(k in n for k in ('mobilenet', 'resnet', 'v2', 'v3')):
                continue
            if p.endswith('.pt'):
                cands.append(p)
    _add(os.path.join(base_dir, 'best.pt'))
    _add(os.path.join(base_dir, 'yolov*.pt'))
    _add(os.path.join(base_dir, '*detect*.pt'))
    _add(os.path.join(os.path.dirname(base_dir), '**', 'runs', 'detect', '**', 'weights', 'best.pt'))
    cands = [p for p in cands if os.path.isfile(p)]
    if not cands:
        return ''
    def score(p: str):
        n = os.path.basename(p).lower()
        s1 = 2 if n == 'best.pt' else (1 if ('best' in n or 'yolo' in n or 'detect' in n) else 0)
        return (s1, os.path.getmtime(p))
    cands.sort(key=score, reverse=True)
    return cands[0]
