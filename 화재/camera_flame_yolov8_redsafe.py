#!/usr/bin/env python3
from __future__ import annotations

"""외장 웹캠 화재 감지(YOLOv8 + 안전 박스)

- YOLOv8 가중치가 있으면 화염을 빨간 박스로 표시
- 붉은계열(비화염)과 그 외 일반 물체는 모두 녹색 박스
- 가중치가 없으면 휴리스틱(색/에지) 기반으로만 녹색/빨간 박스 제공

설치(예):
  pip install ultralytics opencv-python numpy torch torchvision

실행 예:
  python 완벽(예정)/camera_flame_yolov8_redsafe.py --cam -1
  python 완벽(예정)/camera_flame_yolov8_redsafe.py --device /dev/video1
  python 완벽(예정)/camera_flame_yolov8_redsafe.py --weights /path/to/best.pt
"""

import argparse
import os
import time
from typing import List, Tuple, cast

import cv2
import numpy as np

from common_utils import open_camera, force_cpu, find_yolo_weights
from red_utils import RedParams, find_red_regions, find_general_regions, FireHeuParams, filter_fire_like


def _overlay_text(img, text: str, org, color, scale=0.6, thick=2):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def run(weights: str, cam: int, width: int, height: int, fps: int,
        conf: float, iou_th: float, red_iou_th: float,
        disp_scale: float = 1.0, fullscreen: bool = False,
        flicker_off: bool = False) -> int:
    force_cpu()

    have_yolo = False
    model = None
    names = {}
    if weights and os.path.isfile(weights):
        try:
            from ultralytics import YOLO  # type: ignore
            model = YOLO(weights)
            names = model.names if hasattr(model, 'names') else {}
            have_yolo = True
            print(f"[INFO] YOLO weights: {weights}")
        except Exception as e:
            print('[WARN] YOLO 로드 실패. 휴리스틱으로 진행:', e)

    cap = open_camera(cam, width, height, fps)
    if not cap or not cap.isOpened():
        print('[ERR] 카메라 열기 실패')
        return 2

    def cname(idx: int) -> str:
        if isinstance(names, dict):
            return str(names.get(int(idx), int(idx)))
        if isinstance(names, (list, tuple)):
            i = int(idx)
            return str(names[i]) if 0 <= i < len(names) else str(i)
        return str(idx)

    fire_labels = {"fire", "flame"}
    red_params = RedParams()
    t0 = time.time()
    n = 0
    win = 'Flame YOLOv8 (완벽)'

    # Headless safety: if no DISPLAY, run offscreen with auto-exit
    use_gui = bool(os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY') or os.name == 'nt')
    headless_start = None

    # Optional warmup to avoid initial black frames
    for _ in range(5):
        ok_w, _ = cap.read()
        if not ok_w:
            break
    # Prepare window
    win = 'Flame YOLOv8 (완벽)'
    if os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY') or os.name == 'nt':
        try:
            cv2.namedWindow(win, cv2.WINDOW_NORMAL)
            if fullscreen:
                cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        except Exception:
            pass

    cam_label = os.environ.get('FIRECAM_DEVICE') or (f"/dev/video{cam}" if cam >= 0 else 'auto')

    prev_V = None
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        frame = cast(np.ndarray, frame)
        H, W = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        V = hsv[:, :, 2]

        # 1) YOLO fire detection (if available)
        fire_boxes: List[Tuple[int, int, int, int, float]] = []
        general_boxes: List[Tuple[int, int, int, int, float]] = []
        if have_yolo and model is not None:
            try:
                res = model(frame, conf=conf, iou=iou_th, verbose=False)[0]
                boxes = getattr(res, 'boxes', None)
                if boxes is not None:
                    xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes, 'xyxy') else None
                    confs = boxes.conf.cpu().numpy() if hasattr(boxes, 'conf') else None
                    clss = boxes.cls.cpu().numpy().astype(int) if hasattr(boxes, 'cls') else None
                    if xyxy is None or confs is None or clss is None:
                        data = getattr(boxes, 'data', None)
                        if isinstance(data, np.ndarray) and data.ndim == 2 and data.shape[1] >= 6:
                            xyxy = data[:, :4]
                            confs = data[:, 4]
                            clss = data[:, 5].astype(int)
                    if xyxy is not None and confs is not None and clss is not None:
                        for i in range(xyxy.shape[0]):
                            x1, y1, x2, y2 = [int(v) for v in xyxy[i].tolist()]
                            label = cname(clss[i]).lower()
                            if label in fire_labels or (len(fire_labels) == 1 and next(iter(fire_labels)) in label):
                                fire_boxes.append((x1, y1, x2, y2, float(confs[i])))
                            else:
                                general_boxes.append((x1, y1, x2, y2, float(confs[i])))
            except Exception:
                pass

        # 2) Red (non-fire) proposals
        def _iou(a, b) -> float:
            ax1, ay1, ax2, ay2 = a
            bx1, by1, bx2, by2 = b
            ix1, iy1 = max(ax1, bx1), max(ay1, by1)
            ix2, iy2 = min(ax2, bx2), min(ay2, by2)
            iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
            inter = iw * ih
            if inter <= 0: return 0.0
            a_area = max(1, (ax2 - ax1) * (ay2 - ay1))
            b_area = max(1, (bx2 - bx1) * (by2 - by1))
            return float(inter) / float(a_area + b_area - inter)

        base_red_boxes: List[Tuple[int, int, int, int, float]] = []
        for x1, y1, x2, y2, fill in find_red_regions(frame, red_params):
            if all(_iou((x1,y1,x2,y2), (fx1,fy1,fx2,fy2)) < red_iou_th for fx1,fy1,fx2,fy2,_ in fire_boxes):
                base_red_boxes.append((x1, y1, x2, y2, fill))
        # Heuristic flame classification for red regions (draw red)
        fire_like = filter_fire_like(frame, base_red_boxes, FireHeuParams(flicker_gate=not flicker_off), prev_V=prev_V)
        # Keep the red regions that are NOT flame-like as green SAFE
        fire_set = {(x1,y1,x2,y2) for (x1,y1,x2,y2,_) in fire_like}
        red_boxes: List[Tuple[int,int,int,int,float]] = [b for b in base_red_boxes if (b[0],b[1],b[2],b[3]) not in fire_set]

        # 3) General proposals (for anything else)
        if not general_boxes:
            for x1, y1, x2, y2, score in find_general_regions(frame):
                if all(_iou((x1,y1,x2,y2), (fx1,fy1,fx2,fy2)) < red_iou_th for fx1,fy1,fx2,fy2,_ in fire_boxes):
                    general_boxes.append((x1, y1, x2, y2, score))

        # 4) Draw overlays
        # YOLO fire (red)
        for x1, y1, x2, y2, confv in fire_boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            _overlay_text(frame, f"FIRE {confv*100:.1f}%", (x1, max(0, y1-6)), (0,0,255), 0.7, 2)
            
            # 밑에 세 줄 새로 추가했음 - 감지했는지 알려주는 코드
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            print(f"FIRE {cx:.2f} {cy:.2f}", flush=True)        

        # Heuristic fire-like (red)
        for x1, y1, x2, y2, sv in fire_like:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            _overlay_text(frame, f"FIRE {sv*100:.1f}%", (x1, max(0, y1-6)), (0,0,255), 0.7, 2)

            # 밑에 세줄 추가했음 - 감지했는지 알려주는 코드
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0    
            print(f"FIRE {cx:.2f} {cy:.2f}", flush=True)
        
        # for x1, y1, x2, y2, s in red_boxes:
        #     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
        #     _overlay_text(frame, f"SAFE {s*100:.1f}%", (x1, max(0, y1-6)), (0,200,0), 0.6, 2)
        #for x1, y1, x2, y2, s in general_boxes:
        #    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 1)

        # Camera label overlay (helps verify external device)
        try:
            _overlay_text(frame, f"Camera: {cam_label}", (12, H-10), (180,255,180), 0.55, 1)
        except Exception:
            pass

        n += 1
        fps_now = n / (time.time() - t0 + 1e-6)
        _overlay_text(frame, f"{fps_now:.1f} FPS", (12, 26), (220,220,220), 0.7, 1)

        if use_gui:
            try:
                if disp_scale != 1.0:
                    dispW, dispH = int(W * disp_scale), int(H * disp_scale)
                    frame_show = cv2.resize(frame, (dispW, dispH), interpolation=cv2.INTER_LINEAR)
                else:
                    frame_show = frame
                cv2.imshow(win, frame_show)
                if not fullscreen:
                    try:
                        cv2.resizeWindow(win, frame_show.shape[1], frame_show.shape[0])
                    except Exception:
                        pass
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            except Exception:
                use_gui = False
                headless_start = time.time()
                print('[WARN] GUI 불가: 헤드리스 모드로 전환(10초 후 자동 종료)')
        else:
            if headless_start is None:
                headless_start = time.time()
            if time.time() - headless_start > float(os.environ.get('FIRECAM_HEADLESS_SECS', '10')):
                break

        # update temporal cache
        prev_V = V.copy()

    cap.release()
    cv2.destroyAllWindows()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description='완벽 YOLOv8 Flame + Safe boxes')
    ap.add_argument('--weights', type=str, default='', help='YOLO .pt (없으면 자동 탐색)')
    ap.add_argument('--cam', type=int, default=-1, help='카메라 인덱스(-1: 외장 자동)')
    ap.add_argument('--device', type=str, default='', help='카메라 장치 경로(/dev/videoX 또는 by-id)')
    ap.add_argument('--width', type=int, default=1280)
    ap.add_argument('--height', type=int, default=720)
    ap.add_argument('--fps', type=int, default=30)
    ap.add_argument('--conf', type=float, default=0.20) # 조정함
    ap.add_argument('--iou', type=float, default=0.50) # 조정함
    ap.add_argument('--red-iou-th', type=float, default=0.30)
    ap.add_argument('--scale', type=float, default=1.4, help='표시 배율(창에 그릴 때 확대)')
    ap.add_argument('--fullscreen', action='store_true', help='전체화면')
    ap.add_argument('--flicker-off', action='store_true', help='플리커 게이트 끄기(정지 화면의 불꽃도 허용)')
    args = ap.parse_args()

    # Offscreen GUI if no display
    if not (os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY') or os.name == 'nt'):
        os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    else:
        os.environ.setdefault('QT_QPA_PLATFORM', 'xcb')

    if args.device:
        os.environ['FIRECAM_DEVICE'] = args.device

    base_dir = os.path.dirname(os.path.abspath(__file__))
    weights = args.weights or find_yolo_weights(base_dir)
    return run(weights, args.cam, args.width, args.height, args.fps,
               args.conf, args.iou, args.red_iou_th,
               disp_scale=float(getattr(args, 'scale', 1.0)),
               fullscreen=bool(getattr(args, 'fullscreen', False)),
               flicker_off=bool(getattr(args, 'flicker_off', False)))

if __name__ == '__main__':
    raise SystemExit(main())

# ==== 단일 프레임 API (Flask/Qt 연동용) ==========================
# 입력: BGR np.ndarray  /  출력: 오버레이가 그려진 BGR np.ndarray
import cv2, numpy as np
from red_utils import RedParams, find_red_regions, find_general_regions, FireHeuParams, filter_fire_like
from common_utils import find_yolo_weights
import os

# Lazy YOLO loader for single-frame path (prefer real flame over generic red)
try:
    from ultralytics import YOLO  # type: ignore
except Exception:
    YOLO = None  # type: ignore
_yolo_model = None
_yolo_ready = False
_yolo_names = {}

_prev_V_sf = None  # 밝기 채널 캐시(깜빡임 판정용)
# 단일 프레임 경로에서의 마지막 화재 상태/중심 좌표(브리지를 통해 서버 알림에 사용)
last_fire_detected_sf = False
last_fire_center_sf = None  # type: tuple[float, float] | None
# 가장 최근 프레임에서 빨간 박스로 표시된 화재 후보 박스 목록
# 튜플: (x1, y1, x2, y2, score/confidence)
last_fire_boxes_sf: list[tuple[float, float, float, float, float]] = []


def _global_fire_ratio(bgr: np.ndarray) -> float:
    """프레임 전체에서 빨강/주황 성분 비율을 계산해 휴리스틱으로 사용."""
    try:
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, (0, 70, 70), (20, 255, 255))
        mask2 = cv2.inRange(hsv, (160, 70, 70), (180, 255, 255))
        ratio = (mask1.mean() + mask2.mean()) / (2.0 * 255.0)
        return float(ratio)
    except Exception:
        return 0.0


def _iou_sf(a, b) -> float:
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0: return 0.0
    a_area = max(1, (ax2 - ax1) * (ay2 - ay1))
    b_area = max(1, (bx2 - bx1) * (by2 - by1))
    return float(inter) / float(a_area + b_area - inter)

def detect_and_draw(frame):
    """
    Flask 브리지가 호출하는 단일 프레임 처리 함수.
    - red_utils 휴리스틱으로 FIRE/SAFE 박스 그려서 반환
    - (원하면 YOLO도 여기에 추가 가능)
    """
    global _prev_V_sf
    try:
        H, W = frame.shape[:2]
        fire_ratio = _global_fire_ratio(frame)
        ratio_thr = float(os.environ.get('FIRE_GLOBAL_RATIO', '0.045'))
        global_fire_hint = fire_ratio >= ratio_thr
        # 0) Optional YOLO (flame-only) — if weights/model available, use it preferentially
        global _yolo_model, _yolo_ready, _yolo_names
        fire_yolo = False
        yolo_boxes = []  # (x1,y1,x2,y2,conf)
        if not _yolo_ready and YOLO is not None:
            try:
                wp = os.environ.get('FIRECAM_YOLO_WEIGHTS') or find_yolo_weights(os.path.dirname(os.path.abspath(__file__)))
                if wp and os.path.isfile(wp):
                    _yolo_model = YOLO(wp)
                    _yolo_names = getattr(_yolo_model, 'names', {})
                    _yolo_ready = True
                    print(f"[SF] YOLO loaded: {wp}")
            except Exception as e:
                _yolo_ready = False
        if _yolo_ready and _yolo_model is not None:
            try:
                conf = float(os.environ.get('DETECT_CONF', '0.25'))
                iou = float(os.environ.get('YOLO_IOU', '0.45'))
                imgsz = int(float(os.environ.get('YOLO_IMGSZ', '640')))
                aug = os.environ.get('YOLO_AUG', '0').lower() in ('1','true','yes','on')
                res = _yolo_model(frame, conf=conf, iou=iou, imgsz=imgsz, augment=aug, verbose=False)[0]
                boxes = getattr(res, 'boxes', None)
                if boxes is not None:
                    xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes, 'xyxy') else None
                    confs = boxes.conf.cpu().numpy() if hasattr(boxes, 'conf') else None
                    clss = boxes.cls.cpu().numpy().astype(int) if hasattr(boxes, 'cls') else None
                    if xyxy is None or confs is None or clss is None:
                        data = getattr(boxes, 'data', None)
                        if isinstance(data, np.ndarray) and data.ndim == 2 and data.shape[1] >= 6:
                            xyxy = data[:, :4]
                            confs = data[:, 4]
                            clss = data[:, 5].astype(int)
                    def is_fire_label(s: str) -> bool:
                        s = s.lower(); return ('fire' in s) or ('flame' in s)
                    def cname(i: int) -> str:
                        if isinstance(_yolo_names, dict):
                            return str(_yolo_names.get(int(i), int(i)))
                        if isinstance(_yolo_names, (list, tuple)):
                            ii = int(i)
                            return str(_yolo_names[ii]) if 0 <= ii < len(_yolo_names) else str(ii)
                        return str(i)
                    if xyxy is not None and confs is not None and clss is not None:
                        for i in range(xyxy.shape[0]):
                            x1, y1, x2, y2 = [int(v) for v in xyxy[i].tolist()]
                            label = cname(clss[i])
                            if is_fire_label(label):
                                yolo_boxes.append((x1, y1, x2, y2, float(confs[i])))
                if yolo_boxes:
                    fire_yolo = True
                    for x1,y1,x2,y2,confv in yolo_boxes:
                        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),3)
                        cv2.putText(frame, f"FIRE {confv*100:.1f}%", (x1, max(0,y1-6)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2, cv2.LINE_AA)
            except Exception:
                pass
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        V = hsv[:, :, 2]

        # 1) 붉은 후보 추출 (휴리스틱)
        red_params = RedParams()
        base_red = [(x1,y1,x2,y2,fill) for x1,y1,x2,y2,fill in find_red_regions(frame, red_params)]

        # 2) 깜빡임/형상 기반으로 불꽃 유사군 선별(빨강)
        # 단일 프레임 경로에서는 정지 장면도 허용하도록 flicker 게이트 비활성화
        fire_like = filter_fire_like(frame, base_red, FireHeuParams(flicker_gate=False), prev_V=_prev_V_sf)

        # 불꽃 유사군을 제외한 붉은 후보는 SAFE(초록)
        fire_set = {(x1,y1,x2,y2) for (x1,y1,x2,y2,_) in fire_like}
        red_boxes = [b for b in base_red if (b[0],b[1],b[2],b[3]) not in fire_set]

        # 일반 후보(보조용)
        general_boxes = [(x1,y1,x2,y2,score) for x1,y1,x2,y2,score in find_general_regions(frame)]

        # 2.5) 전역 색 힌트 기반 fallback — 화면 내 붉은 비율이 높으면 가장 큰 붉은 영역을 화염으로 인정
        if (not fire_yolo) and (not fire_like) and global_fire_hint and base_red:
            fallback_box = max(base_red, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
            area = (fallback_box[2] - fallback_box[0]) * (fallback_box[3] - fallback_box[1])
            frame_area = max(1.0, float(H * W))
            min_frac = float(os.environ.get('FIRE_FALLBACK_MIN_FRAC', '0.004'))
            min_area = float(os.environ.get('FIRE_FALLBACK_MIN_AREA', '1800'))
            if area >= max(min_area, frame_area * min_frac):
                fire_like = [fallback_box]

        # 3) 그리기: FIRE(빨강) — YOLO 박스가 있으면 우선 표시, 없으면 휴리스틱 박스 표시
        if not fire_yolo:
            for x1,y1,x2,y2,sv in fire_like:
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),3)
                cv2.putText(frame, f"FIRE {sv*100:.1f}%", (x1, max(0,y1-6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2, cv2.LINE_AA)

        # SAFE/General(초록) 박스 표시 제거 → FIRE(빨강)만 유지
        # for x1,y1,x2,y2,s in red_boxes:
        #     cv2.rectangle(frame,(x1,y1),(x2,y2),(0,200,0),2)
        #     cv2.putText(frame, f"SAFE {s*100:.1f}%", (x1, max(0,y1-6)),
        #                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,200,0),2, cv2.LINE_AA)

        # for x1,y1,x2,y2,s in general_boxes:
        #     cv2.rectangle(frame,(x1,y1),(x2,y2),(0,200,0),1)

        # 전역 플래그/좌표 업데이트(브리지 통해 서버 전송 트리거용)
        try:
            global last_fire_detected_sf, last_fire_center_sf, last_fire_boxes_sf
            # 실제로 그려진 빨간 박스 목록을 계산한다.
            if fire_yolo and yolo_boxes:
                active_boxes = yolo_boxes
            elif fire_like:
                active_boxes = fire_like
            else:
                active_boxes = []

            if active_boxes:
                last_fire_detected_sf = True
                bx = max(active_boxes, key=lambda b: float((b[2]-b[0]) * (b[3]-b[1])))
                x1, y1, x2, y2, score = bx
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                last_fire_center_sf = (float(cx), float(cy))
                # score가 없으면 0.0으로 기록(휴리스틱 대비)
                last_fire_boxes_sf = [
                    (float(b[0]), float(b[1]), float(b[2]), float(b[3]), float(b[4]) if len(b) >= 5 else 0.0)
                    for b in active_boxes
                ]
            else:
                last_fire_detected_sf = False
                last_fire_center_sf = None
                last_fire_boxes_sf = []
        except Exception:
            pass

        _prev_V_sf = V.copy()
        return frame
    except Exception:
        # 실패해도 스트림 끊기지 않게 원본 반환
        return frame
# ==== 단일 프레임 API 끝 =========================================
