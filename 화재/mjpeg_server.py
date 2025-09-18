# mjpeg_server.py  — 화재 인식 오버레이 포함 송출 서버
import atexit
import os
import socket
import sys
import threading
import time
from typing import Any, Iterator, Optional, Tuple, cast

import cv2
import numpy as np
from flask import Flask, Response, make_response
from numpy.typing import NDArray

from common_utils import open_camera as _cu_open_camera

SINK_HOST = os.environ.get("SINK_HOST", "192.168.0.11")   # 관리자 서버 IP
SINK_PORT = int(os.environ.get("SINK_PORT", 8888))
X_FIXED   = 1.00
Y_FIXED   = 1.00
# 감지 즉시 전송 요구에 맞춰 기본 쿨다운 0.0으로 설정(환경변수로 재조정 가능)
COOLDOWN_SEC = float(os.environ.get("COOLDOWN_SEC", 120.0))
_last_sent_ts = 0.0
_last_sent_signature = None  # type: tuple[float, float] | None
_fire_active = False
_send_lock = threading.Lock()

# 좌표 전송 모드: 'norm'(0~1 정규화) 또는 'pixel'(픽셀 좌표)
COORD_MODE = os.environ.get("COORD_MODE", "norm").strip().lower()

BGRImage = NDArray[np.uint8]

def _send_coords(x: float, y: float) -> bool:
    try:
        with socket.create_connection((SINK_HOST, SINK_PORT), timeout=1.0) as s:
            payload = f"{x:.2f},{y:.2f}\n".encode("utf-8")
            s.sendall(payload)
            try:
                print(f"[SEND] {x:.2f},{y:.2f} -> {SINK_HOST}:{SINK_PORT}")
            except Exception:
                pass
        return True
    except OSError as e:
        try:
            print(f"[SEND-ERR] connect/send failed to {SINK_HOST}:{SINK_PORT}: {e}")
        except Exception:
            pass
        return False


def _signature_for(x: float, y: float) -> tuple[float, float]:
    if COORD_MODE == 'pixel':
        return (float(round(x)), float(round(y)))
    return (round(x, 2), round(y, 2))


def _mark_sent(x: float, y: float, when: float) -> None:
    global _last_sent_ts, _last_sent_signature, _fire_active
    _last_sent_ts = when
    _last_sent_signature = _signature_for(x, y)
    _fire_active = True


def _reset_fire_state() -> None:
    global _fire_active, _last_sent_signature
    _fire_active = False
    _last_sent_signature = None


def _dispatch_fire_event(x: float, y: float) -> None:
    now = time.time()
    sig = _signature_for(x, y)
    with _send_lock:
        should_send = (
            not _fire_active
            or sig != _last_sent_signature
            or (COOLDOWN_SEC > 0.0 and (now - _last_sent_ts) >= COOLDOWN_SEC)
        )
        if not should_send:
            return
        if _send_coords(x, y):
            _mark_sent(x, y, now)
        else:
            # 실패하면 다음 프레임에서 재시도할 수 있도록 상태를 초기화
            _reset_fire_state()


def notify_fire_fixed():
    _dispatch_fire_event(X_FIXED, Y_FIXED)


def _sanitize_fire_boxes(raw_boxes):
    out = []
    if not isinstance(raw_boxes, (list, tuple)):
        return out
    for b in raw_boxes:
        if not isinstance(b, (list, tuple)) or len(b) < 4:
            continue
        x1, y1, x2, y2 = b[0], b[1], b[2], b[3]
        try:
            x1f, y1f, x2f, y2f = float(x1), float(y1), float(x2), float(y2)
        except (TypeError, ValueError):
            continue
        if x2f <= x1f or y2f <= y1f:
            continue
        score = 0.0
        if len(b) >= 5:
            try:
                score = float(b[4])
            except (TypeError, ValueError):
                score = 0.0
        out.append((x1f, y1f, x2f, y2f, score))
    return out


# ── 브리지 안전 임포트 ───────────────────────────────────────────
try:
    import detector_bridge
    DETECTOR_FILE = getattr(detector_bridge, "__file__", None)
    detect_and_draw = getattr(detector_bridge, "detect_and_draw", None)
except Exception as e:
    detector_bridge = None
    DETECTOR_FILE = None
    detect_and_draw = None
    print(f"[BOOT] bridge import error: {e}")

# 부팅 로그
print(f"[BOOT] py={sys.executable} file={__file__}")
print(f"[BOOT] bridge_ok={callable(detect_and_draw)} bridge_file={DETECTOR_FILE}")

# ── 설정 ─────────────────────────────────────────────────────────
# 비지정 시 외장캠 자동 선택을 기본으로 한다.
# (명시하려면 CAM_PATH에 숫자 인덱스나 /dev/v4l/by-id 경로 지정)
CAM_PATH = os.environ.get("CAM_PATH", "")
WIDTH, HEIGHT, FPS = int(os.environ.get("WIDTH", 1280)), int(os.environ.get("HEIGHT", 720)), int(os.environ.get("FPS", 30))
JPEG_QUALITY = int(os.environ.get("JPEG_QUALITY", 80))

app = Flask(__name__)
cap = None
# 카메라 접근 경쟁을 막기 위한 전역 락 (멀티 클라이언트/스레드 안정화)
_cap_lock = threading.Lock()

# ── 로컬 UI 설정 ──────────────────────────────────────────────────
# UI_ENABLE: '1'이면 서버 실행과 동시에 로컬 미리보기 창 표시(디폴트: 1, 디스플레이 없으면 자동 비활성)
# UI_WITH_OVERLAY: '1'이면 detect/오버레이를 적용해 표시(추가 연산 발생). 기본 0.
# UI_SCALE: 창 크기 배율(기본 1.0)
UI_ENABLE = os.environ.get("UI_ENABLE", "1").lower() in ("1", "true", "yes", "on")
UI_WITH_OVERLAY = os.environ.get("UI_WITH_OVERLAY", "0").lower() in ("1", "true", "yes", "on")
try:
    UI_SCALE = float(os.environ.get("UI_SCALE", "1.0"))
except Exception:
    UI_SCALE = 1.0
_ui_stop = threading.Event()
_ui_thread = None

def _ui_env_ok() -> bool:
    # 디스플레이/하이구이 가용성 점검 및 안내
    disp_ok = bool(os.environ.get('DISPLAY') or os.name == 'nt' or os.environ.get('WAYLAND_DISPLAY'))
    if not disp_ok:
        print("[UI] No display environment detected (DISPLAY/WAYLAND_DISPLAY). UI disabled.")
        return False
    if not (hasattr(cv2, 'imshow') and hasattr(cv2, 'namedWindow')):
        print("[UI] OpenCV HighGUI not available (headless build). Install 'opencv-python' instead of 'opencv-python-headless'.")
        return False
    # 일부 환경에서 Qt 백엔드 필요
    os.environ.setdefault('QT_QPA_PLATFORM', 'xcb')
    return True

# ── 카메라 열기/정리 ─────────────────────────────────────────────
def open_cam():
    global cap
    with _cap_lock:
        if CAM_PATH.strip() == "":
            # 외장캠 자동 선택(-1): common_utils의 오토 피커 사용
            cap = _cu_open_camera(-1, WIDTH, HEIGHT, FPS)
            src_desc = "AUTO-EXT"
        else:
            src = int(CAM_PATH) if CAM_PATH.isdigit() else CAM_PATH
            cap = cv2.VideoCapture(src)
            # 가능하면 MJPG로
            try:
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            except Exception:
                pass
            try:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH,  WIDTH)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
                cap.set(cv2.CAP_PROP_FPS, FPS)
            except Exception:
                pass
            src_desc = str(src)
    opened = bool(cap and cap.isOpened())
    if opened and hasattr(cap, 'get'):
        try:
            w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            f = cap.get(cv2.CAP_PROP_FPS)
        except Exception:
            w = h = f = 0.0
    else:
        w = h = f = 0.0
    print(f"[CAM] open src={src_desc} opened={opened} "
          f"W/H/FPS={w:.0f}/{h:.0f}/{f:.0f}")

def cleanup():
    global cap
    with _cap_lock:
        try:
            if cap and cap.isOpened():
                cap.release()
        except Exception:
            pass
        finally:
            cap = None
    # UI 정리
    try:
        _ui_stop.set()
    except Exception:
        pass
    try:
        cv2.destroyAllWindows()
    except Exception:
        pass

atexit.register(cleanup)
open_cam()

def _read_frame() -> Tuple[bool, Optional[BGRImage]]:
    try:
        with _cap_lock:
            if cap is None or not cap.isOpened():
                return False, None
            ok, raw = cap.read()
    except Exception:
        return False, None
    if not ok or not isinstance(raw, np.ndarray):
        return False, None
    return True, raw

# ── 로컬 UI 스레드 ────────────────────────────────────────────────
def _ui_loop():
    win = "FireCam UI"
    # 디스플레이가 없으면 곧바로 종료
    if not _ui_env_ok():
        return
    try:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        print("[UI] window created")
    except Exception:
        print("[UI] failed to create window (HighGUI backend error).")
        return
    while not _ui_stop.is_set():
        ok, frame = _read_frame()
        if not ok or frame is None:
            time.sleep(0.05)
            continue
        show: BGRImage = frame.copy()
        try:
            if UI_WITH_OVERLAY:
                # 오버레이 포함 표시(전송 래치 덕에 중복 전송은 방지됨)
                show = process_frame(show)
            else:
                # 가벼운 워터마크만
                cv2.putText(show, "SRV-UI", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)
        except Exception:
            pass
        if UI_SCALE and UI_SCALE != 1.0:
            try:
                h, w = show.shape[:2]
                show = cv2.resize(show, (int(w * UI_SCALE), int(h * UI_SCALE)), interpolation=cv2.INTER_LINEAR)
            except Exception:
                pass
        try:
            cv2.imshow(win, show)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                _ui_stop.set()
                break
        except Exception:
            break

def _maybe_start_ui():
    global _ui_thread
    # 디스플레이 없거나 비활성 설정이면 시작하지 않음
    if not UI_ENABLE:
        return
    if not (os.environ.get('DISPLAY') or os.name == 'nt' or os.environ.get('WAYLAND_DISPLAY')):
        return
    if _ui_thread and _ui_thread.is_alive():
        return
    _ui_thread = threading.Thread(target=_ui_loop, name="ui-loop", daemon=True)
    _ui_thread.start()

# [ADD] 간단한 보조 휴리스틱: 빨강/주황 비율로 불빛 추정
def _looks_like_fire(bgr_frame: Optional[BGRImage]) -> bool:
    if bgr_frame is None:
        return False
    try:
        hsv = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2HSV)
        lower1 = cast(Any, np.array((0, 80, 80), dtype=np.uint8))
        upper1 = cast(Any, np.array((15, 255, 255), dtype=np.uint8))
        lower2 = cast(Any, np.array((160, 80, 80), dtype=np.uint8))
        upper2 = cast(Any, np.array((180, 255, 255), dtype=np.uint8))
        mask1 = cv2.inRange(hsv, lower1, upper1)
        mask2 = cv2.inRange(hsv, lower2, upper2)
        ratio = (mask1.mean() + mask2.mean()) / (2 * 255.0)
        return ratio > 0.08
    except Exception:
        return False

# ── 프레임 처리(워터마크 + 감지/오버레이) ─────────────────────────
def process_frame(frame: BGRImage) -> BGRImage:
    canvas: BGRImage = frame
    cv2.putText(canvas, "SRV", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)

    fire_boxes = []
    if callable(detect_and_draw):
        try:
            out = detect_and_draw(canvas)
            if isinstance(out, np.ndarray):
                canvas = out
            try:
                fire_boxes = _sanitize_fire_boxes(getattr(detector_bridge, "fire_boxes", []))
            except Exception:
                fire_boxes = []
        except Exception as e:
            print(f"[DETECT] error: {e}")

    if fire_boxes:
        try:
            notify_fire_fixed()
        except Exception:
            pass
    else:
        _reset_fire_state()

    if not fire_boxes and _looks_like_fire(canvas):
        try:
            cv2.putText(canvas, "FIRE?", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
        except Exception:
            pass

    return canvas


def _jpeg_params() -> Tuple[int, ...]:
    return (int(cv2.IMWRITE_JPEG_QUALITY), int(JPEG_QUALITY))

# ── 스트림 제너레이터(MJPEG) ────────────────────────────────────
def gen() -> Iterator[bytes]:
    period = 1.0 / max(FPS, 1)
    encode_params = _jpeg_params()
    while True:
        ok, frame = _read_frame()
        if not ok or frame is None:
            # 간단 복구 (busy 해소를 위해 약간의 백오프)
            cleanup()
            time.sleep(0.15)
            open_cam()
            time.sleep(0.05)
            continue
        processed = process_frame(frame)
        ok_enc, jpg = cv2.imencode('.jpg', processed, encode_params)
        if not ok_enc:
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpg.tobytes() + b'\r\n')
        time.sleep(period)

# ── 라우트 ───────────────────────────────────────────────────────
@app.route('/video_feed')
def video_feed():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/frame.jpg')
def frame_jpg():
    encode_params = _jpeg_params()
    ok, frame = _read_frame()
    if not ok or frame is None:
        cleanup(); time.sleep(0.15); open_cam()
        ok, frame = _read_frame()
        if not ok or frame is None:
            return make_response(b'', 503)
    processed = process_frame(frame)
    ok_enc, jpg = cv2.imencode('.jpg', processed, encode_params)
    if not ok_enc:
        return make_response(b'', 500)
    resp = make_response(jpg.tobytes())
    resp.headers['Content-Type'] = 'image/jpeg'
    resp.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    return resp

@app.route('/')
def index():
    return '<img src="/video_feed" />'

# (선택) 헬스체크
@app.route('/health')
def health():
    ok = (cap is not None) and cap.isOpened()
    return ("OK" if ok else "BAD"), (200 if ok else 503)

# ── 진입점 ───────────────────────────────────────────────────────
if __name__ == "__main__":
    # 설치: pip install flask opencv-python-headless numpy
    # 실행: CAM_PATH=0 python mjpeg_server.py
    _maybe_start_ui()
    app.run(host="0.0.0.0", port=8000, threaded=True)
