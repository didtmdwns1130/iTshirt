print("[BRIDGE] using camera_flame_yolov8_redsafe.detect_and_draw")
import camera_flame_yolov8_redsafe as _cam

def detect_and_draw(frame):
    out = _cam.detect_and_draw(frame)
    # 단일 프레임 경로의 상태를 bridge 전역으로 노출
    try:
        globals()['fire_flag'] = bool(getattr(_cam, 'last_fire_detected_sf', False))
        globals()['fire_center'] = getattr(_cam, 'last_fire_center_sf', None)
        globals()['fire_boxes'] = list(getattr(_cam, 'last_fire_boxes_sf', []))
    except Exception:
        globals()['fire_flag'] = False
        globals()['fire_center'] = None
        globals()['fire_boxes'] = []
    return out
