import socket, time, subprocess, threading, queue, re, sys, os

HOST = "192.168.0.11"
PORT = 8888

# === 고정 좌표 및 전송 정책 ===
X_FIXED = 1.00
Y_FIXED = 1.00
THRESHOLD = 1.0
COOLDOWN_SEC = 120.0  # 같은 화재를 너무 자주 보내지 않도록 쿨다운

# === 카메라 스크립트 경로/명령 ===
# 너의 폴더 구조에 맞춰 경로만 확인해줘.
CAMERA_PY = os.path.join(os.path.dirname(__file__), "camera_flame_yolov8_redsafe.py")
# 무버퍼 실행(-u)로 실시간성 강화. 필요 옵션(--cam/--device/--weights 등)은 뒤에 추가 가능.
CAMERA_CMD = [
    sys.executable, "-u", CAMERA_PY,
    "--cam", "2",
    "--width", "640", "--height", "480", "--fps", "30",
]

# camera stdout에서 "FIRE x y" 형태를 읽는다.
FIRE_RE = re.compile(r"^\s*FIRE\s+(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s*$")

_cam_q: "queue.Queue[tuple[bool, float|None, float|None]]" = queue.Queue()

def should_send(x, y):
    return (x >= THRESHOLD) and (y >= THRESHOLD)

def send_coords(sock, x, y):
    msg = f"{x:.2f},{y:.2f}\n"
    sock.sendall(msg.encode("utf-8"))
    print("[SEND]", msg.strip())

def send_fixed_coords(sock):
    if should_send(X_FIXED, Y_FIXED):
        send_coords(sock, X_FIXED, Y_FIXED)
    else:
        print("[SKIP] 기준 미달 (x,y 둘 다 1.0m 이상이어야 전송)")

# === 카메라 프로세스 실행 & 출력 파서 ===
def _camera_reader(proc: subprocess.Popen):
    # 안전 가드: stdout이 None이면 바로 종료 신호
    if proc.stdout is None:
        print("[ERR] camera process has no stdout (PIPE 미설정?)")
        _cam_q.put((False, None, None))
        return

    # stdout 핸들을 지역 변수로 고정 (타입체커 경고 제거)
    pipe = proc.stdout

    # 카메라 프로세스 stdout을 줄 단위로 읽어서 큐에 넣는다.
    for raw in iter(pipe.readline, ""):
        line = raw.strip()
        if not line:
            continue
        m = FIRE_RE.match(line)
        if m:
            try:
                x = float(m.group(1))
                y = float(m.group(2))
                _cam_q.put((True, x, y))
                print(f"[CAM] FIRE {x:.2f} {y:.2f}")
            except ValueError:
                # 숫자 파싱 실패시 무시
                pass
        # 필요하면 디버그 로그 확인용:
        # else:
        #     print("[CAM-LOG]", line)

    # 프로세스가 끝나면 종료 신호 한 번
    _cam_q.put((False, None, None))
    print("[CAM] camera process ended")

def start_camera():
    # 카메라 스크립트를 서브프로세스로 실행 (stdout 파이프)
    proc = subprocess.Popen(
        CAMERA_CMD,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,            # (가능하면) 라인 버퍼링
        encoding="utf-8",
        errors="ignore",
    )
    t = threading.Thread(target=_camera_reader, args=(proc,), daemon=True)
    t.start()
    return proc

# === 감지 1회 폴링 ===
def detect_fire_once():
    # 카메라 리더가 큐에 넣어준 최근 이벤트를 non-blocking으로 꺼낸다.
    try:
        detected, x, y = _cam_q.get_nowait()
        return detected, x, y
    except queue.Empty:
        return False, None, None

def main():
    # 1) 카메라 먼저 실행
    # if not os.path.exists(CAMERA_PY):
    #     print(f"[ERR] 카메라 스크립트를 찾을 수 없습니다: {CAMERA_PY}")
    #     return
    # cam_proc = start_camera()
    # print("[OK] Camera process started")

    # 2) 서버 연결
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))
        print(f"[OK] Connected to {HOST}:{PORT}")
        print("하트비트 모드: 고정 좌표를 주기 전송")

        last_sent = 0.0
        try:
            while True:
                # detected, x, y = detect_fire_once()

                # 감지 없이 주기적으로 고정 좌표 하트비트 전송
                now = time.time()
                if now - last_sent >= COOLDOWN_SEC:
                    send_coords(s, X_FIXED, Y_FIXED)
                    # (선택) 서버 응답 짧게 대기
                    s.settimeout(0.2)
                    try:
                        data = s.recv(4096)
                        if data:
                            print("[RECV]", data.decode("utf-8", errors="ignore").rstrip())
                    except socket.timeout:
                        pass
                    finally:
                        s.settimeout(None)
                    last_sent = now
                else:
                    # 너무 자주 보내지 않도록 대기
                    pass

                time.sleep(0.02)  # 폴링 간격

        except KeyboardInterrupt:
            print("\n[INFO] KeyboardInterrupt: shutting down…")
        except (BrokenPipeError, ConnectionResetError) as e:
            print("[WARN] 서버 연결 끊김. 프로그램 종료:", e)
        # finally:
            # # 카메라 프로세스도 정리
            # if cam_proc and (cam_proc.poll() is None):
            #     cam_proc.terminate()
            #     try:
            #         cam_proc.wait(timeout=2)
            #     except Exception:
            #         cam_proc.kill()

if __name__ == "__main__":
    main()

