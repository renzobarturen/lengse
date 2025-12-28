import cv2, time, os, numpy as np
from datetime import datetime
from src.capture_landmarks import mp_holistic, _landmarks_to_array, normalize_frame, draw_overlay

def ensure_dirs():
    os.makedirs("data/interim", exist_ok=True)

def put(img, text, xy=(15,35)):
    cv2.putText(img, text, xy, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2, cv2.LINE_AA)

def _try_open(backend, idx=0, w=640, h=480):
    """Abre cámara SIN FOURCC, con conversión RGB activada, como hace la app de Cámara."""
    cap = cv2.VideoCapture(idx, backend)
    # No fijamos FOURCC ni FPS (dejamos defaults del driver)
    cap.set(cv2.CAP_PROP_CONVERT_RGB, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    # warm-up corto
    for _ in range(6):
        ok, f = cap.read()
        if ok and f is not None:
            return cap
        time.sleep(0.03)
    cap.release()
    return None

def open_cam_dshow_mjpg(idx=0, w=640, h=480, fps=30):
    cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
    # Igual que tu test: sólo FOURCC MJPG + resolución
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_FPS, fps)
    # NO usar CAP_PROP_CONVERT_RGB aquí
    # Warm-up corto
    for _ in range(6):
        ok, f = cap.read()
        if ok and f is not None:
            break
        time.sleep(0.03)
    return cap


def _to_bgr_if_needed(frame):
    """Si el frame no viene en BGR (3 canales), intentamos NV12 y YUY2."""
    if frame is None:
        return None
    if len(frame.shape) == 3 and frame.shape[2] == 3:
        return frame  # ya está en BGR
    # Algunos drivers entregan NV12 o YUY2 cuando no fijamos FOURCC.
    # Intentamos NV12 primero (frecuente con MSMF).
    try:
        return cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_NV12)
    except Exception:
        pass
    # Intento YUY2
    try:
        return cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_YUY2)
    except Exception:
        pass
    return frame  # devolvemos tal cual si no se pudo convertir

def main():
    ensure_dirs()
    print("[INFO] R=grabar/pausar | S=guardar | M=toggle MediaPipe | ESC=salir")
    cv2.namedWindow("SignRT - Preview", cv2.WINDOW_NORMAL)

    cap = open_cam_dshow_mjpg(0, 640, 480, 30)
    if not cap or not cap.isOpened():
        print("[ERROR] No pude abrir la cámara con DSHOW+MJPG como en el test.")
        return

    use_mediapipe = True
    recording = False
    buffer_seq = []
    fps_disp, frames, t0 = 0.0, 0, time.time()

    with mp_holistic.Holistic(model_complexity=1, enable_segmentation=False, refine_face_landmarks=False) as holistic:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("[WARN] Frame nulo; reabriendo cámara…")
                cap.release()
                cap = open_cam_dshow_mjpg(0, 640, 480, 30)
                if not cap or not cap.isOpened():
                    break
                continue

            # NO convertir aquí; mostrar crudo tal cual driver lo entrega
            # Sólo para MediaPipe convertimos a RGB en un buffer aparte
            if use_mediapipe:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = holistic.process(rgb)
                # Dibujamos sobre una copia para no corromper el buffer original
                frame_draw = frame.copy()
                draw_overlay(frame_draw, results)
            else:
                frame_draw = frame

            # HUD / FPS
            frames += 1
            now = time.time()
            if now - t0 >= 0.5:
                fps_disp = frames / (now - t0)
                frames, t0 = 0, now
            put(frame_draw, f"FPS ~ {fps_disp:.1f}", (15,35))
            put(frame_draw, "MediaPipe ON" if use_mediapipe else "MediaPipe OFF", (15,65))
            put(frame_draw, "REC" if recording else "PAUSE", (15,95))
            put(frame_draw, "BE: DSHOW+MJPG  640x480@30", (15,125))

            if use_mediapipe and recording:
                L = _landmarks_to_array(results)
                buffer_seq.append(normalize_frame(L))

            cv2.imshow("SignRT - Preview", frame_draw)
            k = cv2.waitKey(1) & 0xFF
            if k == 27:  # ESC
                break
            elif k in (ord('m'), ord('M')):
                use_mediapipe = not use_mediapipe
            elif k in (ord('r'), ord('R')):
                recording = not recording
                print("[INFO] Grabando…" if recording else f"[INFO] Pausado. T={len(buffer_seq)}")
            elif k in (ord('s'), ord('S')):
                if buffer_seq:
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    np.savez_compressed(f"data/interim/SEQ_{ts}.npz",
                                        landmarks=np.stack(buffer_seq).astype(np.float32))
                    print(f"[OK] Guardado data/interim/SEQ_{ts}.npz")
                    buffer_seq.clear()


if __name__ == "__main__":
    main()
