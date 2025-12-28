import cv2, time, os, json, numpy as np
from datetime import datetime
from src.capture_landmarks import (
    mp_holistic, _landmarks_to_array, normalize_frame, draw_overlay
)

# -------------------- utilidades de E/S --------------------
def ensure_dirs(label: str | None = None):
    os.makedirs("data/interim", exist_ok=True)
    if label:
        os.makedirs(os.path.join("data", "interim", label), exist_ok=True)

def put(img, text, xy=(15, 35)):
    cv2.putText(img, text, xy, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2, cv2.LINE_AA)

def save_sequence(label: str, seq: list[np.ndarray], fps: float, backend: str):
    """Guarda landmarks normalizados en npz + metadata.json en data/interim/<label>/"""
    ensure_dirs(label)
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    base = os.path.join("data", "interim", label, f"SEQ_{ts}")
    arr = np.stack(seq, axis=0).astype(np.float32)
    np.savez_compressed(base + ".npz", landmarks=np.ascontiguousarray(arr))

    meta = {
        "label": label,
        "frames": int(arr.shape[0]),
        "fps": float(fps),
        "backend": backend,
        "created_at": ts,
        "schema": "signrt-landmarks-v1"
    }
    with open(base + ".metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    return base + ".npz"

# -------------------- cámara (DSHOW + MJPG, estable en Windows) --------------------
def open_cam_dshow_mjpg(idx=0, w=640, h=480, fps=30):
    cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_FPS, fps)
    # warm-up breve
    ok = False
    for _ in range(6):
        ok, f = cap.read()
        if ok and f is not None:
            break
        time.sleep(0.03)
    return cap

# -------------------- app principal --------------------
def main():
    print("[INFO] Controles → R=grabar/pausar | S=guardar | L=etiqueta | M=toggle MediaPipe | ESC=salir")
    cv2.namedWindow("SignRT - Preview", cv2.WINDOW_NORMAL)

    cap = open_cam_dshow_mjpg(0, 640, 480, 30)
    if not cap or not cap.isOpened():
        print("[ERROR] No pude abrir la cámara con DSHOW+MJPG.")
        return

    backend_txt = f"DSHOW+MJPG  {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}@{int(cap.get(cv2.CAP_PROP_FPS))}"

    # Estado
    use_mediapipe = True
    recording = False
    label = "unlabeled"
    ensure_dirs(label)
    buffer_seq: list[np.ndarray] = []

    fps_disp, frames, t0 = 0.0, 0, time.time()

    # Parámetros de higiene de captura
    MIN_FRAMES_TO_SAVE = 8         # evita guardar capturas “vacías”
    MAX_FRAMES_PER_CLIP = 450      # ~15s a 30 fps, por seguridad

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

            # Procesamiento (solo en copia para no corromper el buffer crudo)
            frame_draw = frame.copy()
            results = None
            if use_mediapipe:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = holistic.process(rgb)
                draw_overlay(frame_draw, results)

            # HUD
            frames += 1
            now = time.time()
            if now - t0 >= 0.5:
                fps_disp = frames / (now - t0)
                frames, t0 = 0, now

            put(frame_draw, f"FPS ~ {fps_disp:.1f}", (15, 35))
            put(frame_draw, f"MediaPipe {'ON' if use_mediapipe else 'OFF'}", (15, 70))
            put(frame_draw, f"{'REC' if recording else 'PAUSE'}", (15, 105))
            put(frame_draw, f"Label: {label}", (15, 140))
            put(frame_draw, f"BE: {backend_txt}", (15, 175))
            if recording:
                put(frame_draw, f"Frames: {len(buffer_seq)}", (15, 210))

            # Acumulación de secuencia
            if use_mediapipe and recording:
                L = _landmarks_to_array(results)
                buffer_seq.append(normalize_frame(L))
                # corte de seguridad
                if len(buffer_seq) >= MAX_FRAMES_PER_CLIP:
                    print("[INFO] Máximo por clip alcanzado. Guardando…")
                    path = save_sequence(label, buffer_seq, fps_disp or 30.0, backend_txt)
                    print(f"[OK] Guardado {path}")
                    buffer_seq.clear()
                    recording = False

            # Mostrar
            cv2.imshow("SignRT - Preview", frame_draw)

            # Teclado
            k = cv2.waitKey(1) & 0xFF
            if k == 27:                   # ESC
                # guardar pendiente si corresponde
                if recording and len(buffer_seq) >= MIN_FRAMES_TO_SAVE:
                    path = save_sequence(label, buffer_seq, fps_disp or 30.0, backend_txt)
                    print(f"[OK] Guardado {path}")
                break
            elif k in (ord('m'), ord('M')):
                use_mediapipe = not use_mediapipe
            elif k in (ord('r'), ord('R')):
                recording = not recording
                if recording:
                    print(f"[INFO] Grabando… etiqueta='{label}'")
                else:
                    # al pausar, autoguarda si hay material suficiente
                    if len(buffer_seq) >= MIN_FRAMES_TO_SAVE:
                        path = save_sequence(label, buffer_seq, fps_disp or 30.0, backend_txt)
                        print(f"[OK] Guardado {path}")
                    else:
                        print("[WARN] Muy pocos frames; descartado.")
                    buffer_seq.clear()
            elif k in (ord('s'), ord('S')):
                if len(buffer_seq) >= MIN_FRAMES_TO_SAVE:
                    path = save_sequence(label, buffer_seq, fps_disp or 30.0, backend_txt)
                    print(f"[OK] Guardado {path}")
                    buffer_seq.clear()
                else:
                    print("[WARN] No hay datos suficientes para guardar.")
            elif k in (ord('l'), ord('L')):
                # cambiar etiqueta desde consola (bloqueante pero simple y robusto)
                print("\n>>> Nueva etiqueta (ej. 'hola', 'gracias', 'si_no'): ", end="", flush=True)
                try:
                    new_label = input().strip()
                except EOFError:
                    new_label = ""
                if new_label:
                    label = new_label
                    ensure_dirs(label)
                    print(f"[INFO] Etiqueta actual: {label}")
                else:
                    print("[INFO] Etiqueta sin cambios.")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
