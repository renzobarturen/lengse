# --------------------------- imports ---------------------------
import cv2, time, os, json, numpy as np             # OpenCV, tiempo, archivos, JSON y NumPy
from datetime import datetime                        # Timestamps legibles
from src.capture_landmarks import (                  # Utilidades basadas en MediaPipe Holistic
    mp_holistic, _landmarks_to_array, normalize_frame, draw_overlay
)

# -------------------- utilidades de E/S --------------------
def ensure_dirs(label: str | None = None):
    """Crea (si no existen) las carpetas donde guardaremos los datos."""
    os.makedirs("data/interim", exist_ok=True)                       # raíz de capturas intermedias
    if label:                                                        # si hay etiqueta, crea subcarpeta
        os.makedirs(os.path.join("data", "interim", label), exist_ok=True)

def put(img, text, xy=(15, 35)):
    """Escribe texto de HUD sobre la imagen (en verde)."""
    cv2.putText(img, text, xy, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2, cv2.LINE_AA)

def save_sequence(label: str, seq: list[np.ndarray], fps: float, backend: str):
    """
    Guarda una secuencia de landmarks ya normalizados como:
      - .npz con array 'landmarks'
      - .metadata.json con información auxiliar
    Estructura: data/interim/<label>/SEQ_YYYY-mm-dd_HHMMSS.*
    """
    ensure_dirs(label)                                              # asegura carpeta de la etiqueta
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")                 # timestamp legible
    base = os.path.join("data", "interim", label, f"SEQ_{ts}")     # ruta base sin extensión

    arr = np.stack(seq, axis=0).astype(np.float32)                  # (T, D) → float32
    np.savez_compressed(base + ".npz",                             # guarda comprimido
                        landmarks=np.ascontiguousarray(arr))

    # metadatos útiles para entrenamiento/reproducibilidad
    meta = {
        "label": label,                                            # etiqueta de la secuencia
        "frames": int(arr.shape[0]),                               # número de frames T
        "fps": float(fps),                                         # FPS estimado en captura
        "backend": backend,                                        # backend/FOURCC usado
        "created_at": ts,                                          # fecha de creación
        "schema": "signrt-landmarks-v1"                            # versión del esquema
    }
    with open(base + ".metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)           # guarda JSON legible

    return base + ".npz"                                           # devuelve la ruta del .npz

# -------- cámara (DirectShow + MJPG; suele ser estable en Windows) --------
def open_cam_dshow_mjpg(idx=0, w=640, h=480, fps=30):
    """Abre la cámara usando DSHOW con FOURCC MJPG y hace un warm-up corto."""
    cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)                     # backend DirectShow
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))  # MJPG para evitar NV12/YUY2
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  w)                          # ancho deseado
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)                          # alto deseado
    cap.set(cv2.CAP_PROP_FPS, fps)                                 # FPS objetivo

    # Warm-up: algunas cámaras devuelven frames inválidos al inicio
    ok = False
    for _ in range(6):
        ok, f = cap.read()                                         # lee un frame
        if ok and f is not None:                                   # si es válido, salimos
            break
        time.sleep(0.03)                                           # pequeño delay entre intentos
    return cap                                                     # devolvemos el objeto VideoCapture

# ------------------------------ app principal ------------------------------
def main():
    """Visor + capturador etiquetado de landmarks con MediaPipe Holistic."""
    print("[INFO] Controles → R=grabar/pausar | S=guardar | L=etiqueta | M=toggle MediaPipe | ESC=salir")
    cv2.namedWindow("SignRT - Preview", cv2.WINDOW_NORMAL)         # ventana redimensionable

    cap = open_cam_dshow_mjpg(0, 640, 480, 30)                     # abre cámara 0 en 640x480@30
    if not cap or not cap.isOpened():                              # valida apertura
        print("[ERROR] No pude abrir la cámara con DSHOW+MJPG.")
        return

    # Texto del backend para mostrar en pantalla (toma valores reales de la cámara)
    backend_txt = f"DSHOW+MJPG  {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}@{int(cap.get(cv2.CAP_PROP_FPS))}"

    # -------- estado de la sesión --------
    use_mediapipe = True                                           # activar/desactivar Holistic
    recording = False                                              # modo grabación
    label = "unlabeled"                                            # etiqueta por defecto
    ensure_dirs(label)                                             # crea carpeta 'unlabeled'
    buffer_seq: list[np.ndarray] = []                              # buffer de landmarks (lista de frames)

    fps_disp, frames, t0 = 0.0, 0, time.time()                     # cálculo de FPS para HUD

    # Parámetros de higiene para clips (evitan basura)
    MIN_FRAMES_TO_SAVE = 8                                         # mínimo para guardar
    MAX_FRAMES_PER_CLIP = 450                                      # corte de seguridad (~15s @30fps)

    # Crea el detector Holistic (modelo “equilibrado”, sin segmentación ni refinado facial)
    with mp_holistic.Holistic(model_complexity=1, enable_segmentation=False, refine_face_landmarks=False) as holistic:
        while True:                                                # bucle principal de la app
            ok, frame = cap.read()                                 # captura un frame de la cámara
            if not ok or frame is None:                            # si falla lectura…
                print("[WARN] Frame nulo; reabriendo cámara…")
                cap.release()                                      # suelta el dispositivo
                cap = open_cam_dshow_mjpg(0, 640, 480, 30)         # intenta reabrir
                if not cap or not cap.isOpened():                  # si no pudo, salimos
                    break
                continue                                           # intenta en la siguiente iteración

            # --- Procesamiento (sobre una COPIA para no tocar el frame crudo) ---
            frame_draw = frame.copy()                              # copia para overlay
            results = None
            if use_mediapipe:                                      # si está activado, corre Holistic
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)       # MediaPipe espera RGB
                results = holistic.process(rgb)                     # inferencia
                draw_overlay(frame_draw, results)                  # dibuja cara/manos/pose

            # --- HUD (texto informativo) ---
            frames += 1                                            # acumula frames para FPS
            now = time.time()
            if now - t0 >= 0.5:                                    # actualiza cada ~0.5s
                fps_disp = frames / (now - t0)                     # FPS estimado
                frames, t0 = 0, now

            put(frame_draw, f"FPS ~ {fps_disp:.1f}", (15, 35))     # muestra FPS
            put(frame_draw, f"MediaPipe {'ON' if use_mediapipe else 'OFF'}", (15, 70))
            put(frame_draw, f"{'REC' if recording else 'PAUSE'}", (15, 105))
            put(frame_draw, f"Label: {label}", (15, 140))          # etiqueta actual
            put(frame_draw, f"BE: {backend_txt}", (15, 175))       # backend/cfg de cámara
            if recording:
                put(frame_draw, f"Frames: {len(buffer_seq)}", (15, 210))  # tamaño del clip

            # --- Acumulación de landmarks durante la grabación ---
            if use_mediapipe and recording:                        # solo si MP está ON y estamos grabando
                L = _landmarks_to_array(results)                   # convierte resultados → vector fijo
                buffer_seq.append(normalize_frame(L))              # normaliza y añade al buffer

                # Corte de seguridad: evita clips demasiado largos
                if len(buffer_seq) >= MAX_FRAMES_PER_CLIP:
                    print("[INFO] Máximo por clip alcanzado. Guardando…")
                    path = save_sequence(label, buffer_seq, fps_disp or 30.0, backend_txt)
                    print(f"[OK] Guardado {path}")
                    buffer_seq.clear()                             # limpia buffer
                    recording = False                              # vuelve a pausa

            # --- Mostrar en pantalla ---
            cv2.imshow("SignRT - Preview", frame_draw)             # enseña el cuadro con overlay

            # --- Lectura de teclado ---
            k = cv2.waitKey(1) & 0xFF                              # lee tecla (si hay)
            if k == 27:                                            # ESC → salir
                # si salimos grabando, intenta guardar lo acumulado si es suficiente
                if recording and len(buffer_seq) >= MIN_FRAMES_TO_SAVE:
                    path = save_sequence(label, buffer_seq, fps_disp or 30.0, backend_txt)
                    print(f"[OK] Guardado {path}")
                break                                              # sale del bucle

            elif k in (ord('m'), ord('M')):                        # M → alternar MediaPipe
                use_mediapipe = not use_mediapipe

            elif k in (ord('r'), ord('R')):                        # R → grabar/pausar
                recording = not recording
                if recording:
                    print(f"[INFO] Grabando… etiqueta='{label}'")
                else:
                    # al pausar, autoguarda si hay frames suficientes
                    if len(buffer_seq) >= MIN_FRAMES_TO_SAVE:
                        path = save_sequence(label, buffer_seq, fps_disp or 30.0, backend_txt)
                        print(f"[OK] Guardado {path}")
                    else:
                        print("[WARN] Muy pocos frames; descartado.")
                    buffer_seq.clear()                             # limpia para el siguiente clip

            elif k in (ord('s'), ord('S')):                        # S → guardar manual
                if len(buffer_seq) >= MIN_FRAMES_TO_SAVE:
                    path = save_sequence(label, buffer_seq, fps_disp or 30.0, backend_txt)
                    print(f"[OK] Guardado {path}")
                    buffer_seq.clear()
                else:
                    print("[WARN] No hay datos suficientes para guardar.")

            elif k in (ord('l'), ord('L')):                        # L → cambiar etiqueta
                # lectura por consola (bloqueante pero robusta)
                print("\n>>> Nueva etiqueta (ej. 'hola', 'gracias', 'si_no'): ", end="", flush=True)
                try:
                    new_label = input().strip()
                except EOFError:
                    new_label = ""
                if new_label:                                      # si el usuario escribió algo
                    label = new_label
                    ensure_dirs(label)                             # crea carpeta si no existía
                    print(f"[INFO] Etiqueta actual: {label}")
                else:
                    print("[INFO] Etiqueta sin cambios.")

    # Limpieza de recursos al salir del bucle
    cap.release()                                                  # libera la cámara
    cv2.destroyAllWindows()                                        # cierra la(s) ventana(s)

# Punto de entrada estándar de Python
if __name__ == "__main__":
    main()
