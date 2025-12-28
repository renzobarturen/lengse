import os
import time
from collections import deque

import cv2
import numpy as np
import torch
import torch.nn as nn

# Reutilizamos tu pipeline de landmarks (exactamente el mismo que en captura)
from src.capture_landmarks import (
    mp_holistic, _landmarks_to_array, normalize_frame, draw_overlay
)

# ----------------------------
# Modelo (debe coincidir con el entrenado)
# ----------------------------
class BiLSTMClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, num_classes: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)     # (B, T, 2H)
        last = out[:, -1, :]      # último frame
        last = self.dropout(last)
        return self.fc(last)      # (B, C)


# ----------------------------
# Cámara (idéntica a tu configuración estable)
# ----------------------------
def open_cam_dshow_mjpg(idx=0, w=640, h=480, fps=30):
    cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_FPS, fps)

    # warm-up
    for _ in range(10):
        ok, f = cap.read()
        if ok and f is not None:
            break
        time.sleep(0.03)
    return cap


def put(img, text, xy=(15, 35), scale=0.8):
    cv2.putText(img, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, (0,255,0), 2, cv2.LINE_AA)


def _frame_to_flat(frame_feat: np.ndarray) -> np.ndarray:
    """
    Normaliza la forma del frame:
      - si viene (J, C) -> lo aplana a (J*C,)
      - si ya viene (D,) -> lo deja igual
    """
    if frame_feat is None:
        return None
    if frame_feat.ndim == 2:
        return frame_feat.reshape(-1)
    if frame_feat.ndim == 1:
        return frame_feat
    # cualquier otro caso es inesperado
    return frame_feat.reshape(-1)


def load_checkpoint(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)

    input_dim = int(ckpt["input_dim"])
    num_classes = int(ckpt["num_classes"])
    hidden_dim = int(ckpt["hidden_dim"])
    num_layers = int(ckpt["num_layers"])
    dropout = float(ckpt["dropout"])

    label_map = ckpt.get("label_map", None)
    if label_map is None:
        raise RuntimeError("El checkpoint no contiene 'label_map'. Reentrena guardando label_map en el .pt")

    # idx_to_label puede venir como dict con keys string -> convertimos a int
    idx_to_label = label_map.get("idx_to_label", {})
    idx_to_label_int = {}
    for k, v in idx_to_label.items():
        try:
            idx_to_label_int[int(k)] = v
        except Exception:
            pass
    if idx_to_label_int:
        idx_to_label = idx_to_label_int

    model = BiLSTMClassifier(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_classes=num_classes,
        dropout=dropout
    ).to(device)

    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, idx_to_label, input_dim


def main():
    # Ruta por defecto: el mejor modelo que vienes guardando
    ckpt_path = os.path.join("models", "signrt_bilstm_best.pt")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"No existe el checkpoint: {ckpt_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] Device:", device)

    model, idx_to_label, input_dim = load_checkpoint(ckpt_path, device)

    # IMPORTANTE:
    # El dataset actual muestra T=45 en entrenamiento.
    # Para inferencia, usamos esa misma T. Si cambia T en build_dataset.py,
    # se debe actualizar este valor para que coincida.
    T_WINDOW = 45

    # Buffer temporal: guardamos frames (D,) y luego apilamos a (T, D)
    buf = deque(maxlen=T_WINDOW)

    # Suavizado de predicción para evitar “parpadeo”
    ema = None
    EMA_ALPHA = 0.25  # 0.1 más suave, 0.3 más reactivo

    print("[INFO] Controles → ESC=salir | R=reset buffer")
    cap = open_cam_dshow_mjpg(0, 640, 480, 30)
    if not cap or not cap.isOpened():
        raise RuntimeError("No se pudo abrir la cámara.")

    cv2.namedWindow("SignRT - Realtime", cv2.WINDOW_NORMAL)

    with mp_holistic.Holistic(model_complexity=1, enable_segmentation=False, refine_face_landmarks=False) as holistic:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                put(frame, "Frame nulo; reintentando...", (15, 35))
                time.sleep(0.05)
                continue

            # MediaPipe: siempre procesa en RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(rgb)

            # Dibujamos overlay (puntos del rostro/manos/pose)
            frame_draw = frame.copy()
            draw_overlay(frame_draw, results)

            # Extraer landmarks con tu mismo pipeline
            L = _landmarks_to_array(results)          # típicamente (543, 4)
            F = normalize_frame(L)                    # debe ser consistente con lo entrenado
            F = _frame_to_flat(np.array(F))           # -> (D,) donde D debe ser 2172

            if F is not None:
                # Validación de dimensión
                if F.shape[0] != input_dim:
                    put(frame_draw, f"ERROR D={F.shape[0]} != input_dim={input_dim}", (15, 70), 0.7)
                else:
                    buf.append(F)

            # Solo predecimos cuando el buffer está lleno (T_WINDOW frames)
            pred_txt = "Pred: (llenando buffer...)"
            if len(buf) == T_WINDOW:
                x = np.stack(list(buf), axis=0).astype(np.float32)  # (T, D)
                x_t = torch.from_numpy(x).unsqueeze(0).to(device)   # (1, T, D)

                with torch.no_grad():
                    logits = model(x_t)                   # (1, C)
                    probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

                # EMA para estabilizar
                if ema is None:
                    ema = probs
                else:
                    ema = (1 - EMA_ALPHA) * ema + EMA_ALPHA * probs

                top = np.argsort(-ema)[:3]
                top1 = int(top[0])
                top1_label = idx_to_label.get(top1, str(top1))
                top1_conf = float(ema[top1])

                top3_str = " | ".join([f"{idx_to_label.get(int(i), i)}:{ema[int(i)]:.2f}" for i in top])
                pred_txt = f"Pred: {top1_label}  conf={top1_conf:.2f}"
                put(frame_draw, pred_txt, (15, 35), 0.9)
                put(frame_draw, f"Top3: {top3_str}", (15, 70), 0.7)
            else:
                put(frame_draw, pred_txt, (15, 35), 0.9)
                put(frame_draw, f"Buffer: {len(buf)}/{T_WINDOW}", (15, 70), 0.7)

            cv2.imshow("SignRT - Realtime", frame_draw)
            k = cv2.waitKey(1) & 0xFF
            if k == 27:  # ESC
                break
            elif k in (ord('r'), ord('R')):
                buf.clear()
                ema = None
                print("[INFO] Buffer reiniciado.")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
