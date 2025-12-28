import os
import json
import glob
import numpy as np
from datetime import datetime

# Parámetros del dataset
INTERIM_DIR = "data/interim"
OUT_DIR = "data/processed"
TRAIN_RATIO = 0.8          # 80% train, 20% test
TARGET_LEN = 45            # número de frames por secuencia (padding / truncado)
MIN_FRAMES = 5             # mínimo de frames para aceptar un clip


def list_label_dirs(base_dir):
    """Lista todas las subcarpetas dentro de data/interim como etiquetas."""
    labels = []
    for name in os.listdir(base_dir):
        path = os.path.join(base_dir, name)
        if os.path.isdir(path):
            labels.append(name)
    labels.sort()
    return labels


def load_sequences_for_label(label):
    """
    Carga todas las secuencias (.npz) para una etiqueta dada.
    Soporta:
      - (T, D)           ya aplanado
      - (T, J, C)        por ejemplo (T, 543, 4)
    Devuelve lista de arrays (T, D_flat).
    """
    pattern = os.path.join(INTERIM_DIR, label, "SEQ_*.npz")
    paths = sorted(glob.glob(pattern))
    seqs: list[np.ndarray] = []

    for p in paths:
        data = np.load(p)

        # 1) Verificar que exista la clave 'landmarks'
        if "landmarks" not in data.files:
            print(f"[WARN] {p} no tiene clave 'landmarks'; omitido.")
            continue

        arr = data["landmarks"]  # puede ser (T, D) o (T, J, C)

        # 2) Normalizar forma a (T, D_flat)
        if arr.ndim == 3:
            # (T, J, C)  ->  (T, J*C)
            T, J, C = arr.shape
            arr2 = arr.reshape(T, J * C)
        elif arr.ndim == 2:
            # ya está como (T, D)
            arr2 = arr
        else:
            print(f"[WARN] {p} tiene ndim={arr.ndim}; shape={arr.shape}; omitido.")
            continue

        # 3) Filtro por mínimo de frames
        if arr2.shape[0] < MIN_FRAMES:
            print(f"[WARN] {p} tiene pocos frames (T={arr2.shape[0]}); omitido.")
            continue

        seqs.append(arr2.astype(np.float32))

    return seqs



def pad_or_truncate(seq, target_len):
    """
    Ajusta una secuencia (T, D) a longitud fija target_len.
    - Si T > target_len: toma los primeros target_len frames.
    - Si T < target_len: hace padding al final con ceros.
    """
    T, D = seq.shape
    if T == target_len:
        return seq

    if T > target_len:
        return seq[:target_len, :]

    # padding
    padded = np.zeros((target_len, D), dtype=np.float32)
    padded[:T, :] = seq
    return padded


def build_dataset():
    # 1. Descubrir etiquetas
    if not os.path.isdir(INTERIM_DIR):
        raise FileNotFoundError(f"No existe la carpeta {INTERIM_DIR}")

    labels = list_label_dirs(INTERIM_DIR)
    if not labels:
        raise RuntimeError(f"No se encontraron subcarpetas de etiquetas dentro de {INTERIM_DIR}")

    print(f"[INFO] Etiquetas encontradas: {labels}")

    # Mapa etiqueta → índice numérico
    label_to_idx = {lab: i for i, lab in enumerate(labels)}
    idx_to_label = {i: lab for lab, i in label_to_idx.items()}

    all_X = []   # lista de (T, D) ya ajustados a TARGET_LEN
    all_y = []   # lista de índices de etiqueta

    # 2. Cargar y normalizar secuencias por etiqueta
    for label in labels:
        seqs = load_sequences_for_label(label)
        print(f"[INFO] {label}: {len(seqs)} secuencias válidas.")
        for seq in seqs:
            # aseguramos que todas tienen la misma D
            T, D = seq.shape
            if T < MIN_FRAMES:
                continue
            fixed = pad_or_truncate(seq, TARGET_LEN)
            all_X.append(fixed)
            all_y.append(label_to_idx[label])

    if not all_X:
        raise RuntimeError("No se cargó ninguna secuencia válida. Verifica data/interim.")

    X = np.stack(all_X, axis=0)     # (N, T, D)
    y = np.array(all_y, dtype=np.int64)  # (N,)

    N, T, D = X.shape
    print(f"[INFO] Dataset completo: N={N}, T={T}, D={D}, clases={len(labels)}")

    # 3. Barajar y partir en train/test
    rng = np.random.default_rng(seed=42)  # semilla para reproducibilidad
    indices = np.arange(N)
    rng.shuffle(indices)

    X = X[indices]
    y = y[indices]

    n_train = int(TRAIN_RATIO * N)
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

    print(f"[INFO] Split: train={X_train.shape[0]}, test={X_test.shape[0]}")

    # 4. Crear carpeta de salida
    os.makedirs(OUT_DIR, exist_ok=True)

    # Timestamp para versionar datasets
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    base_train = os.path.join(OUT_DIR, f"signrt_train_{ts}")
    base_test  = os.path.join(OUT_DIR, f"signrt_test_{ts}")

    # 5. Guardar .npz con dataset
    np.savez_compressed(base_train + ".npz",
                        X=X_train.astype(np.float32),
                        y=y_train.astype(np.int64))
    np.savez_compressed(base_test + ".npz",
                        X=X_test.astype(np.float32),
                        y=y_test.astype(np.int64))

    print(f"[OK] Guardado train: {base_train}.npz")
    print(f"[OK] Guardado test : {base_test}.npz")

    # 6. Guardar mapping de etiquetas
    label_map = {
        "label_to_idx": label_to_idx,
        "idx_to_label": idx_to_label,
        "target_len": TARGET_LEN,
        "min_frames": MIN_FRAMES
    }
    with open(os.path.join(OUT_DIR, f"label_map_{ts}.json"), "w", encoding="utf-8") as f:
        json.dump(label_map, f, indent=2, ensure_ascii=False)

    print(f"[OK] Guardado mapa de etiquetas en label_map_{ts}.json")


if __name__ == "__main__":
    build_dataset()
