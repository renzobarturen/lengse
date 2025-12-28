import os
import glob
import json
import argparse
from dataclasses import dataclass
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# -------------------------
# Utilidades: localizar últimos archivos generados
# -------------------------
def _latest_file(pattern: str) -> str:
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No encontré archivos con patrón: {pattern}")
    files.sort(key=os.path.getmtime, reverse=True)
    return files[0]


def resolve_paths(data_dir="data/processed", train_path=None, test_path=None, label_map_path=None):
    if train_path is None:
        train_path = _latest_file(os.path.join(data_dir, "signrt_train_*.npz"))
    if test_path is None:
        test_path = _latest_file(os.path.join(data_dir, "signrt_test_*.npz"))
    if label_map_path is None:
        label_map_path = _latest_file(os.path.join(data_dir, "label_map_*.json"))
    return train_path, test_path, label_map_path


# -------------------------
# Dataset
# -------------------------
class NpzSeqDataset(Dataset):
    def __init__(self, npz_path: str):
        z = np.load(npz_path)
        self.X = z["X"].astype(np.float32)  # (N, T, D)
        self.y = z["y"].astype(np.int64)    # (N,)
        if self.X.ndim != 3:
            raise ValueError(f"X debe ser 3D (N,T,D). shape={self.X.shape}")
        if self.y.ndim != 1:
            raise ValueError(f"y debe ser 1D (N,). shape={self.y.shape}")

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.tensor(self.y[idx])



# -------------------------
# Modelo: BiLSTM baseline
# -------------------------
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
        self.fc = nn.Linear(hidden_dim * 2, num_classes)  # *2 por bidirectional

    def forward(self, x):
        # x: (B, T, D)
        out, _ = self.lstm(x)          # out: (B, T, 2H)
        last = out[:, -1, :]           # tomamos el último frame (simple y efectivo)
        last = self.dropout(last)
        logits = self.fc(last)         # (B, C)
        return logits


# -------------------------
# Métricas y loops
# -------------------------
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0
    ce = nn.CrossEntropyLoss()

    for X, y in loader:

        X = X.to(device)
        y = y.to(device).long()


        logits = model(X)
        loss = ce(logits, y)

        pred = torch.argmax(logits, dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
        loss_sum += loss.item() * y.size(0)

    return loss_sum / max(total, 1), correct / max(total, 1)


def train_one_epoch(model, loader, optimizer, device, class_weights=None):
    model.train()

    if class_weights is not None:
        ce = nn.CrossEntropyLoss(weight=class_weights.to(device))
    else:
        ce = nn.CrossEntropyLoss()

    total = 0
    correct = 0
    loss_sum = 0.0

    for X, y in loader:
 

        X = X.to(device)
        y = y.to(device)


        optimizer.zero_grad(set_to_none=True)
        logits = model(X)
        loss = ce(logits, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        pred = torch.argmax(logits, dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
        loss_sum += loss.item() * y.size(0)

    return loss_sum / max(total, 1), correct / max(total, 1)


def compute_class_weights(y: np.ndarray, num_classes: int):
    # pesos inversamente proporcionales a la frecuencia por clase
    counts = np.bincount(y, minlength=num_classes).astype(np.float32)
    counts[counts == 0] = 1.0
    w = 1.0 / counts
    w = w / w.sum() * num_classes
    return torch.tensor(w, dtype=torch.float32)


@dataclass
class TrainConfig:
    epochs: int = 30
    batch_size: int = 16
    lr: float = 1e-3
    hidden_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.2
    use_class_weights: bool = True
    num_workers: int = 0
    seed: int = 42
    out_dir: str = "models"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/processed")
    ap.add_argument("--train", default=None)
    ap.add_argument("--test", default=None)
    ap.add_argument("--label_map", default=None)

    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden_dim", type=int, default=128)
    ap.add_argument("--num_layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--no_class_weights", action="store_true")
    ap.add_argument("--out_dir", default="models")

    args = ap.parse_args()

    # Reproducibilidad
    torch.manual_seed(42)
    np.random.seed(42)

    train_path, test_path, label_map_path = resolve_paths(
        data_dir=args.data_dir,
        train_path=args.train,
        test_path=args.test,
        label_map_path=args.label_map
    )

    print("[INFO] Train:", train_path)
    print("[INFO] Test :", test_path)
    print("[INFO] Map  :", label_map_path)

    with open(label_map_path, "r", encoding="utf-8") as f:
        lm = json.load(f)

    num_classes = len(lm["label_to_idx"])
    print("[INFO] Clases:", num_classes)

    ds_tr = NpzSeqDataset(train_path)
    ds_te = NpzSeqDataset(test_path)

    N, T, D = ds_tr.X.shape
    print(f"[INFO] Train tensor: N={ds_tr.X.shape[0]}  T={T}  D={D}")
    print(f"[INFO] Test  tensor: N={ds_te.X.shape[0]}  T={ds_te.X.shape[1]}  D={ds_te.X.shape[2]}")

    # DataLoaders
    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=False)
    dl_te = DataLoader(ds_te, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False)

    # Dispositivo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] Device:", device)
    if device.type == "cuda":
        print("[INFO] GPU:", torch.cuda.get_device_name(0))

    # Modelo
    model = BiLSTMClassifier(
        input_dim=D,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_classes=num_classes,
        dropout=args.dropout
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)

    # Pesos por clase (útil cuando hay pocas muestras por clase)
    class_weights = None
    if not args.no_class_weights:
        class_weights = compute_class_weights(ds_tr.y, num_classes)
        print("[INFO] ClassWeights: ON")
    else:
        print("[INFO] ClassWeights: OFF")

    # Entrenamiento con “mejor modelo” según accuracy en test
    os.makedirs(args.out_dir, exist_ok=True)
    best_acc = -1.0
    best_path = os.path.join(args.out_dir, "signrt_bilstm_best.pt")

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, dl_tr, optimizer, device, class_weights=class_weights)
        te_loss, te_acc = evaluate(model, dl_te, device)

        print(f"[E{epoch:03d}] train loss={tr_loss:.4f} acc={tr_acc*100:.1f}% | test loss={te_loss:.4f} acc={te_acc*100:.1f}%")

        if te_acc > best_acc:
            best_acc = te_acc
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "input_dim": D,
                    "num_classes": num_classes,
                    "hidden_dim": args.hidden_dim,
                    "num_layers": args.num_layers,
                    "dropout": args.dropout,
                    "label_map": lm,
                },
                best_path
            )
            print(f"[OK] Nuevo mejor modelo guardado: {best_path} (acc={best_acc*100:.1f}%)")

    print(f"[DONE] Mejor accuracy test: {best_acc*100:.1f}%")
    print(f"[DONE] Modelo: {best_path}")


if __name__ == "__main__":
    main()
