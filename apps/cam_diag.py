# apps/cam_diag.py
import cv2, time, os

COMBOS = [
    ("DSHOW+MJPG", cv2.CAP_DSHOW, "MJPG"),
    ("DSHOW+default", cv2.CAP_DSHOW, None),
    ("MSMF+MJPG", cv2.CAP_MSMF, "MJPG"),
    ("MSMF+default", cv2.CAP_MSMF, None),
]

def open_try(name, backend, fourcc, w=640, h=480, fps=30):
    cap = cv2.VideoCapture(0, backend)
    if fourcc:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_FPS, fps)

    # warm-up
    ok, frame = False, None
    for _ in range(8):
        ok, frame = cap.read()
        if ok and frame is not None:
            break
        time.sleep(0.03)

    real_w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    real_h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    real_fps = cap.get(cv2.CAP_PROP_FPS)
    fcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    fcc_str = "".join([chr((fcc >> 8*i) & 0xFF) for i in range(4)])

    print(f"[TRY] {name}: opened={cap.isOpened()}  size={real_w}x{real_h}  fps={real_fps:.1f}  fourcc='{fcc_str}'  ok={ok}")
    return cap, ok, frame, (real_w, real_h, fcc_str)

def show(name, frame):
    if frame is None:
        return
    mean, std = frame.mean(), frame.std()
    print(f"[INFO] {name}: frame mean={mean:.2f} std={std:.2f}")
    cv2.imshow(f"DIAG - {name}", frame)
    print("[TIP] Mira la ventana. Pulsa ESPACIO si se ve bien; ENTER para probar la siguiente; ESC para salir.")
    while True:
        k = cv2.waitKey(0) & 0xFF
        if k in (27, 13, 32):  # ESC, ENTER, SPACE
            break

def main():
    os.makedirs("data/diag", exist_ok=True)
    cv2.namedWindow("DIAG - Preview", cv2.WINDOW_NORMAL)

    for (name, backend, fcc) in COMBOS:
        cap, ok, frame, meta = open_try(name, backend, fcc)
        if ok and frame is not None:
            show(name, frame)
            # guarda un snapshot para revisar si se ve ruido o correcto
            path = f"data/diag/{name.replace('+','_')}.png"
            cv2.imwrite(path, frame)
            print(f"[SAVE] {path}")
        if cap: cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
