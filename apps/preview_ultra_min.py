# apps/preview_ultra_min.py
import cv2, time

def open_cam_dshow_default(idx=0, w=640, h=480, fps=30):
    cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_FPS, fps)
    # warm-up
    for _ in range(8):
        ok, f = cap.read()
        if ok and f is not None:
            break
        time.sleep(0.03)
    return cap

cap = open_cam_dshow_default(0, 640, 480, 30)
if not cap or not cap.isOpened():
    print("ERROR: no se pudo abrir la cámara con DSHOW (default).")
    raise SystemExit

cv2.namedWindow("ULTRA", cv2.WINDOW_NORMAL)
print("[INFO] Ventana 'ULTRA' abierta. Pulsa ESC para salir.")

while True:
    ok, frame = cap.read()
    if not ok or frame is None:
        print("WARN: frame nulo")
        break
    cv2.imshow("ULTRA", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
