# apps/preview_cam_min.py
import cv2, time

def open_cam_dshow_mjpg(idx=0, w=640, h=480, fps=30):
    cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_FPS, fps)
    for _ in range(6):
        ok, f = cap.read()
        if ok and f is not None:
            break
        time.sleep(0.03)
    return cap

cap = open_cam_dshow_mjpg(0, 640, 480, 30)
cv2.namedWindow("MIN", cv2.WINDOW_NORMAL)

while True:
    ok, frame = cap.read()
    if not ok or frame is None:
        print("WARN: frame nulo")
        break
    cv2.imshow("MIN", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
