import cv2
import numpy as np
import mediapipe as mp

mp_holistic = mp.solutions.holistic
_DRAW = mp.solutions.drawing_utils
_STYLE = mp.solutions.drawing_styles

FACE_N = 468
HAND_N = 21
POSE_N = 33
TOTAL_N = FACE_N + HAND_N + HAND_N + POSE_N  # 543

def _landmarks_to_array(results) -> np.ndarray:
    """Convierte los landmarks de MediaPipe a un array [543,4] (x,y,z,vis)."""
    def to_arr(lms, expected):
        if lms is None:
            return np.zeros((expected, 4), dtype=np.float32)
        arr = np.array([[lm.x, lm.y, lm.z, getattr(lm, "visibility", 1.0)] for lm in lms.landmark],
                       dtype=np.float32)
        if arr.shape[0] < expected:
            pad = np.zeros((expected - arr.shape[0], 4), dtype=np.float32)
            arr = np.vstack([arr, pad])
        return arr[:expected]

    face = to_arr(results.face_landmarks, FACE_N)
    left = to_arr(results.left_hand_landmarks, HAND_N)
    right = to_arr(results.right_hand_landmarks, HAND_N)
    pose = to_arr(results.pose_landmarks, POSE_N)
    return np.concatenate([face, left, right, pose], axis=0)

def normalize_frame(X: np.ndarray) -> np.ndarray:
    """Normaliza por centro y escala (distancia entre hombros). X: [543,4]."""
    POSE_START = FACE_N + HAND_N + HAND_N
    L_SH = POSE_START + 11
    R_SH = POSE_START + 12
    # Si no hay hombros, caer a escala=1
    pL, pR = X[L_SH, :3], X[R_SH, :3]
    scale = np.linalg.norm(pL - pR) + 1e-6
    center = (pL + pR) / 2.0
    Xn = X.copy()
    Xn[:, :3] = (X[:, :3] - center) / scale
    return Xn

def draw_overlay(bgr, results):
    _DRAW.draw_landmarks(
        bgr, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=_STYLE.get_default_face_mesh_tesselation_style()
    )
    _DRAW.draw_landmarks(bgr, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    _DRAW.draw_landmarks(bgr, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    _DRAW.draw_landmarks(
        bgr, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
        landmark_drawing_spec=_STYLE.get_default_pose_landmarks_style()
    )
