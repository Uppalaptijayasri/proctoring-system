import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO

# Initialize mediapipe
mp_face_mesh = mp.solutions.face_mesh
mp_face_detection = mp.solutions.face_detection

face_mesh = mp_face_mesh.FaceMesh(max_num_faces=5, refine_landmarks=True)
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

# Load YOLO model for phone detection
yolo_model = YOLO("yolov8n.pt")

# Start webcam
cap = cv2.VideoCapture(0)

# --------------- Helper Functions ----------------

def detect_blink(landmarks, w, h):
    """Detect blink based on eye aspect ratio"""
    left_eye = [33, 133, 159, 145]
    right_eye = [362, 263, 386, 374]
    blink_ratios = []

    for eye in [left_eye, right_eye]:
        top = np.array([landmarks[eye[2]].x * w, landmarks[eye[2]].y * h])
        bottom = np.array([landmarks[eye[3]].x * w, landmarks[eye[3]].y * h])
        left = np.array([landmarks[eye[0]].x * w, landmarks[eye[0]].y * h])
        right = np.array([landmarks[eye[1]].x * w, landmarks[eye[1]].y * h])

        vertical = np.linalg.norm(top - bottom)
        horizontal = np.linalg.norm(left - right)
        blink_ratios.append(vertical / horizontal)

    avg_ratio = np.mean(blink_ratios)
    return avg_ratio < 0.23  # Adjust threshold as needed


def detect_eye_direction(landmarks, w, h):
    """Detect if eyes are looking away"""
    left_eye = [33, 133]
    right_eye = [362, 263]
    iris_left = 468
    iris_right = 473

    left_iris_x = landmarks[iris_left].x * w
    left_corner_x = landmarks[left_eye[0]].x * w
    right_corner_x = landmarks[left_eye[1]].x * w
    ratio_left = (left_iris_x - left_corner_x) / (right_corner_x - left_corner_x)

    right_iris_x = landmarks[iris_right].x * w
    left_corner_x_r = landmarks[right_eye[0]].x * w
    right_corner_x_r = landmarks[right_eye[1]].x * w
    ratio_right = (right_iris_x - left_corner_x_r) / (right_corner_x_r - left_corner_x_r)

    avg_ratio = (ratio_left + ratio_right) / 2

    if avg_ratio < 0.35:
        return "Left"
    elif avg_ratio > 0.65:
        return "Right"
    else:
        return "Center"


def detect_talking(landmarks, w, h):
    """Detect if mouth is open (talking or reading aloud)"""
    top_lip = landmarks[13]
    bottom_lip = landmarks[14]
    lip_distance = abs(top_lip.y * h - bottom_lip.y * h)
    return lip_distance > 15  # Adjust based on your camera


def detect_phone(frame):
    """Detect if phone is visible using YOLOv8"""
    results = yolo_model.predict(frame, verbose=False)
    for r in results:
        if hasattr(r, "boxes") and r.boxes is not None:
            for c in r.boxes.cls:
                class_id = int(c.item())
                class_name = r.names[class_id]
                if "cell phone" in class_name.lower():
                    return True
    return False


# ------------------- Main Loop -------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, _ = frame.shape

    results_mesh = face_mesh.process(frame_rgb)
    detections = face_detection.process(frame_rgb)

    face_count = 0
    blink = False
    looking_away = False
    talking = False
    phone_detected = False

    # Count faces
    if detections.detections:
        face_count = len(detections.detections)

    # Phone detection
    phone_detected = detect_phone(frame)

    # Facial features
    if results_mesh.multi_face_landmarks:
        for face_landmarks in results_mesh.multi_face_landmarks:
            blink = detect_blink(face_landmarks.landmark, w, h)
            eye_dir = detect_eye_direction(face_landmarks.landmark, w, h)
            if eye_dir != "Center":
                looking_away = True
            talking = detect_talking(face_landmarks.landmark, w, h)

    # --------------- Alerts -----------------
    if face_count > 1:
        status = f"⚠ Multiple Faces Detected ({face_count})"
        color = (0, 0, 255)
    elif phone_detected:
        status = "📱 Phone Detected!"
        color = (0, 0, 255)
    elif looking_away:
        status = "⚠ Please Look at the Screen!"
        color = (0, 0, 255)
    elif talking:
        status = "🤐 Don't Talk!"
        color = (0, 0, 255)
    elif blink:
        status = "👁 Blink Detected"
        color = (0, 255, 255)
    else:
        status = "✅ Monitoring Normal"
        color = (0, 255, 0)

    # Display result
    cv2.putText(frame, status, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    cv2.imshow("AI Proctoring System", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
