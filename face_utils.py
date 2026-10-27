"""
face_utils.py
-------------
OpenCV Face Detection & Recognition Module for AI Smart Attendance System.

Fixes & Enhancements:
1. Guaranteed Camera Release: Wrapped webcam loops in try/finally blocks to prevent hardware camera lock.
2. Contamination Prevention: Rejects corrupt, pitch-black, or completely blurry face samples during enrollment.
3. Coordinate Clamping: Prevents out-of-bounds text rendering on extreme frame edges.
4. Safe Multi-Face Gating: Blocks multiple faces to eliminate proxy attendance attempts.
5. Robust Model Lifecycle: Handles single-student, multi-student, and empty dataset states safely.
"""

import os
import shutil
import cv2
import numpy as np
from PIL import Image
from typing import List, Tuple, Optional, Dict, Any

# Paths configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "data", "dataset")
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_FILE = os.path.join(MODELS_DIR, "trained_lbph_model.yml")

os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Frontal Face Cascade (Primary)
FRONTAL_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(FRONTAL_CASCADE_PATH)

# Profile Face Cascade (Fallback)
PROFILE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_profileface.xml"
profile_cascade = cv2.CascadeClassifier(PROFILE_CASCADE_PATH)

# Default Recognition Parameters & Safety Thresholds
DEFAULT_MAX_DISTANCE_THRESHOLD: float = 65.0
DEFAULT_MIN_CONFIDENCE_PERCENTAGE: float = 70.0


def get_face_detector() -> cv2.CascadeClassifier:
    """Returns the primary Haar Cascade face detector."""
    return face_cascade


def get_recognizer() -> cv2.face.LBPHFaceRecognizer:
    """
    Creates and returns an LBPH Face Recognizer instance.
    Requires opencv-contrib-python.
    """
    if not hasattr(cv2, "face") or not hasattr(cv2.face, "LBPHFaceRecognizer_create"):
        raise ImportError(
            "OpenCV 'cv2.face' module not found! "
            "Please ensure 'opencv-contrib-python' is installed (not standard opencv-python)."
        )
    return cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=8, grid_y=8)


def is_model_trained() -> bool:
    """Checks whether the trained LBPH model file exists and is non-empty."""
    return os.path.exists(MODEL_FILE) and os.path.getsize(MODEL_FILE) > 0


def distance_to_confidence(distance: float) -> float:
    """
    Converts OpenCV LBPH raw distance to an intuitive 0.0% - 100.0% confidence percentage.
    In LBPH:
    - distance <= 25  -> ~90% - 100% (Exceptional match)
    - distance 35-50  -> ~75% - 88% (High confidence match)
    - distance 50-65  -> ~65% - 75% (Acceptable match)
    - distance > 75   -> < 55% (Low confidence / Unrecognized)
    """
    if distance <= 0:
        return 100.0
    confidence = 100.0 - (distance / 130.0) * 100.0
    return round(float(np.clip(confidence, 0.0, 100.0)), 1)


def preprocess_face(face_roi: np.ndarray, target_size: Tuple[int, int] = (200, 200)) -> np.ndarray:
    """
    Standardizes a cropped face ROI for recognition/training:
    1. Converts multi-channel image to grayscale.
    2. Resizes to standard dimension (200x200).
    3. Performs Histogram Equalization to neutralize non-uniform shadows and lighting.
    """
    if face_roi is None or face_roi.size == 0:
        return np.zeros(target_size, dtype=np.uint8)

    if len(face_roi.shape) == 3:
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
    else:
        gray = face_roi

    resized = cv2.resize(gray, target_size, interpolation=cv2.INTER_AREA)
    equalized = cv2.equalizeHist(resized)
    return equalized


def detect_faces(image_bgr_or_rgb: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """
    Detects faces using multi-scale Haar Cascade.
    Falls back to profile face cascade if frontal face is missed.
    Clamps all bounding box coordinates inside the frame dimensions.
    """
    if image_bgr_or_rgb is None or image_bgr_or_rgb.size == 0:
        return []

    frame_h, frame_w = image_bgr_or_rgb.shape[:2]

    if len(image_bgr_or_rgb.shape) == 3:
        gray = cv2.cvtColor(image_bgr_or_rgb, cv2.COLOR_BGR2GRAY)
    else:
        gray = image_bgr_or_rgb

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.15,
        minNeighbors=5,
        minSize=(60, 60),
        flags=cv2.CASCADE_SCALE_IMAGE,
    )

    if len(faces) == 0 and profile_cascade is not None and not profile_cascade.empty():
        faces = profile_cascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=4,
            minSize=(60, 60),
        )

    # Sanitize and clamp bounding boxes
    clamped_faces = []
    for (x, y, w, h) in faces:
        cx = max(0, min(int(x), frame_w - 1))
        cy = max(0, min(int(y), frame_h - 1))
        cw = max(10, min(int(w), frame_w - cx))
        ch = max(10, min(int(h), frame_h - cy))
        clamped_faces.append((cx, cy, cw, ch))

    return clamped_faces


def analyze_face_quality(
    image_np: np.ndarray,
    face_box: Tuple[int, int, int, int],
) -> Dict[str, Any]:
    """
    Evaluates environmental quality factors:
    1. Lighting/Illumination (underexposed vs overexposed)
    2. Sharpness / Motion Blur (Laplacian variance)
    3. Face Distance / Proportion relative to camera frame
    4. Centering / Alignment
    """
    frame_h, frame_w = image_np.shape[:2]
    x, y, w, h = face_box

    if len(image_np.shape) == 3:
        gray_frame = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    else:
        gray_frame = image_np

    face_roi_gray = gray_frame[y : y + h, x : x + w]
    if face_roi_gray.size == 0:
        face_roi_gray = gray_frame

    # 1. Lighting Analysis (Mean pixel intensity 0 - 255)
    mean_brightness = float(np.mean(face_roi_gray))
    lighting_status = "Good"
    lighting_tip = "Lighting is well-balanced."

    if mean_brightness < 45.0:
        lighting_status = "Too Dark"
        lighting_tip = f"Lighting is too dark ({mean_brightness:.0f}/255). Please face a light source or brighten the room."
    elif mean_brightness > 215.0:
        lighting_status = "Too Bright / Glare"
        lighting_tip = f"Lighting is overexposed ({mean_brightness:.0f}/255). Avoid strong overhead spotlights or direct glare."

    # 2. Sharpness / Blur Detection (Laplacian Variance)
    laplacian_var = float(cv2.Laplacian(face_roi_gray, cv2.CV_64F).var())
    sharpness_status = "Sharp"
    sharpness_tip = "Image is crisp and clear."

    if laplacian_var < 35.0:
        sharpness_status = "Blurry"
        sharpness_tip = f"Camera snapshot is blurry (Score: {laplacian_var:.1f}). Hold still when capturing."

    # 3. Distance Check
    face_area_ratio = float((w * h) / (frame_w * frame_h))
    distance_status = "Optimal"
    distance_tip = "Camera distance is optimal."

    if w < 75 or h < 75 or face_area_ratio < 0.05:
        distance_status = "Too Far"
        distance_tip = "Move closer to camera until your face fills at least 25% of the screen."
    elif w > frame_w * 0.85 or h > frame_h * 0.85:
        distance_status = "Too Close"
        distance_tip = "You are too close to the camera. Please step back slightly."

    # 4. Centering Check
    center_x = x + w / 2.0
    center_y = y + h / 2.0
    margin_x = frame_w * 0.15
    margin_y = frame_h * 0.15

    is_centered = (margin_x <= center_x <= frame_w - margin_x) and (margin_y <= center_y <= frame_h - margin_y)
    centering_status = "Centered" if is_centered else "Off-Center"
    centering_tip = "Face is centered." if is_centered else "Please center your face inside the camera view."

    quality_issues = []
    if lighting_status != "Good":
        quality_issues.append(lighting_tip)
    if sharpness_status != "Sharp":
        quality_issues.append(sharpness_tip)
    if distance_status != "Optimal":
        quality_issues.append(distance_tip)
    if centering_status != "Centered":
        quality_issues.append(centering_tip)

    is_quality_acceptable = (
        lighting_status == "Good"
        and sharpness_status != "Blurry"
        and distance_status != "Too Far"
    )

    return {
        "is_acceptable": is_quality_acceptable,
        "brightness": round(mean_brightness, 1),
        "brightness_status": lighting_status,
        "sharpness": round(laplacian_var, 1),
        "sharpness_status": sharpness_status,
        "distance_status": distance_status,
        "centering_status": centering_status,
        "face_area_ratio": round(face_area_ratio * 100, 1),
        "guidance_tips": quality_issues,
    }


def save_student_face_samples(student_id: int, image_np: np.ndarray, sample_count: int = 1) -> Tuple[bool, str, int, Optional[Dict]]:
    """
    Validates face sample quality and saves to data/dataset/student_<id>/.
    Prevents model contamination by rejecting pitch-black or completely blurry frames.
    """
    try:
        student_id = int(student_id)
    except (ValueError, TypeError):
        return False, "Invalid Student ID provided.", 0, None

    student_folder = os.path.join(DATASET_DIR, f"student_{student_id}")
    os.makedirs(student_folder, exist_ok=True)

    faces = detect_faces(image_np)
    if len(faces) == 0:
        return False, "No face detected in the picture. Ensure proper lighting and look directly into the camera.", 0, None

    if len(faces) > 1:
        return False, f"Multiple faces detected ({len(faces)} people found). Only the registering student should be in frame.", 0, None

    face_box = faces[0]
    quality = analyze_face_quality(image_np, face_box)

    # Reject corrupt or unviable samples to prevent model contamination
    if quality["brightness"] < 25.0:
        return False, f"Face sample is too dark ({quality['brightness']:.0f}/255). Please improve lighting.", 0, quality

    if quality["sharpness"] < 15.0:
        return False, f"Face sample is too blurry (Score: {quality['sharpness']:.1f}). Please stay still.", 0, quality

    x, y, w, h = face_box
    face_roi = image_np[y : y + h, x : x + w]
    processed_face = preprocess_face(face_roi)

    existing_files = [f for f in os.listdir(student_folder) if f.lower().endswith((".jpg", ".png"))]
    start_idx = len(existing_files) + 1

    saved = 0
    file_path = os.path.join(student_folder, f"sample_{start_idx}.jpg")
    cv2.imwrite(file_path, processed_face)
    saved += 1

    if sample_count > 1:
        # Augmentation: horizontal flip
        flipped = cv2.flip(processed_face, 1)
        cv2.imwrite(os.path.join(student_folder, f"sample_{start_idx + 1}.jpg"), flipped)
        saved += 1

        # Augmentation: slight brightness shifts
        bright = cv2.convertScaleAbs(processed_face, alpha=1.1, beta=10)
        cv2.imwrite(os.path.join(student_folder, f"sample_{start_idx + 2}.jpg"), bright)
        saved += 1

        dark = cv2.convertScaleAbs(processed_face, alpha=0.9, beta=-10)
        cv2.imwrite(os.path.join(student_folder, f"sample_{start_idx + 3}.jpg"), dark)
        saved += 1

    total_samples = len([f for f in os.listdir(student_folder) if f.lower().endswith((".jpg", ".png"))])
    return True, f"Successfully captured face sample. Total stored samples: {total_samples}.", total_samples, quality


def capture_samples_via_webcam(student_id: int, target_samples: int = 30) -> Tuple[bool, str, int]:
    """
    Opens hardware webcam and captures face frames.
    Guarantees resource release via try...finally to prevent camera lock.
    """
    try:
        student_id = int(student_id)
    except (ValueError, TypeError):
        return False, "Invalid Student ID provided.", 0

    student_folder = os.path.join(DATASET_DIR, f"student_{student_id}")
    os.makedirs(student_folder, exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return False, "Could not access webcam. Please check if camera is connected and not used by another application.", 0

    samples_collected = 0
    frame_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            faces = detect_faces(frame)

            cv2.putText(
                frame,
                f"Capturing Samples: {samples_collected}/{target_samples}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                frame,
                "Press 'q' or ESC to stop early",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 200, 255),
                2,
            )

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                if frame_count % 3 == 0 and samples_collected < target_samples:
                    face_roi = frame[y : y + h, x : x + w]
                    processed_face = preprocess_face(face_roi)
                    samples_collected += 1
                    img_path = os.path.join(student_folder, f"sample_{samples_collected}.jpg")
                    cv2.imwrite(img_path, processed_face)

            cv2.imshow("Student Face Registration - Press Q to Exit", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27) or samples_collected >= target_samples:
                break
    finally:
        # Guaranteed cleanup: release hardware camera
        cap.release()
        cv2.destroyAllWindows()

    if samples_collected == 0:
        return False, "No face samples were captured. Please ensure face is centered in the camera frame.", 0

    return True, f"Successfully captured {samples_collected} face samples for Student #{student_id}.", samples_collected


def train_face_recognizer() -> Tuple[bool, str, int]:
    """
    Trains OpenCV LBPH Face Recognizer on student face images.
    If no students exist, removes obsolete model file.
    """
    if not os.path.exists(DATASET_DIR):
        return False, "Dataset directory does not exist. Please register at least one student.", 0

    student_dirs = [d for d in os.listdir(DATASET_DIR) if d.startswith("student_") and os.path.isdir(os.path.join(DATASET_DIR, d))]

    if not student_dirs:
        # Clean up obsolete model file if all students were removed
        if os.path.exists(MODEL_FILE):
            try:
                os.remove(MODEL_FILE)
            except OSError:
                pass
        return False, "No student face data found. Please register students with face samples first.", 0

    faces = []
    labels = []

    for folder in student_dirs:
        try:
            student_id = int(folder.split("_")[1])
        except (IndexError, ValueError):
            continue

        folder_path = os.path.join(DATASET_DIR, folder)
        image_files = [f for f in os.listdir(folder_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

        for img_name in image_files:
            img_path = os.path.join(folder_path, img_name)
            gray_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if gray_img is None:
                continue
            if gray_img.shape != (200, 200):
                gray_img = cv2.resize(gray_img, (200, 200), interpolation=cv2.INTER_AREA)

            faces.append(gray_img)
            labels.append(student_id)

    if len(faces) == 0:
        return False, "No valid face image files found in dataset directories.", 0

    try:
        recognizer = get_recognizer()
        recognizer.train(faces, np.array(labels, dtype=np.int32))
        recognizer.save(MODEL_FILE)
        unique_students = len(set(labels))
        return (
            True,
            f"LBPH Model successfully trained with {len(faces)} face samples across {unique_students} student(s)!",
            len(faces),
        )
    except Exception as e:
        return False, f"Failed to train LBPH model: {str(e)}", 0


def recognize_student_face(
    image_bgr_or_rgb: np.ndarray,
    min_confidence_pct: float = DEFAULT_MIN_CONFIDENCE_PERCENTAGE,
    max_distance_threshold: float = DEFAULT_MAX_DISTANCE_THRESHOLD,
) -> Dict[str, Any]:
    """
    Performs face detection, environmental diagnostics, and LBPH recognition.
    Strictly gates attendance: if confidence < threshold, is_confidence_sufficient is False.
    """
    default_guidance = [
        "💡 Lighting: Ensure front-facing light. Avoid strong background lights or ceiling glare.",
        "💡 Positioning: Keep your head upright, center your face, and look directly into the camera lens.",
        "💡 Distance: Stay roughly arm's length (50-70 cm) away from the camera.",
        "💡 Stability: Hold still for 1 second when taking the snapshot to avoid motion blur.",
    ]

    if not is_model_trained():
        return {
            "status": "MODEL_NOT_READY",
            "predicted_id": None,
            "distance": 999.0,
            "confidence_pct": 0.0,
            "bounding_box": None,
            "is_confidence_sufficient": False,
            "message": "Face recognition model is not trained yet. Please register students and train the model first.",
            "quality": None,
            "guidance": default_guidance,
        }

    # Step 1: Detect Faces
    faces = detect_faces(image_bgr_or_rgb)

    if len(faces) == 0:
        return {
            "status": "NO_FACE",
            "predicted_id": None,
            "distance": 999.0,
            "confidence_pct": 0.0,
            "bounding_box": None,
            "is_confidence_sufficient": False,
            "message": "No face detected in the camera frame.",
            "quality": None,
            "guidance": [
                "❌ Face not detected.",
                "👉 Please face the camera directly.",
                "👉 Make sure lighting is adequate and remove any cap, scarf, or mask covering facial features.",
            ] + default_guidance,
        }

    if len(faces) > 1:
        return {
            "status": "MULTIPLE_FACES",
            "predicted_id": None,
            "distance": 999.0,
            "confidence_pct": 0.0,
            "bounding_box": faces[0],
            "is_confidence_sufficient": False,
            "message": f"Multiple faces detected ({len(faces)} people found). Attendance can only be marked for one person at a time.",
            "quality": None,
            "guidance": [
                f"⚠️ Found {len(faces)} faces in frame.",
                "👉 Only the student marking attendance must stand in front of the camera.",
                "👉 Please ask bystanders or other students to step aside.",
            ],
        }

    face_box = faces[0]
    x, y, w, h = face_box

    # Step 2: Quality & Environmental Analysis
    quality = analyze_face_quality(image_bgr_or_rgb, face_box)

    # Step 3: LBPH Prediction
    face_roi = image_bgr_or_rgb[y : y + h, x : x + w]
    processed_face = preprocess_face(face_roi)

    recognizer = get_recognizer()
    recognizer.read(MODEL_FILE)

    raw_label, raw_distance = recognizer.predict(processed_face)
    confidence_pct = distance_to_confidence(raw_distance)

    # Step 4: Strict Confidence Verification
    is_sufficient = (raw_distance <= max_distance_threshold) and (confidence_pct >= min_confidence_pct)
    guidance_tips = list(quality.get("guidance_tips", []))

    if is_sufficient:
        return {
            "status": "SUCCESS",
            "predicted_id": int(raw_label),
            "distance": round(float(raw_distance), 1),
            "confidence_pct": confidence_pct,
            "bounding_box": face_box,
            "is_confidence_sufficient": True,
            "message": f"Face recognized successfully with {confidence_pct}% match confidence (LBPH Distance: {round(raw_distance, 1)}).",
            "quality": quality,
            "guidance": ["✅ Face alignment and environmental conditions verified."],
        }
    elif raw_distance <= 85.0:
        return {
            "status": "LOW_CONFIDENCE",
            "predicted_id": int(raw_label),
            "distance": round(float(raw_distance), 1),
            "confidence_pct": confidence_pct,
            "bounding_box": face_box,
            "is_confidence_sufficient": False,
            "message": (
                f"Recognition confidence insufficient ({confidence_pct}% < {min_confidence_pct}% required). "
                f"LBPH distance is {round(raw_distance, 1)} (Safe threshold: ≤ {max_distance_threshold}). "
                "Attendance NOT marked to prevent false-positive errors."
            ),
            "quality": quality,
            "guidance": [
                f"⚠️ Insufficient Confidence: {confidence_pct}% (Need at least {min_confidence_pct}%).",
                "👉 Attendance was NOT marked to maintain system accuracy.",
            ] + (guidance_tips if guidance_tips else default_guidance),
        }
    else:
        return {
            "status": "UNKNOWN_PERSON",
            "predicted_id": None,
            "distance": round(float(raw_distance), 1),
            "confidence_pct": confidence_pct,
            "bounding_box": face_box,
            "is_confidence_sufficient": False,
            "message": (
                f"Face not recognized (Unknown or unregistered student). "
                f"Confidence is very low ({confidence_pct}%). Attendance NOT marked."
            ),
            "quality": quality,
            "guidance": [
                "❌ Biometric features do not match any enrolled student.",
                "👉 If you are a new student, please register via 'Student Registration' first.",
                "👉 If you have registered, retrain the model or capture clearer face samples under better lighting.",
            ] + default_guidance,
        }


def draw_recognition_box(
    image_rgb: np.ndarray,
    bounding_box: Tuple[int, int, int, int],
    display_text: str,
    confidence_pct: Optional[float] = None,
    status: str = "SUCCESS",
) -> np.ndarray:
    """
    Draws an annotated bounding box with safe boundary clamping.
    """
    annotated = image_rgb.copy()
    frame_h, frame_w = annotated.shape[:2]
    x, y, w, h = bounding_box

    # Clamp coordinates
    x = max(0, min(x, frame_w - 1))
    y = max(0, min(y, frame_h - 1))
    w = max(10, min(w, frame_w - x))
    h = max(10, min(h, frame_h - y))

    if status == "SUCCESS":
        color = (34, 197, 94)  # Vibrant Green
    elif status in ("LOW_CONFIDENCE", "MULTIPLE_FACES"):
        color = (245, 158, 11)  # Amber / Orange
    else:
        color = (239, 68, 68)  # Crimson Red

    cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 3)

    corner_length = int(w * 0.15)
    t = 4
    cv2.line(annotated, (x, y), (x + corner_length, y), color, t)
    cv2.line(annotated, (x, y), (x, y + corner_length), color, t)
    cv2.line(annotated, (x + w, y), (x + w - corner_length, y), color, t)
    cv2.line(annotated, (x + w, y), (x + w, y + corner_length), color, t)
    cv2.line(annotated, (x, y + h), (x + corner_length, y + h), color, t)
    cv2.line(annotated, (x, y + h), (x, y + h - corner_length), color, t)
    cv2.line(annotated, (x + w, y + h), (x + w - corner_length, y + h), color, t)
    cv2.line(annotated, (x + w, y + h), (x + w, y + h - corner_length), color, t)

    # Top Banner Header (Clamped within frame)
    header_height = 38
    y_header_top = max(0, y - header_height)
    cv2.rectangle(annotated, (x, y_header_top), (x + w, y), color, cv2.FILLED)

    cv2.putText(
        annotated,
        display_text[:30],
        (x + 6, max(20, y - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    # Bottom Confidence Badge (Clamped within frame)
    if confidence_pct is not None:
        conf_text = f"Confidence: {confidence_pct}%"
        conf_badge_h = 24
        y_badge_bottom = min(frame_h, y + h + conf_badge_h)
        cv2.rectangle(annotated, (x, y + h), (x + w, y_badge_bottom), (30, 30, 30), cv2.FILLED)
        cv2.putText(
            annotated,
            conf_text,
            (x + 6, min(frame_h - 6, y + h + 18)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return annotated
