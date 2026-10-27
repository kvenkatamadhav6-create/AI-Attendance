"""
flask_app.py
------------
REST API and Flask Backend Service for the AI Smart Attendance Management System.

Endpoints:
- POST   /api/login             : Authenticate admin user
- POST   /api/register          : Register student and upload face image
- POST   /api/attendance/mark   : Recognize face from image and mark attendance
- GET    /api/attendance        : Retrieve filtered attendance records
- GET    /api/attendance/export : Download attendance report as CSV
- GET    /api/students          : List all registered students
- DELETE /api/students/<id>     : Delete student, attendance logs, and dataset images
- POST   /api/model/train       : Trigger LBPH model training
- GET    /api/dashboard/stats   : Fetch summary metrics
- GET    /                      : Health check & API documentation
"""

import os
import base64
import io
from datetime import datetime
import cv2
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, Response

# Import core modules
import database as db
import face_utils as fu

app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB max payload


def decode_base64_image(base64_string: str) -> Optional[np.ndarray]:
    """Decodes a base64 string or data URI into an OpenCV BGR/RGB numpy array."""
    try:
        if "," in base64_string:
            base64_string = base64_string.split(",")[1]
        img_bytes = base64.b64decode(base64_string)
        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        return np.array(pil_img)
    except Exception:
        return None


@app.route("/", methods=["GET"])
def api_home():
    """Health check and API catalog."""
    return jsonify({
        "system": "AI Based Smart Attendance Management API",
        "status": "ONLINE",
        "version": "2.0.0",
        "model_trained": fu.is_model_trained(),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "endpoints": {
            "POST /api/login": "Authenticate admin user",
            "POST /api/register": "Register student (roll_number, name, department, [image_base64])",
            "POST /api/attendance/mark": "Recognize face and mark attendance (image_base64)",
            "GET /api/attendance": "Query attendance records (?date=YYYY-MM-DD&department=...&search=...)",
            "GET /api/attendance/export": "Download attendance report as CSV",
            "GET /api/students": "List all enrolled students",
            "DELETE /api/students/<id>": "Delete student and associated face dataset",
            "POST /api/model/train": "Train or retrain LBPH face recognition model",
            "GET /api/dashboard/stats": "Get dashboard summary statistics",
        }
    }), 200


@app.route("/api/login", methods=["POST"])
def api_login():
    """Admin login authentication endpoint."""
    data = request.get_json(silent=True) or {}
    username = data.get("username", "").strip()
    password = data.get("password", "")

    if not username or not password:
        return jsonify({"success": False, "message": "Username and password are required."}), 400

    success, msg, user_data = db.verify_admin_login(username, password)
    if success and user_data:
        return jsonify({
            "success": True,
            "message": msg,
            "user": user_data,
        }), 200
    return jsonify({"success": False, "message": msg}), 401


@app.route("/api/register", methods=["POST"])
def api_register_student():
    """Register student profile and optional face sample."""
    data = request.get_json(silent=True) or {}
    roll_number = data.get("roll_number", "").strip().upper()
    name = data.get("name", "").strip().title()
    department = data.get("department", "").strip().upper()
    image_base64 = data.get("image_base64", None)

    if not roll_number or not name or not department:
        return jsonify({"success": False, "message": "Missing required fields: roll_number, name, department."}), 400

    success, msg, student_id = db.add_student(roll_number, name, department)
    if not success:
        return jsonify({"success": False, "message": msg}), 409

    face_saved = False
    face_msg = "No image provided. Please register face samples later."

    if image_base64 and student_id:
        img_np = decode_base64_image(image_base64)
        if img_np is not None:
            f_ok, f_text, f_count, _ = fu.save_student_face_samples(student_id, img_np, sample_count=4)
            face_saved = f_ok
            face_msg = f_text
            # Automatically train model
            fu.train_face_recognizer()

    return jsonify({
        "success": True,
        "message": msg,
        "student_id": student_id,
        "face_saved": face_saved,
        "face_details": face_msg,
    }), 201


@app.route("/api/attendance/mark", methods=["POST"])
def api_mark_attendance():
    """
    Scans face, validates match confidence >= 70%, and logs attendance.
    Prevents duplicate attendance for the same day.
    """
    data = request.get_json(silent=True) or {}
    image_base64 = data.get("image_base64", "")
    min_confidence = float(data.get("min_confidence", fu.DEFAULT_MIN_CONFIDENCE_PERCENTAGE))
    max_distance = float(data.get("max_distance", fu.DEFAULT_MAX_DISTANCE_THRESHOLD))

    if not image_base64:
        return jsonify({"success": False, "message": "Missing 'image_base64' parameter."}), 400

    img_np = decode_base64_image(image_base64)
    if img_np is None:
        return jsonify({"success": False, "message": "Invalid base64 image data."}), 400

    rec_result = fu.recognize_student_face(
        img_np,
        min_confidence_pct=min_confidence,
        max_distance_threshold=max_distance,
    )

    status = rec_result["status"]
    is_sufficient = rec_result["is_confidence_sufficient"]
    predicted_id = rec_result["predicted_id"]
    confidence_pct = rec_result["confidence_pct"]

    # Security Gate: Insufficient confidence blocks attendance
    if not is_sufficient or predicted_id is None:
        return jsonify({
            "success": False,
            "status": status,
            "confidence_pct": confidence_pct,
            "distance": rec_result["distance"],
            "message": rec_result["message"],
            "guidance": rec_result.get("guidance", []),
            "quality": rec_result.get("quality", None),
        }), 422

    student = db.get_student_by_id(predicted_id)
    if not student:
        return jsonify({
            "success": False,
            "status": "STUDENT_NOT_FOUND",
            "message": f"Predicted ID #{predicted_id} has no matching profile in the database.",
        }), 404

    # Mark attendance in SQLite
    marked, mark_msg, record_data = db.mark_attendance(
        student_id=predicted_id,
        roll_number=student["roll_number"],
        name=student["name"],
        department=student["department"],
    )

    if not marked:
        # Duplicate entry prevented (Requirement 4)
        return jsonify({
            "success": False,
            "status": "DUPLICATE_ENTRY_PREVENTED",
            "message": mark_msg,
            "student": student,
            "prior_record": record_data,
        }), 409

    return jsonify({
        "success": True,
        "status": "ATTENDANCE_LOGGED",
        "message": mark_msg,
        "confidence_pct": confidence_pct,
        "distance": rec_result["distance"],
        "record": record_data,
    }), 200


@app.route("/api/attendance", methods=["GET"])
def api_get_attendance():
    """Fetches attendance records with optional filters."""
    date_filter = request.args.get("date", None)
    dept_filter = request.args.get("department", None)
    search_query = request.args.get("search", None)

    df = db.get_attendance_records(date_filter=date_filter, dept_filter=dept_filter, search_query=search_query)
    records = df.to_dict(orient="records")
    return jsonify({
        "success": True,
        "count": len(records),
        "records": records,
    }), 200


@app.route("/api/attendance/export", methods=["GET"])
def api_export_csv():
    """Generates and downloads a CSV attendance report."""
    date_filter = request.args.get("date", None)
    dept_filter = request.args.get("department", None)
    search_query = request.args.get("search", None)

    df = db.get_attendance_records(date_filter=date_filter, dept_filter=dept_filter, search_query=search_query)
    csv_data = df.to_csv(index=False)
    filename = f"Attendance_Report_{date_filter or 'All'}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    return Response(
        csv_data,
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment;filename={filename}"},
    )


@app.route("/api/students", methods=["GET"])
def api_get_students():
    """Lists all registered students."""
    df = db.get_all_students()
    students = df.to_dict(orient="records")
    return jsonify({
        "success": True,
        "count": len(students),
        "students": students,
    }), 200


@app.route("/api/students/<int:student_id>", methods=["DELETE"])
def api_delete_student(student_id: int):
    """Deletes a student, their attendance logs, and physical face dataset."""
    success, msg = db.delete_student(student_id)
    if success:
        # Re-train model to eliminate deleted face
        fu.train_face_recognizer()
        return jsonify({"success": True, "message": msg}), 200
    return jsonify({"success": False, "message": msg}), 404


@app.route("/api/model/train", methods=["POST"])
def api_train_model():
    """Retrains the LBPH model on all available student faces."""
    success, msg, total_samples = fu.train_face_recognizer()
    status_code = 200 if success else 400
    return jsonify({
        "success": success,
        "message": msg,
        "total_samples": total_samples,
    }), status_code


@app.route("/api/dashboard/stats", methods=["GET"])
def api_dashboard_stats():
    """Fetches summary stats for dashboard cards."""
    stats = db.get_dashboard_metrics()
    return jsonify({
        "success": True,
        "stats": stats,
    }), 200


if __name__ == "__main__":
    print("[*] Starting AI Smart Attendance REST API on http://127.0.0.1:5000 ...")
    app.run(host="127.0.0.1", port=5000, debug=False)
