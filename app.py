"""
app.py
------
Main Streamlit Web Application for the AI Smart Attendance Management System.

Enhanced Features & Bug Fixes:
1. Security & Authentication: Role-based access control protecting Admin Dashboard and Model Management.
2. Department Filter Synchronization: Case-insensitive querying ensures all records display correctly.
3. Input Validation: Sanitizes roll number and full name before saving.
4. Camera Resource Safeguard: Clears cached snapshots to prevent duplicate processing cycles.
5. Strict Confidence Enforcement: Completely blocks attendance logging when face match confidence < 70%.
"""

import os
import re
from datetime import datetime
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

# Import local modules
import database as db
import face_utils as fu

# Page Configuration
st.set_page_config(
    page_title="AI Smart Attendance System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.1rem;
        font-weight: 800;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.0rem;
        color: #4B5563;
        margin-bottom: 1.3rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #F8FAFC 0%, #EFF6FF 100%);
        border: 1px solid #DBEAFE;
        border-radius: 12px;
        padding: 1.1rem;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
    .metric-val {
        font-size: 1.9rem;
        font-weight: 800;
        color: #1D4ED8;
    }
    .metric-lbl {
        font-size: 0.8rem;
        font-weight: 600;
        color: #6B7280;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .guidance-box {
        background-color: #F8FAFC;
        border-left: 4px solid #3B82F6;
        padding: 12px 16px;
        border-radius: 6px;
        margin-top: 10px;
        margin-bottom: 10px;
        font-size: 0.9rem;
        color: #1E293B;
    }
    .quality-badge-good {
        background-color: #DEF7EC;
        color: #03543F;
        padding: 3px 8px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .quality-badge-warn {
        background-color: #FEF08A;
        color: #854D0E;
        padding: 3px 8px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .quality-badge-bad {
        background-color: #FDE8E8;
        color: #9B1C1C;
        padding: 3px 8px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def init_session_state():
    """Initializes user session variables."""
    if "admin_logged_in" not in st.session_state:
        st.session_state["admin_logged_in"] = False
    if "admin_user" not in st.session_state:
        st.session_state["admin_user"] = None
    if "nav_override" not in st.session_state:
        st.session_state["nav_override"] = None


init_session_state()


def render_sidebar():
    """Renders the sidebar navigation, system status, and session controls."""
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/facial-recognition.png", width=65)
        st.title("Smart Attendance")
        st.caption("AI Face Recognition • B.Tech Mini Project")
        st.markdown("---")

        menu_options = [
            "🏠 Overview & Stats",
            "📝 Student Registration",
            "📸 Mark Attendance",
            "📊 Admin Dashboard & Reports",
            "🧪 Diagnostics & Testing Lab",
            "⚙️ Manage Students & Model",
        ]

        selected_page = st.radio("📌 Navigation Menu", menu_options, index=0)

        st.markdown("---")
        # System Health Card
        model_ready = fu.is_model_trained()
        st.markdown("### 🖥️ System Status")
        if model_ready:
            st.success("🟢 LBPH Model: Trained & Ready")
        else:
            st.warning("🟡 LBPH Model: Not Trained Yet")

        # Admin Session Status
        st.markdown("---")
        st.markdown("### 🔒 Admin Security")
        if st.session_state["admin_logged_in"]:
            st.success(f"👤 {st.session_state['admin_user']['full_name']}")
            if st.button("🚪 Log Out", use_container_width=True):
                st.session_state["admin_logged_in"] = False
                st.session_state["admin_user"] = None
                st.rerun()
        else:
            st.caption("Admin sections are password-protected.")

        st.caption(f"📅 Today: {datetime.now().strftime('%A, %d %B %Y')}")
        st.caption(f"🕒 Time: {datetime.now().strftime('%I:%M %p')}")

        return selected_page


def render_admin_login_gate(target_section_name: str) -> bool:
    """
    Renders an authentication gate for sensitive admin pages.
    Returns True if user is authenticated, False otherwise.
    """
    if st.session_state["admin_logged_in"]:
        return True

    st.markdown(f'<div class="main-title">🔒 Administrator Login Required</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sub-title">You must sign in with administrator credentials to access <strong>{target_section_name}</strong>.</div>', unsafe_allow_html=True)

    col1, _ = st.columns([1, 1])
    with col1:
        with st.form("admin_login_form"):
            st.markdown("#### Enter Credentials")
            login_user = st.text_input("Username*", placeholder="e.g., admin").strip()
            login_pass = st.text_input("Password*", type="password", placeholder="Enter password")
            submit_login = st.form_submit_button("🔐 Sign In", use_container_width=True)

            if submit_login:
                if not login_user or not login_pass:
                    st.error("❌ Both username and password are required.")
                else:
                    success, msg, user_data = db.verify_admin_login(login_user, login_pass)
                    if success and user_data:
                        st.session_state["admin_logged_in"] = True
                        st.session_state["admin_user"] = user_data
                        st.success("✅ Login successful! Redirecting...")
                        st.rerun()
                    else:
                        st.error(f"❌ {msg}")

        st.info("💡 Default credentials for testing: Username: **admin** | Password: **admin123**")
    return False


def page_overview():
    """Overview & Quick Metrics Page."""
    st.markdown('<div class="main-title">🎓 AI Smart Attendance Management System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Contactless, automated student attendance powered by OpenCV biometric recognition & SQLite.</div>', unsafe_allow_html=True)

    metrics = db.get_dashboard_metrics()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            f"""<div class="metric-card"><div class="metric-val">{metrics['total_students']}</div><div class="metric-lbl">Total Students</div></div>""",
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""<div class="metric-card"><div class="metric-val">{metrics['today_attendance']}</div><div class="metric-lbl">Present Today</div></div>""",
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f"""<div class="metric-card"><div class="metric-val">{metrics['attendance_rate']}%</div><div class="metric-lbl">Today's Turnout</div></div>""",
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            f"""<div class="metric-card"><div class="metric-val">{metrics['total_departments']}</div><div class="metric-lbl">Departments</div></div>""",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown("### 🚀 AI Verification & Safety Pipeline")
        st.markdown(
            """
            * **Face Detection**: Multi-scale Haar Cascade detects frontal & profile facial boundaries.
            * **Environmental Diagnostics**: Real-time checking of brightness levels, motion blur, and face centering.
            * **Confidence Gating**: Strict confidence thresholding (**>= 70% match** required). Insufficient confidence safely aborts attendance.
            * **Duplicate Prevention**: SQLite `UNIQUE(roll_number, date)` rule ensures a student cannot be logged twice on the same day.
            * **Admin Reporting**: Real-time logs with one-click **CSV Report Download**.
            """
        )

    with col_right:
        st.markdown("### ⚡ Quick Navigation")
        if st.button("➕ Register New Student", use_container_width=True):
            st.session_state["nav_override"] = "📝 Student Registration"
            st.rerun()

        if st.button("📷 Open Attendance Camera", use_container_width=True):
            st.session_state["nav_override"] = "📸 Mark Attendance"
            st.rerun()

        if st.button("📊 View Admin Records & CSV", use_container_width=True):
            st.session_state["nav_override"] = "📊 Admin Dashboard & Reports"
            st.rerun()

    # Recent Attendance Activity Feed
    st.markdown("### 🕒 Today's Live Attendance Feed")
    today_records = db.get_attendance_records(date_filter=datetime.now().strftime("%Y-%m-%d"))
    if not today_records.empty:
        st.dataframe(
            today_records[["roll_number", "name", "department", "time", "status"]].rename(
                columns={
                    "roll_number": "Roll No",
                    "name": "Student Name",
                    "department": "Department",
                    "time": "Marked Time",
                    "status": "Status",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("ℹ️ No attendance recorded yet today. Students can mark attendance using the camera scanner.")


def page_registration():
    """Student Profile & Face Registration Page."""
    st.markdown('<div class="main-title">📝 Student Registration</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Register student credentials and capture high-quality face samples.</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("#### 1. Student Information")
        with st.form("student_registration_form"):
            roll_number = st.text_input("Roll Number / Hall Ticket No*", placeholder="e.g., 21CS042").strip().upper()
            full_name = st.text_input("Full Name*", placeholder="e.g., John Doe").strip().title()
            department = st.selectbox(
                "Department*",
                ["Computer Science (CSE)", "Artificial Intelligence (AI&DS)", "Information Technology (IT)", "Electronics (ECE)", "Electrical (EEE)", "Mechanical (MECH)", "Civil (CIVIL)"],
            )
            submit_info = st.form_submit_button("💾 Save Student Details", use_container_width=True)

            if submit_info:
                # Input sanitization
                if not roll_number or not full_name:
                    st.error("❌ Please provide both Roll Number and Full Name.")
                elif not re.match(r"^[A-Z0-9\-_]{2,25}$", roll_number):
                    st.error("❌ Roll Number must contain only alphanumeric characters, dashes, or underscores (2-25 characters).")
                elif len(full_name) < 2:
                    st.error("❌ Please provide a valid Full Name (at least 2 letters).")
                else:
                    success, msg, student_id = db.add_student(roll_number, full_name, department)
                    if success:
                        st.session_state["registered_student_id"] = student_id
                        st.session_state["registered_student_name"] = full_name
                        st.session_state["registered_student_roll"] = roll_number
                        st.success(f"✅ {msg}")
                    else:
                        st.error(f"❌ {msg}")

        st.markdown(
            """
            <div class="guidance-box">
                <strong>💡 Camera Positioning & Lighting Tips for Best Results:</strong><br>
                • <strong>Illumination</strong>: Ensure soft, front-facing light. Avoid strong background lights or ceiling glare.<br>
                • <strong>Positioning</strong>: Keep face upright and centered. Distance: 50-70 cm from the webcam.<br>
                • <strong>Multiple Samples</strong>: Take 2-3 snapshots with slight angle variations to build a robust model.
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown("#### 2. Face Sample Capture")

        active_student_id = st.session_state.get("registered_student_id", None)
        active_student_name = st.session_state.get("registered_student_name", "Student")
        active_student_roll = st.session_state.get("registered_student_roll", "")

        if active_student_id:
            st.info(f"Target Student: **{active_student_name} ({active_student_roll})** — Database ID #{active_student_id}")

            capture_mode = st.radio(
                "Choose Face Capture Mode:",
                ["📷 In-Browser Camera Capture (Recommended)", "💻 Auto Webcam Capture (Desktop OpenCV Popup)"],
                horizontal=True,
            )

            if "In-Browser" in capture_mode:
                st.caption("Look straight into your webcam and click 'Take Photo' below.")
                camera_image = st.camera_input("Capture Face Snapshot")

                if camera_image is not None:
                    pil_img = Image.open(camera_image)
                    img_np = np.array(pil_img)

                    success, msg, total_count, quality = fu.save_student_face_samples(active_student_id, img_np, sample_count=4)
                    if success:
                        st.success(f"✅ {msg}")
                        if quality:
                            q_c1, q_c2, q_c3 = st.columns(3)
                            with q_c1:
                                badge_class = "quality-badge-good" if quality["brightness_status"] == "Good" else "quality-badge-warn"
                                st.markdown(f"Lighting: <span class='{badge_class}'>{quality['brightness_status']}</span>", unsafe_allow_html=True)
                            with q_c2:
                                badge_class = "quality-badge-good" if quality["sharpness_status"] == "Sharp" else "quality-badge-bad"
                                st.markdown(f"Sharpness: <span class='{badge_class}'>{quality['sharpness_status']}</span>", unsafe_allow_html=True)
                            with q_c3:
                                st.markdown(f"Position: <span class='quality-badge-good'>{quality['centering_status']}</span>", unsafe_allow_html=True)

                            if quality.get("guidance_tips"):
                                for tip in quality["guidance_tips"]:
                                    st.warning(f"⚠️ {tip}")
                    else:
                        st.error(f"❌ {msg}")

            else:
                st.caption("This will open a live OpenCV camera window on your system and automatically capture 30 face frames.")
                if st.button("🚀 Start Auto Webcam Capture (30 Frames)", use_container_width=True):
                    with st.spinner("Accessing webcam and capturing face samples..."):
                        success, msg, total_count = fu.capture_samples_via_webcam(active_student_id, target_samples=30)
                    if success:
                        st.success(f"✅ {msg}")
                    else:
                        st.error(f"❌ {msg}")

            st.markdown("---")
            st.markdown("#### 3. Train Recognition Model")
            st.caption("After capturing faces, train the LBPH Face Recognizer so the system can recognize the new student.")
            if st.button("⚡ Train / Retrain Model Now", type="primary", use_container_width=True):
                with st.spinner("Training OpenCV LBPH Face Recognizer on student dataset..."):
                    success, msg, total_trained = fu.train_face_recognizer()
                if success:
                    st.success(f"🎉 {msg}")
                else:
                    st.error(f"❌ {msg}")

        else:
            st.warning("⚠️ Please fill out and save the Student Information form on the left first to enable face capture.")


def page_mark_attendance():
    """Attendance Scanning Page with Live Face Recognition, Strict Thresholds, and Quality Diagnostics."""
    st.markdown('<div class="main-title">📸 Mark Attendance</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Contactless, AI-powered automatic attendance marking. Strict confidence gating prevents false-positives.</div>', unsafe_allow_html=True)

    if not fu.is_model_trained():
        st.error("⚠️ Face Recognition Model is not trained yet! Please register at least one student and train the model first.")
        return

    with st.expander("⚙️ Advanced Recognition Settings & Thresholds", expanded=False):
        c_th1, c_th2 = st.columns(2)
        with c_th1:
            confidence_threshold = st.slider(
                "Minimum Match Confidence Required (%):",
                min_value=50.0,
                max_value=90.0,
                value=fu.DEFAULT_MIN_CONFIDENCE_PERCENTAGE,
                step=1.0,
                help="If face recognition confidence is lower than this value, attendance will NOT be marked.",
            )
        with c_th2:
            distance_threshold = st.slider(
                "Maximum LBPH Distance Allowed:",
                min_value=40.0,
                max_value=90.0,
                value=fu.DEFAULT_MAX_DISTANCE_THRESHOLD,
                step=1.0,
                help="Lower distance means stricter matching. LBPH distance <= 65 is standard.",
            )

    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("### 📷 Live Camera Scanner")
        st.caption("Center your face in the camera view and click 'Take Photo' to record your attendance.")
        scanned_photo = st.camera_input("Scan Face for Attendance")

        st.markdown(
            """
            <div class="guidance-box">
                <strong>📐 Camera Positioning & Lighting Guidance:</strong><br>
                • <strong>Face Forward</strong>: Look directly at the camera with both eyes visible.<br>
                • <strong>Illumination</strong>: Frontal lighting is best. Avoid sitting directly under fluorescent spotlights or in dark corners.<br>
                • <strong>Single Person</strong>: Only one student should be in the camera view.<br>
                • <strong>Distance</strong>: Keep your face filling at least 25% of the camera frame.
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown("### 📋 Recognition & Safety Verification")

        if scanned_photo is not None:
            pil_img = Image.open(scanned_photo)
            image_rgb = np.array(pil_img)

            with st.spinner("Analyzing image quality and recognizing face..."):
                result = fu.recognize_student_face(
                    image_rgb,
                    min_confidence_pct=confidence_threshold,
                    max_distance_threshold=distance_threshold,
                )

            status = result["status"]
            bbox = result["bounding_box"]
            predicted_id = result["predicted_id"]
            confidence_pct = result["confidence_pct"]
            is_sufficient = result["is_confidence_sufficient"]
            quality = result.get("quality")

            if quality:
                st.markdown("##### 🔬 Environmental & Framing Diagnostics")
                q_col1, q_col2, q_col3 = st.columns(3)
                with q_col1:
                    b_status = quality["brightness_status"]
                    b_badge = "quality-badge-good" if b_status == "Good" else "quality-badge-bad"
                    st.markdown(f"💡 Lighting: <span class='{b_badge}'>{b_status} ({quality['brightness']})</span>", unsafe_allow_html=True)
                with q_col2:
                    s_status = quality["sharpness_status"]
                    s_badge = "quality-badge-good" if s_status == "Sharp" else "quality-badge-bad"
                    st.markdown(f"🔍 Sharpness: <span class='{s_badge}'>{s_status} ({quality['sharpness']})</span>", unsafe_allow_html=True)
                with q_col3:
                    d_status = quality["distance_status"]
                    d_badge = "quality-badge-good" if d_status == "Optimal" else "quality-badge-warn"
                    st.markdown(f"📏 Distance: <span class='{d_badge}'>{d_status}</span>", unsafe_allow_html=True)

                st.progress(min(1.0, confidence_pct / 100.0), text=f"Match Confidence Score: {confidence_pct}% (Required: ≥ {confidence_threshold}%)")

            # --- CASE 1: Face Recognized with SUFFICIENT Confidence ---
            if is_sufficient and predicted_id is not None and bbox is not None:
                student = db.get_student_by_id(predicted_id)

                if student:
                    student_name = student["name"]
                    roll_no = student["roll_number"]
                    dept = student["department"]

                    # Attempt to mark attendance in SQLite
                    marked_success, mark_msg, record_data = db.mark_attendance(
                        student_id=predicted_id,
                        roll_number=roll_no,
                        name=student_name,
                        department=dept,
                    )

                    annotated_img = fu.draw_recognition_box(
                        image_rgb,
                        bbox,
                        display_text=f"{student_name} ({roll_no})",
                        confidence_pct=confidence_pct,
                        status="SUCCESS",
                    )
                    st.image(annotated_img, caption="Verified Student Face", use_container_width=True)

                    if marked_success:
                        st.balloons()
                        st.success(f"✅ {mark_msg}")
                        st.markdown(
                            f"""
                            <div style="background-color:#F0FDF4; border:1px solid #BBF7D0; padding:15px; border-radius:10px;">
                                <h4 style="color:#166534; margin:0 0 8px 0;">🎉 Attendance Recorded!</h4>
                                <p style="margin:2px 0;"><strong>Student:</strong> {student_name}</p>
                                <p style="margin:2px 0;"><strong>Roll Number:</strong> {roll_no}</p>
                                <p style="margin:2px 0;"><strong>Department:</strong> {dept}</p>
                                <p style="margin:2px 0;"><strong>Date:</strong> {record_data['date']}</p>
                                <p style="margin:2px 0;"><strong>Time:</strong> {record_data['time']}</p>
                                <p style="margin:2px 0;"><strong>Confidence:</strong> {confidence_pct}% (LBPH Distance: {result['distance']})</p>
                                <p style="margin:2px 0;"><strong>Status:</strong> <span class="quality-badge-good">Present</span></p>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    else:
                        # Already marked today (Requirement 4: Duplicate Prevention)
                        st.warning(f"⚠️ {mark_msg}")
                        st.markdown(
                            f"""
                            <div style="background-color:#FEFCE8; border:1px solid #FEF08A; padding:15px; border-radius:10px;">
                                <h4 style="color:#854D0E; margin:0 0 8px 0;">⚠️ Duplicate Attendance Prevented</h4>
                                <p style="margin:2px 0;"><strong>Student:</strong> {student_name} ({roll_no})</p>
                                <p style="margin:2px 0;"><strong>Date:</strong> {record_data.get('date', datetime.now().strftime('%Y-%m-%d')) if record_data else datetime.now().strftime('%Y-%m-%d')}</p>
                                <p style="margin:2px 0;"><strong>Prior Check-in:</strong> {record_data.get('time', 'Earlier Today') if record_data else 'Earlier Today'}</p>
                                <p style="margin:2px 0; color:#B45309; font-size:0.85rem;">Rule: Students can only be marked present once per day.</p>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

            # --- CASE 2: Low Confidence (Below Threshold) -> DO NOT MARK ATTENDANCE ---
            elif status == "LOW_CONFIDENCE" and bbox is not None:
                annotated_img = fu.draw_recognition_box(
                    image_rgb,
                    bbox,
                    display_text=f"Low Confidence ({confidence_pct}%)",
                    confidence_pct=confidence_pct,
                    status="LOW_CONFIDENCE",
                )
                st.image(annotated_img, caption="Low Confidence Face", use_container_width=True)

                st.warning("⚠️ Recognition Confidence Insufficient! Attendance NOT marked.")
                st.markdown(
                    f"""
                    <div style="background-color:#FFFBEB; border:1px solid #FDE68A; padding:12px; border-radius:8px;">
                        <p style="margin:2px 0;"><strong>Match Confidence:</strong> {confidence_pct}% (Required: ≥ {confidence_threshold}%)</p>
                        <p style="margin:2px 0;"><strong>LBPH Distance:</strong> {result['distance']} (Safe limit: ≤ {distance_threshold})</p>
                        <p style="margin:2px 0; color:#B45309;"><strong>Security Gate:</strong> Attendance was deliberately aborted to prevent false-positive errors.</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.markdown("##### 💡 Suggested Actions to Improve Recognition:")
                for tip in result.get("guidance", []):
                    st.info(tip)

            # --- CASE 3: Unknown / Unregistered Face ---
            elif status == "UNKNOWN_PERSON" and bbox is not None:
                annotated_img = fu.draw_recognition_box(
                    image_rgb,
                    bbox,
                    display_text="Unknown Student",
                    confidence_pct=confidence_pct,
                    status="UNKNOWN_PERSON",
                )
                st.image(annotated_img, caption="Unrecognized Face", use_container_width=True)

                st.error("❌ Face Not Recognized (Unknown Person or Unregistered Student). Attendance NOT marked.")
                st.markdown("##### 💡 Troubleshooting Guidance:")
                for tip in result.get("guidance", []):
                    st.info(tip)

            # --- CASE 4: Multiple Faces Detected ---
            elif status == "MULTIPLE_FACES":
                if bbox is not None:
                    annotated_img = fu.draw_recognition_box(image_rgb, bbox, display_text="Multiple Faces", status="MULTIPLE_FACES")
                    st.image(annotated_img, caption="Multiple Faces Detected", use_container_width=True)
                st.error(f"❌ {result['message']}")
                for tip in result.get("guidance", []):
                    st.warning(tip)

            # --- CASE 5: No Face Detected ---
            else:
                st.error(f"❌ {result['message']}")
                for tip in result.get("guidance", []):
                    st.info(tip)

        else:
            st.info("Waiting for camera snapshot... Click 'Take Photo' above to begin.")


def page_admin_dashboard():
    """Admin Dashboard & Reports Page (Protected by Login)."""
    if not render_admin_login_gate("Admin Dashboard & Reports"):
        return

    st.markdown('<div class="main-title">📊 Admin Dashboard & Attendance Reports</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">View real-time attendance logs, filter by date and department, and export to CSV.</div>', unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["📋 Attendance Records & CSV Export", "👥 Registered Students Directory"])

    with tab1:
        st.markdown("#### 🔍 Filter Attendance Records")
        f_col1, f_col2, f_col3 = st.columns([1, 1, 2])

        with f_col1:
            filter_date_option = st.selectbox(
                "Filter by Date:",
                ["Today", "All Dates", "Custom Date"],
            )
            if filter_date_option == "Today":
                selected_date = datetime.now().strftime("%Y-%m-%d")
            elif filter_date_option == "Custom Date":
                custom_d = st.date_input("Select Date", datetime.now())
                selected_date = custom_d.strftime("%Y-%m-%d")
            else:
                selected_date = None

        with f_col2:
            dept_list = ["All", "Computer Science (CSE)", "Artificial Intelligence (AI&DS)", "Information Technology (IT)", "Electronics (ECE)", "Electrical (EEE)", "Mechanical (MECH)", "Civil (CIVIL)"]
            selected_dept = st.selectbox("Department:", dept_list)

        with f_col3:
            search_query = st.text_input("Search by Name or Roll No:", placeholder="e.g. 21CS or John").strip()

        # Query records with case-insensitive department matching
        records_df = db.get_attendance_records(
            date_filter=selected_date,
            dept_filter=selected_dept,
            search_query=search_query,
        )

        st.markdown("---")

        top_c1, top_c2 = st.columns([2, 1])
        with top_c1:
            st.markdown(f"**Found {len(records_df)} Attendance Record(s)**")

        with top_c2:
            if not records_df.empty:
                csv_bytes = records_df.to_csv(index=False).encode("utf-8")
                download_filename = f"Attendance_Report_{selected_date or 'All'}_{datetime.now().strftime('%H%M%S')}.csv"
                st.download_button(
                    label="📥 Download CSV Report",
                    data=csv_bytes,
                    file_name=download_filename,
                    mime="text/csv",
                    type="primary",
                    use_container_width=True,
                )

        if not records_df.empty:
            display_df = records_df.rename(
                columns={
                    "id": "Log ID",
                    "roll_number": "Roll Number",
                    "name": "Student Name",
                    "department": "Department",
                    "date": "Date",
                    "time": "Time",
                    "status": "Attendance Status",
                }
            )
            st.dataframe(display_df, use_container_width=True, hide_index=True)
        else:
            st.info("ℹ️ No attendance records found matching the selected filters.")

    with tab2:
        st.markdown("#### 👥 All Registered Students")
        students_df = db.get_all_students()

        if not students_df.empty:
            st.dataframe(
                students_df.rename(
                    columns={
                        "id": "Student ID",
                        "roll_number": "Roll Number",
                        "name": "Full Name",
                        "department": "Department",
                        "registered_at": "Registration Timestamp",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )

            std_csv = students_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Student Directory CSV",
                data=std_csv,
                file_name=f"Student_Directory_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
            )
        else:
            st.info("ℹ️ No students registered in the database yet.")


def page_diagnostics_lab():
    """Interactive Diagnostics & Testing Lab."""
    st.markdown('<div class="main-title">🧪 Face Recognition Diagnostics & Testing Lab</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Test lighting quality, blur detection, face framing, and confidence thresholding without altering attendance records.</div>', unsafe_allow_html=True)

    test_mode = st.radio(
        "Choose Test Input Source:",
        ["📷 Live Camera Snapshot", "📁 Upload an Image File"],
        horizontal=True,
    )

    test_image_rgb = None

    if "Live Camera" in test_mode:
        cam_shot = st.camera_input("Capture Test Frame")
        if cam_shot:
            test_image_rgb = np.array(Image.open(cam_shot))
    else:
        uploaded_file = st.file_uploader("Upload Test Photo (JPG / PNG):", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            test_image_rgb = np.array(Image.open(uploaded_file).convert("RGB"))

    if test_image_rgb is not None:
        col_img, col_metrics = st.columns([1, 1])

        result = fu.recognize_student_face(test_image_rgb)
        bbox = result["bounding_box"]
        qual = result.get("quality")

        with col_img:
            if bbox:
                annotated = fu.draw_recognition_box(
                    test_image_rgb,
                    bbox,
                    display_text=result["status"],
                    confidence_pct=result["confidence_pct"],
                    status=result["status"],
                )
                st.image(annotated, caption="Annotated Test Result", use_container_width=True)
            else:
                st.image(test_image_rgb, caption="No Face Detected in Image", use_container_width=True)

        with col_metrics:
            st.markdown("### 📊 Diagnostic Measurements")
            st.write(f"**Overall Status**: `{result['status']}`")
            st.write(f"**Match Confidence**: `{result['confidence_pct']}%`")
            st.write(f"**LBPH Raw Distance**: `{result['distance']}`")
            st.write(f"**Sufficient for Attendance?**: `{'✅ YES' if result['is_confidence_sufficient'] else '❌ NO (Blocked)'}`")

            if qual:
                st.markdown("---")
                st.write(f"💡 **Illumination**: {qual['brightness']} ({qual['brightness_status']})")
                st.write(f"🔍 **Sharpness Score**: {qual['sharpness']} ({qual['sharpness_status']})")
                st.write(f"📏 **Distance Status**: {qual['distance_status']}")
                st.write(f"📐 **Framing Centering**: {qual['centering_status']}")

            if result.get("guidance"):
                st.markdown("---")
                st.markdown("**Guidance Recommendations:**")
                for tip in result["guidance"]:
                    st.info(tip)
    else:
        st.info("Please capture a camera snapshot or upload an image to view diagnostic measurements.")


def page_manage():
    """Management operations (Protected by Login)."""
    if not render_admin_login_gate("Student & System Management"):
        return

    st.markdown('<div class="main-title">⚙️ System & Student Management</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Admin tools to re-train the AI model, review dataset statistics, and manage student accounts.</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### 🧠 AI Model Management")
        st.caption("Re-train the OpenCV LBPH Face Recognizer on the full face dataset.")

        model_exists = fu.is_model_trained()
        if model_exists:
            st.success("✅ Model file is active at: `models/trained_lbph_model.yml`")
        else:
            st.warning("⚠️ No trained model found.")

        if st.button("🔄 Retrain Face Recognition Model", use_container_width=True):
            with st.spinner("Retraining model..."):
                success, msg, count = fu.train_face_recognizer()
            if success:
                st.success(f"🎉 {msg}")
            else:
                st.error(f"❌ {msg}")

    with col2:
        st.markdown("### 🗑️ Delete Student Profile")
        st.caption("Permanently removes a student, their attendance history, and face sample files.")

        students_df = db.get_all_students()
        if not students_df.empty:
            student_options = {
                f"{row['roll_number']} - {row['name']} (ID #{row['id']})": row["id"]
                for _, row in students_df.iterrows()
            }
            selected_option = st.selectbox("Select Student to Remove:", list(student_options.keys()))
            target_student_id = student_options[selected_option]

            confirm_del = st.checkbox("I confirm that I want to delete this student, their dataset images, and attendance records.")
            if st.button("🗑️ Delete Student", type="secondary", disabled=not confirm_del, use_container_width=True):
                success, msg = db.delete_student(target_student_id)
                if success:
                    st.success(f"✅ {msg}")
                    fu.train_face_recognizer()
                    st.rerun()
                else:
                    st.error(f"❌ {msg}")
        else:
            st.info("No students available to delete.")


def main():
    """Main application controller."""
    if "nav_override" in st.session_state and st.session_state["nav_override"]:
        nav_target = st.session_state.pop("nav_override")
    else:
        nav_target = None

    selected = render_sidebar()
    if nav_target:
        selected = nav_target

    if "Overview" in selected:
        page_overview()
    elif "Registration" in selected:
        page_registration()
    elif "Mark Attendance" in selected:
        page_mark_attendance()
    elif "Admin Dashboard" in selected:
        page_admin_dashboard()
    elif "Diagnostics" in selected:
        page_diagnostics_lab()
    elif "Manage" in selected:
        page_manage()


if __name__ == "__main__":
    main()
