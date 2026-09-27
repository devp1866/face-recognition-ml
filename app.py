import os
import shutil
import numpy as np
import cv2
from flask import Flask, render_template, request, redirect, url_for, flash
from datetime import datetime
import tempfile
from dotenv import load_dotenv

load_dotenv()

# This ensures InsightFace uses the local resources folder instead of user home
insightface_home = os.path.join(os.getcwd(), "resources")
os.environ["INSIGHTFACE_HOME"] = insightface_home
print(f" CONFIG: Set INSIGHTFACE_HOME = {insightface_home}")

#  Core Modules 
from core.recognition import FaceEngine
from core.storage import (
    init_db,
    add_user,
    get_all_embeddings,
    get_all_users,
    delete_user_by_id,
    get_user_count,
    reset_db,
    update_user_embedding,
    get_user_embedding,
)
from core.quality import FaceQualityChecker
from core.embedding_tracker import (
    compute_drift,
    apply_ema_update,
    record_drift,
    get_drift_status,
    get_all_drift_statuses,
    clear_drift_for_user,
)


app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY")
if not app.secret_key:
    raise RuntimeError(
        "SECRET_KEY is not set. Add it to your .env file before starting the app."
    )

# Use temp directory for Vercel/Render/AWS compatibility (ephemeral storage)
TEMP_DIR = tempfile.gettempdir()

app.config["UPLOAD_FOLDER"] = os.path.join(TEMP_DIR, "uploads")
app.config["RESULT_FOLDER"] = os.path.join(TEMP_DIR, "results")
app.config["DATASET_FOLDER"] = os.path.join(TEMP_DIR, "dataset")
app.config["MAX_CONTENT_LENGTH"] = 128 * 1024 * 1024

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(app.config["RESULT_FOLDER"], exist_ok=True)
os.makedirs(app.config["DATASET_FOLDER"], exist_ok=True)

#  App-Level Constants
MAX_USERS = 10

# TESTING: Skip liveness for uploaded images (flat files always fail liveness).
# Set to False to enforce liveness on all sources.
SKIP_UPLOAD_LIVENESS = True

# Subjects / Periods — edit this list to match your class schedule
SUBJECTS = [
    "General",
    "Mathematics",
    "Physics",
    "Chemistry",
    "English",
    "Computer Science",
]

# Motion detection threshold for webcam multi-frame analysis.
# Mean absolute pixel difference between frames:
#   Real face  → ~0.8–8 (natural micro-motion: breathing, eye movement)
#   Still photo → ~0–0.5 (perfectly static)
# Lowered from 1.5 → 0.8 to reduce false-positive FAKE detections for real users.
MOTION_THRESHOLD = 0.8

# Attendance log file (CSV)
ATTENDANCE_FILE = os.path.join(TEMP_DIR, "attendance.csv")

# AI Engine
# Using 'buffalo_l' for high-quality recognition as required for deployment.
face_engine = FaceEngine(model_name="buffalo_l", ctx_id=0, det_size=(640, 640))

# Quality Gating for enrollment
quality_checker = FaceQualityChecker()

init_db()


def check_motion(frames):
    """
    Check whether there is natural motion between a sequence of video frames.

    A real human face produces micro-motion (breathing, eye movement, subtle
    head shift). A photo or screen held in front of the camera is perfectly
    static. We exploit this by computing the mean absolute pixel difference
    between consecutive greyscale frames.

    Args:
        frames: List of BGR images (numpy arrays) from consecutive captures.

    Returns:
        True  → motion detected (likely a real person)
        False → no motion (likely a spoofing attempt with a static image)
    """
    if len(frames) < 2:
        # Cannot compute motion with a single frame — give benefit of the doubt
        return True

    gray_frames = [
        cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32) for f in frames
    ]
    diffs = [
        np.abs(gray_frames[i] - gray_frames[i + 1])
        for i in range(len(gray_frames) - 1)
    ]
    mean_motion = float(np.mean([np.mean(d) for d in diffs]))
    print(f"[MOTION] score: {mean_motion:.4f}  (threshold: {MOTION_THRESHOLD})")
    return mean_motion >= MOTION_THRESHOLD


def mark_attendance_csv(user_id, name, subject="General"):
    """
    Write an attendance record if the user has not been marked today for this subject.
    Supports per-subject duplicate checking so the same person can attend
    multiple classes in one day without being blocked.
    CSV format: ID,Name,Timestamp,Subject
    """
    now = datetime.now()
    ts = now.strftime("%Y-%m-%d %H:%M:%S")
    date_str = now.strftime("%Y-%m-%d")

    # Check for duplicate entry today for the same subject
    already_marked = False
    if os.path.exists(ATTENDANCE_FILE):
        with open(ATTENDANCE_FILE, "r") as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) >= 3:
                    csv_id = parts[0]
                    csv_ts = parts[2]
                    csv_subject = parts[3] if len(parts) >= 4 else "General"
                    if (str(csv_id) == str(user_id)
                            and date_str in csv_ts
                            and csv_subject == subject):
                        already_marked = True
                        break

    if not already_marked:
        write_header = not os.path.exists(ATTENDANCE_FILE)
        with open(ATTENDANCE_FILE, "a", newline="") as f:
            if write_header:
                f.write("ID,Name,Timestamp,Subject\n")
            f.write(f"{user_id},{name},{ts},{subject}\n")


@app.route("/")
def index():
    total_users = get_user_count()

    attendance_count = 0
    if os.path.exists(ATTENDANCE_FILE):
        with open(ATTENDANCE_FILE, "r") as f:
            lines = f.readlines()
            if len(lines) > 1:
                today = datetime.now().strftime("%Y-%m-%d")
                attendance_count = sum(1 for line in lines[1:] if today in line)

    return render_template(
        "index.html", total_users=total_users, attendance_count=attendance_count
    )


@app.route("/enroll", methods=["GET", "POST"])
def enroll():
    if request.method == "POST":
        if get_user_count() >= MAX_USERS:
            flash(
                f"Maximum limit of {MAX_USERS} users reached. Please delete some users first.",
                "error",
            )
            return redirect(url_for("enroll"))

        name = request.form.get("name")
        files = request.files.getlist("file")

        if not name or not files:
            flash("Name and files are required", "error")
            return redirect(request.url)

        if len(files) > 4:
            flash("Maximum 4 images allowed", "error")
            return redirect(request.url)

        embeddings_list = []
        quality_reports = []   # Per-image quality report for template
        rejected_count  = 0

        for file in files:
            if file.filename == "":
                continue

            img_bytes = file.read()
            img = face_engine.process_image(img_bytes)

            if img is None:
                quality_reports.append({
                    "filename": file.filename,
                    "passed": False,
                    "issues": ["Could not decode image — invalid or corrupt file"],
                    "scores": {},
                })
                rejected_count += 1
                continue

            # ── AI Quality Gate ─────────────────────────────────────────────
            emb, face_obj = face_engine.get_best_face_embedding(img)

            if face_obj is None:
                quality_reports.append({
                    "filename": file.filename,
                    "passed": False,
                    "issues": ["No face detected in this image"],
                    "scores": {},
                })
                rejected_count += 1
                continue

            passed, issues, scores = quality_checker.assess(img, face_obj)
            quality_reports.append({
                "filename": file.filename,
                "passed":   passed,
                "issues":   issues,
                "scores":   scores,
            })

            if not passed:
                rejected_count += 1
                continue
            # ────────────────────────────────────────────────────────────────

            if emb is not None:
                embeddings_list.append(emb)

        if not embeddings_list:
            issues_summary = "; ".join(
                r["issues"][0] for r in quality_reports if not r["passed"] and r["issues"]
            )
            flash(
                f"No valid faces accepted. Quality issues: {issues_summary or 'No faces detected'}.",
                "error",
            )
            return redirect(request.url)

        # Average all accepted embeddings into one representative vector
        avg_embedding = np.mean(embeddings_list, axis=0)
        user_id = add_user(name, avg_embedding)

        accepted = len(embeddings_list)
        total    = len([r for r in quality_reports if r["filename"]])
        msg = f"Successfully enrolled {name} (ID: {user_id}) — {accepted}/{total} images accepted by AI quality check."
        if rejected_count > 0:
            msg += f" {rejected_count} image(s) were rejected due to quality issues."
        flash(msg, "success")
        return redirect(url_for("enroll"))

    return render_template("enroll.html")


@app.route("/attendance", methods=["GET", "POST"])
def attendance():
    if request.method == "POST":
        source = request.form.get("source", "upload")

        # ── Load known embeddings ──────────────────────────────────────────────
        ids, names, known_embeddings = get_all_embeddings()

        # ──────────────────────────────────────────────────────────────
        # WEBCAM PATH — Active liveness via blink detection
        # Blink detection using InsightFace 106-point landmarks is the primary gate.
        # ──────────────────────────────────────────────────────────────
        if source == "webcam":
            # Collect up to 10 frames (JS sends frame_0 … frame_9)
            frames = []
            for i in range(10):
                ff = request.files.get(f"frame_{i}")
                if ff:
                    frame_img = face_engine.process_image(ff.read())
                    if frame_img is not None:
                        frames.append(frame_img)

            if not frames:
                flash("No valid frames received from webcam.", "error")
                return redirect(request.url)

            # ── Blink detection (active liveness) ────────────────────────────
            blink_ok, ear_seq, blink_debug = face_engine.detect_blink_in_sequence(frames)

            print(
                f"👁️  Blink detected: {blink_ok} | "
                f"min_EAR={blink_debug.get('min_ear', 'N/A')} "
                f"max_EAR={blink_debug.get('max_ear', 'N/A')} | "
                f"frames={blink_debug.get('valid_frames', 0)}/{blink_debug.get('total_frames', 0)}"
            )

            img = frames[-1]  # Use the freshest frame for recognition

            if not blink_ok:
                # ── No blink → SPOOF ────────────────────────────────────────
                reason = (
                    "NO LANDMARK"
                    if blink_debug.get("valid_frames", 0) < 2
                    else "NO BLINK"
                )
                out_img, results = face_engine.annotate_as_spoof(img, reason)

            else:
                # ── Blink confirmed → run recognition only (liveness already passed) ─
                out_img, results = face_engine.recognize_faces(
                    img,
                    known_embeddings,
                    ids,
                    names,
                    skip_liveness=True,   # blink detection IS the liveness gate
                )

        # ──────────────────────────────────────────────────────────────
        # UPLOAD PATH — unchanged; liveness skipped (SKIP_UPLOAD_LIVENESS = True)
        # ──────────────────────────────────────────────────────────────
        else:
            file = request.files.get("file")
            if not file:
                flash("File required", "error")
                return redirect(request.url)

            img = face_engine.process_image(file.read())
            if img is None:
                flash("Invalid image", "error")
                return redirect(request.url)

        # Run face recognition (with or without liveness)
        skip_liveness = (source == "upload") and SKIP_UPLOAD_LIVENESS
        # Webcam frames are JPEG-compressed and lighting-variable; use a
        # lower cosine-similarity threshold so enrolled users are still matched.
        recog_threshold = 0.42 if source == "webcam" else 0.50
        out_img, results = face_engine.recognize_faces(
            img,
            known_embeddings,
            ids,
            names,
            skip_liveness=skip_liveness,
            threshold=recog_threshold,
        )

        # Motion-check override for webcam
        if source == "webcam" and not motion_ok:
            print(" Motion check FAILED — marking all faces as FAKE (static image)")
            out_img = img.copy()  # Redraw from scratch
            for r in results:
                r["is_real"] = False
                # Preserve the recognised name inside the FAKE label if matched
                if r.get("user_id") is not None:
                    r["name"] = f"FAKE: {r['name']}"
                else:
                    r["name"] = "Unknown"
                r["user_id"] = None  # Prevent attendance being marked

                # Redraw box in red with STATIC SPOOF label
                x1, y1, x2, y2 = r["box"]
                cv2.rectangle(out_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(
                    out_img,
                    f"STATIC SPOOF ({r['score']:.2f})",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                )

        # Mark attendance for confirmed real, matched faces
        # Also: run Adaptive Embedding Learning (EMA update)
        subject = request.form.get("subject", "General")
        for r in results:
            uid = r.get("user_id")
            if uid is not None and r["is_real"]:
                mark_attendance_csv(uid, r["name"], subject)

                # ── Adaptive Embedding Learning ──────────────────────────────
                # Only update if confidence is high enough (avoid noisy updates)
                if r["score"] >= 0.65:
                    stored_emb = get_user_embedding(uid)
                    if stored_emb is not None:
                        # Reconstruct current embedding from the result
                        # We need the raw face embedding — re-get from recognized faces
                        try:
                            # Get fresh face embedding from the result image
                            faces = face_engine.get_faces(img)
                            for face in faces:
                                fx1, fy1, fx2, fy2 = face.bbox.astype(int)
                                # Match this face to the result by box proximity
                                rx1, ry1, rx2, ry2 = r["box"]
                                if abs(fx1 - rx1) < 20 and abs(fy1 - ry1) < 20:
                                    current_emb = face.embedding
                                    drift = compute_drift(stored_emb, current_emb)
                                    updated_emb = apply_ema_update(stored_emb, current_emb)
                                    update_user_embedding(uid, updated_emb)
                                    record_drift(uid, drift)
                                    print(f"[ADAPT] Adaptive update: user {uid}, drift={drift:.4f}")
                                    break
                        except Exception as e:
                            print(f"[WARN] Adaptive embedding update failed: {e}")
                # ─────────────────────────────────────────────────────────────

        # ── Save annotated result image ───────────────────────────────────
        filename = f"result_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
        cv2.imwrite(os.path.join(app.config["RESULT_FOLDER"], filename), out_img)

        return render_template(
            "attendance.html",
            result_image=filename,
            results=results,
            active_tab=source,
            skip_upload_liveness=SKIP_UPLOAD_LIVENESS,
            subjects=SUBJECTS,
            selected_subject=request.form.get("subject", "General"),
        )

    return render_template(
        "attendance.html",
        active_tab="upload",
        skip_upload_liveness=SKIP_UPLOAD_LIVENESS,
        subjects=SUBJECTS,
        selected_subject="General",
    )


@app.route("/group-attendance", methods=["GET", "POST"])
def group_attendance():
    """Multi-face group photo attendance — mark all recognized faces at once."""
    if request.method == "POST":
        file = request.files.get("file")
        if not file:
            flash("Please upload a group photo.", "error")
            return redirect(request.url)

        img = face_engine.process_image(file.read())
        if img is None:
            flash("Invalid image file.", "error")
            return redirect(request.url)

        # Load all known embeddings
        ids, names, known_embeddings = get_all_embeddings()

        # Recognize ALL faces in the image simultaneously
        out_img, results = face_engine.recognize_faces(
            img,
            known_embeddings,
            ids,
            names,
            skip_liveness=True,  # Group photos are always uploads
        )

        # Mark attendance for every recognized real face
        subject = request.form.get("subject", "General")
        marked = []
        unknown_count = 0
        for r in results:
            uid = r.get("user_id")
            if uid is not None and r["is_real"] and r["is_authentic"]:
                mark_attendance_csv(uid, r["name"], subject)
                marked.append(r["name"])
            elif r["name"] == "Unknown":
                unknown_count += 1

        # Save annotated result image
        filename = f"group_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
        save_path = os.path.join(app.config["RESULT_FOLDER"], filename)
        cv2.imwrite(save_path, out_img)

        summary = {
            "total_faces":    len(results),
            "recognized":     len(marked),
            "unknown":        unknown_count,
            "marked_names":   marked,
            "subject":        subject,
        }

        return render_template(
            "group_attendance.html",
            result_image=filename,
            results=results,
            summary=summary,
            subjects=SUBJECTS,
            selected_subject=subject,
        )

    return render_template("group_attendance.html", subjects=SUBJECTS, selected_subject="General")


@app.route("/logs")
def logs():
    search_name    = request.args.get("name", "").lower()
    search_date    = request.args.get("date", "")
    search_subject = request.args.get("subject", "")

    logs_data = []
    if os.path.exists(ATTENDANCE_FILE):
        with open(ATTENDANCE_FILE, "r") as f:
            lines = f.readlines()
            if len(lines) > 1:
                for line in lines[1:]:
                    parts = line.strip().split(",")
                    if len(parts) >= 3:
                        uid  = parts[0]
                        name = parts[1]
                        ts   = parts[2]
                        subject = parts[3] if len(parts) >= 4 else "General"

                        if search_name and search_name not in name.lower():
                            continue
                        if search_date and search_date not in ts:
                            continue
                        if search_subject and search_subject != subject:
                            continue

                        logs_data.append({
                            "id":      uid,
                            "name":    name,
                            "time":    ts,
                            "subject": subject,
                        })

    logs_data.reverse()  # Most recent first
    return render_template("logs.html", logs=logs_data, subjects=SUBJECTS)


@app.route("/results/<filename>")
def serve_result(filename):
    from flask import send_from_directory
    return send_from_directory(app.config["RESULT_FOLDER"], filename)


@app.route("/analytics")
def analytics():
    """Analytics dashboard — attendance trends, per-user stats, subject breakdown."""
    import json
    from collections import defaultdict
    from datetime import timedelta

    daily_counts   = defaultdict(int)
    user_counts    = defaultdict(int)
    subject_counts = defaultdict(int)
    today          = datetime.now().strftime("%Y-%m-%d")
    today_count    = 0
    total_records  = 0

    if os.path.exists(ATTENDANCE_FILE):
        with open(ATTENDANCE_FILE, "r") as f:
            lines = f.readlines()
            for line in lines[1:]:   # skip header
                parts = line.strip().split(",")
                if len(parts) >= 3:
                    name    = parts[1]
                    ts      = parts[2]
                    subject = parts[3] if len(parts) >= 4 else "General"
                    date    = ts[:10]   # YYYY-MM-DD

                    daily_counts[date]     += 1
                    user_counts[name]      += 1
                    subject_counts[subject] += 1
                    total_records          += 1

                    if date == today:
                        today_count += 1

    # Last 7 days bar chart data
    last_7_days = []
    for i in range(6, -1, -1):
        d = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
        last_7_days.append({"date": d[-5:], "count": daily_counts.get(d, 0)})  # MM-DD label

    # Top users (sorted by attendance count)
    top_users = sorted(user_counts.items(), key=lambda x: x[1], reverse=True)[:10]

    total_enrolled   = get_user_count()
    attendance_rate  = round(today_count / total_enrolled * 100, 1) if total_enrolled > 0 else 0
    avg_daily        = round(total_records / 7, 1) if total_records > 0 else 0

    return render_template(
        "analytics.html",
        today_count      = today_count,
        total_enrolled   = total_enrolled,
        attendance_rate  = attendance_rate,
        total_records    = total_records,
        avg_daily        = avg_daily,
        # Chart.js data as JSON strings
        chart_dates      = json.dumps([d["date"] for d in last_7_days]),
        chart_daily      = json.dumps([d["count"] for d in last_7_days]),
        chart_user_names = json.dumps([u[0] for u in top_users]),
        chart_user_data  = json.dumps([u[1] for u in top_users]),
        chart_subj_names = json.dumps(list(subject_counts.keys())),
        chart_subj_data  = json.dumps(list(subject_counts.values())),
    )


@app.route("/download_attendance")
def download_attendance():
    from flask import send_file
    if os.path.exists(ATTENDANCE_FILE):
        return send_file(
            ATTENDANCE_FILE, as_attachment=True, download_name="attendance_logs.csv"
        )
    else:
        flash("No attendance logs found.", "error")
        return redirect(url_for("logs"))


@app.route("/users")
def users():
    all_users = get_all_users()
    # Attach drift status to each user
    drift_statuses = get_all_drift_statuses()
    for user in all_users:
        user["drift"] = drift_statuses.get(user["id"], {
            "status": "stable", "avg_drift": 0.0,
            "scan_count": 0, "label": "No scans yet",
        })
    return render_template("users.html", users=all_users)


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/delete_user/<int:user_id>", methods=["POST"])
def delete_user(user_id):
    delete_user_by_id(user_id)
    clear_drift_for_user(user_id)  # Clean up drift history

    user_dir = os.path.join(app.config["DATASET_FOLDER"], str(user_id))
    if os.path.exists(user_dir):
        shutil.rmtree(user_dir)

    flash(f"User {user_id} deleted successfully.", "success")
    return redirect(url_for("users"))


@app.route("/reset", methods=["POST"])
def reset_app():
    action = request.form.get("action")
    if action == "RESTART":
        reset_db()

        for key in ["DATASET_FOLDER", "RESULT_FOLDER", "UPLOAD_FOLDER"]:
            folder = app.config.get(key)
            if folder and os.path.exists(folder):
                shutil.rmtree(folder)
                os.makedirs(folder)

        if os.path.exists(ATTENDANCE_FILE):
            os.remove(ATTENDANCE_FILE)

        flash("App has been reset successfully.", "success")
    else:
        flash('Invalid confirmation code. Type "RESTART" to reset.', "error")

    return redirect(url_for("index"))


if __name__ == "__main__":
    print("[START] Starting Flask App...")
    print(f"[DIR] Runtime Storage (Uploads/Results): {TEMP_DIR}")
    print(f"[AI]  AI Model Storage (Weights): {os.environ.get('INSIGHTFACE_HOME')}")
    app.run(debug=True, port=5000)
    # Production: app.run(host="0.0.0.0", port=5000, debug=False)
