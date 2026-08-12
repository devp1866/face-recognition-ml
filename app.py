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

# ── App-Level Constants ──────────────────────────────────────────────────
MAX_USERS = 10

# TESTING: Skip liveness for uploaded images (flat files always fail liveness).
# Set to False to enforce liveness on all sources.
SKIP_UPLOAD_LIVENESS = True

# Attendance log
ATTENDANCE_FILE = os.path.join(TEMP_DIR, "attendance.csv")

# AI Engine
# Using 'buffalo_l' for high-quality recognition as required for deployment.
face_engine = FaceEngine(model_name="buffalo_l", ctx_id=0, det_size=(640, 640))

init_db()



def mark_attendance_csv(user_id, name):
    """Write an attendance record if the user has not been marked today."""
    now = datetime.now()
    ts = now.strftime("%Y-%m-%d %H:%M:%S")
    date_str = now.strftime("%Y-%m-%d")

    # Check for duplicate entry today
    already_marked = False
    if os.path.exists(ATTENDANCE_FILE):
        with open(ATTENDANCE_FILE, "r") as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) >= 3:
                    csv_id, _, csv_ts = parts[0], parts[1], parts[2]
                    if str(csv_id) == str(user_id) and date_str in csv_ts:
                        already_marked = True
                        break

    if not already_marked:
        write_header = not os.path.exists(ATTENDANCE_FILE)
        with open(ATTENDANCE_FILE, "a", newline="") as f:
            if write_header:
                f.write("ID,Name,Timestamp\n")
            f.write(f"{user_id},{name},{ts}\n")


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

        for file in files:
            if file.filename == "":
                continue

            img_bytes = file.read()
            img = face_engine.process_image(img_bytes)

            if img is None:
                continue

            emb, _ = face_engine.get_best_face_embedding(img)

            if emb is not None:
                embeddings_list.append(emb)

        if not embeddings_list:
            flash("No faces detected in any of the uploaded images", "error")
            return redirect(request.url)

        # Average all embeddings into a single representative vector
        avg_embedding = np.mean(embeddings_list, axis=0)
        user_id = add_user(name, avg_embedding)

        flash(f"Successfully enrolled {name} (ID: {user_id})", "success")
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

            out_img, results = face_engine.recognize_faces(
                img,
                known_embeddings,
                ids,
                names,
                skip_liveness=SKIP_UPLOAD_LIVENESS,
            )

        # ── Mark attendance for confirmed real, matched faces ─────────────────
        for r in results:
            uid = r.get("user_id")
            if uid is not None and r["is_real"]:
                mark_attendance_csv(uid, r["name"])

        # ── Save annotated result image ───────────────────────────────────
        filename = f"result_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
        cv2.imwrite(os.path.join(app.config["RESULT_FOLDER"], filename), out_img)

        return render_template(
            "attendance.html",
            result_image=filename,
            results=results,
            active_tab=source,
            skip_upload_liveness=SKIP_UPLOAD_LIVENESS,
        )

    return render_template(
        "attendance.html",
        active_tab="upload",
        skip_upload_liveness=SKIP_UPLOAD_LIVENESS,
    )


@app.route("/logs")
def logs():
    search_name = request.args.get("name", "").lower()
    search_date = request.args.get("date", "")

    logs_data = []
    if os.path.exists(ATTENDANCE_FILE):
        with open(ATTENDANCE_FILE, "r") as f:
            lines = f.readlines()
            if len(lines) > 1:
                for line in lines[1:]:
                    parts = line.strip().split(",")
                    if len(parts) >= 3:
                        uid, name, ts = parts[0], parts[1], parts[2]

                        if search_name and search_name not in name.lower():
                            continue
                        if search_date and search_date not in ts:
                            continue

                        logs_data.append({"id": uid, "name": name, "time": ts})

    logs_data.reverse()  # Most recent first
    return render_template("logs.html", logs=logs_data)


@app.route("/results/<filename>")
def serve_result(filename):
    from flask import send_from_directory
    return send_from_directory(app.config["RESULT_FOLDER"], filename)


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
    return render_template("users.html", users=all_users)


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/delete_user/<int:user_id>", methods=["POST"])
def delete_user(user_id):
    delete_user_by_id(user_id)

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
    print("🚀 Starting Flask App...")
    print(f"📁 Runtime Storage (Uploads/Results): {TEMP_DIR}")
    print(f"🧠 AI Model Storage (Weights): {os.environ.get('INSIGHTFACE_HOME')}")
    app.run(debug=True, port=5000)
    # Production: app.run(host="0.0.0.0", port=5000, debug=False)
