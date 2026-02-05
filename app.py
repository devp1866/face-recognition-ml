import os
import shutil
import numpy as np
import cv2
from flask import Flask, render_template, request, redirect, url_for, flash
from datetime import datetime
import tempfile

# SETUP INSIGHTFACE HOME BEFORE IMPORTS
# This ensures InsightFace uses the local resources folder instead of user home
insightface_home = os.path.join(os.getcwd(), "resources")
os.environ["INSIGHTFACE_HOME"] = insightface_home
print(f"🔧 CONFIG: Set INSIGHTFACE_HOME = {insightface_home}")

# Helper Core Modules
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
app.secret_key = "supersecretkey"

# Use temp directory for Vercel/Render/AWS compatibility (Ephemeral storage)
TEMP_DIR = tempfile.gettempdir()

app.config["UPLOAD_FOLDER"] = os.path.join(TEMP_DIR, "uploads")
app.config["RESULT_FOLDER"] = os.path.join(TEMP_DIR, "results")
app.config["DATASET_FOLDER"] = os.path.join(TEMP_DIR, "dataset")
app.config["MAX_CONTENT_LENGTH"] = 128 * 1024 * 1024

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(app.config["RESULT_FOLDER"], exist_ok=True)
os.makedirs(app.config["DATASET_FOLDER"], exist_ok=True)


# Using 'buffalo_l' for high quality as requested for deployment
face_engine = FaceEngine(model_name="buffalo_l", ctx_id=0, det_size=(640, 640))

# Initialize DB
init_db()

ATTENDANCE_FILE = os.path.join(TEMP_DIR, "attendance.csv")
MAX_USERS = 10


def mark_attendance_csv(user_id, name):
    now = datetime.now()
    ts = now.strftime("%Y-%m-%d %H:%M:%S")
    date_str = now.strftime("%Y-%m-%d")

    # Check if already marked today
    already_marked = False
    if os.path.exists(ATTENDANCE_FILE):
        with open(ATTENDANCE_FILE, "r") as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) >= 3:
                    csv_id = parts[0]
                    csv_ts = parts[2]
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
    # Get total users
    total_users = get_user_count()

    # Get today's attendance count
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
        # Check user limit
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
        valid_images = []

        # Process each image using Core Engine
        for file in files:
            if file.filename == "":
                continue

            # Read image bytes
            img_bytes = file.read()
            img = face_engine.process_image(img_bytes)

            if img is None:
                continue

            # Get best face embedding
            emb, face_obj = face_engine.get_best_face_embedding(img)

            if emb is not None:
                embeddings_list.append(emb)
                valid_images.append(img)

        if not embeddings_list:
            flash("No faces detected in any of the uploaded images", "error")
            return redirect(request.url)

        # Compute Average Embedding
        avg_embedding = np.mean(embeddings_list, axis=0)

        # Save to DB via Storage Module
        user_id = add_user(name, avg_embedding)

        # Save Reference Images (Optional, for visual confirmation)
        # aws-free-tier-optimization: Disabled to save storage space. Only embeddings are stored.
        # user_dir = os.path.join(app.config["DATASET_FOLDER"], str(user_id))
        # if not os.path.exists(user_dir):
        #     os.makedirs(user_dir, exist_ok=True)
        #
        # for idx, img in enumerate(valid_images):
        #     save_path = os.path.join(user_dir, f"{name}_{idx+1}.jpg")
        #     cv2.imwrite(save_path, img)

        flash(f"Successfully enrolled {name} (ID: {user_id})", "success")
        return redirect(url_for("enroll"))

    return render_template("enroll.html")


@app.route("/attendance", methods=["GET", "POST"])
def attendance():
    if request.method == "POST":
        file = request.files.get("file")
        if not file:
            flash("File required", "error")
            return redirect(request.url)

        # 1. Load Embeddings from DB
        ids, names, known_embeddings = get_all_embeddings()

        # 2. Read Image
        img_bytes = file.read()
        img = face_engine.process_image(img_bytes)

        if img is None:
            flash("Invalid image", "error")
            return redirect(request.url)

        # Save the uploaded file log
        upload_filename = f"upload_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
        upload_path = os.path.join(app.config["UPLOAD_FOLDER"], upload_filename)
        cv2.imwrite(upload_path, img)

        # 3. Recognize Faces using Core Engine Logic (Custom Loop for App)
        faces = face_engine.get_faces(img)
        out_img = img.copy()
        results = []

        # Pre-normalize (business logic)
        known_norm = None
        if known_embeddings.size > 0:
            norms = np.linalg.norm(known_embeddings, axis=1, keepdims=True)
            known_norm = known_embeddings / norms

        for face in faces:
            box = face.bbox.astype(int)
            x1, y1, x2, y2 = box[0], box[1], box[2], box[3]

            # 1. Liveness Check
            source = request.form.get("source", "upload")
            if source == "upload":
                # User Requirement: Skip spoof checks for upload files, assume Real
                is_real = True
                live_score = 1.0
                live_label = "Real"
            else:
                is_real, live_score, live_label = face_engine.check_liveness(img, box)

            # Draw Box & Color
            # Default: Red (Unknown or Spoof)
            color = (0, 0, 255)
            name = "Unknown"
            best_score = 0.0

            if known_norm is not None:
                emb = face.embedding
                emb_norm = emb / np.linalg.norm(emb)

                sims = np.dot(known_norm, emb_norm)
                best_idx = np.argmax(sims)
                best_score = float(sims[best_idx])

                if best_score >= 0.40:  # Threshold
                    user_id = ids[best_idx]
                    # FIX: Use user_id to look up name, NOT the index
                    name_candidate = names.get(user_id, "Unknown")

                    # LOGIC: Only Mark Attendance if REAL
                    if is_real:
                        name = name_candidate
                        color = (0, 200, 0)  # Green for Match

                        # Mark Attendance (CSV Logic)
                        mark_attendance_csv(user_id, name)
                    else:
                        name = f"SPOOF: {name_candidate}"
                        color = (0, 0, 255)  # Red for Spoof Match

            # If not matched but Real -> Unknown (Red)
            # If Spoof -> Display "SPOOF"

            if not is_real:
                label_text = f"SPOOF ({live_score:.2f})"
                color = (0, 0, 255)
            else:
                label_text = f"{name} ({best_score:.2f})"

            # Visualize
            results.append(
                {
                    "name": name,
                    "score": best_score,
                    "is_real": is_real,
                    "liveness": live_score,
                }
            )

            cv2.rectangle(out_img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                out_img,
                label_text,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2,
            )

        # Save Result
        filename = f"result_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
        save_path = os.path.join(app.config["RESULT_FOLDER"], filename)
        cv2.imwrite(save_path, out_img)

        # Determine if request came from webcam
        source = request.form.get("source", "upload")

        return render_template(
            "attendance.html", result_image=filename, results=results, active_tab=source
        )

    return render_template("attendance.html", active_tab="upload")


@app.route("/logs")
def logs():
    search_name = request.args.get("name", "").lower()
    search_date = request.args.get("date", "")

    logs_data = []
    if os.path.exists(ATTENDANCE_FILE):
        with open(ATTENDANCE_FILE, "r") as f:
            lines = f.readlines()
            if len(lines) > 1:
                # Skip header
                for line in lines[1:]:
                    parts = line.strip().split(",")
                    if len(parts) >= 3:
                        # ID,Name,Timestamp
                        uid, name, ts = parts[0], parts[1], parts[2]

                        # Filter Logic
                        if search_name and search_name not in name.lower():
                            continue
                        if search_date and search_date not in ts:
                            continue

                        logs_data.append({"id": uid, "name": name, "time": ts})

    # Sort by time desc
    logs_data.reverse()
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

    # Delete dataset folder
    user_dir = os.path.join(app.config["DATASET_FOLDER"], str(user_id))
    if os.path.exists(user_dir):
        shutil.rmtree(user_dir)

    flash(f"User {user_id} deleted successfully.", "success")
    return redirect(url_for("users"))


@app.route("/reset", methods=["POST"])
def reset_app():
    action = request.form.get("action")
    if action == "RESTART":
        # 1. Reset DB
        reset_db()

        # 2. Clear Folders
        for key in ["DATASET_FOLDER", "RESULT_FOLDER", "UPLOAD_FOLDER"]:
            folder = app.config.get(key)
            if folder and os.path.exists(folder):
                shutil.rmtree(folder)
                os.makedirs(folder)

        # 3. Clear Logs
        if os.path.exists(ATTENDANCE_FILE):
            os.remove(ATTENDANCE_FILE)

        flash("App has been reset successfully.", "success")
    else:
        flash('Invalid confirmation code. Type "RESTART" to reset.', "error")

    return redirect(url_for("index"))


if __name__ == "__main__":
    # Ensure port 5000 is used info
    print("🚀 Starting Flask App...")
    print(f"📂 Runtime Storage (Uploads/Results): {TEMP_DIR}")
    print(f"🧠 AI Model Storage (Weights): {os.environ.get('INSIGHTFACE_HOME')}")
    app.run(debug=True, port=5000)
    # Production Configuration
    # app.run(host="0.0.0.0", port=5000, debug=False)
