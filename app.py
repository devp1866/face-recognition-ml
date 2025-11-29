import os
import sqlite3
import cv2
import shutil
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, jsonify, flash
from insightface.app import FaceAnalysis
from database import (
    init_db,
    add_user,
    get_all_embeddings,
    get_all_users,
    delete_user_by_id,
    get_user_count,
    reset_db,
)
from datetime import datetime

import tempfile

app = Flask(__name__)
app.secret_key = "supersecretkey"  # Required for flash messages

# Use temp directory for Vercel compatibility
TEMP_DIR = tempfile.gettempdir()
os.environ["INSIGHTFACE_HOME"] = TEMP_DIR  # Set home for model downloads

app.config["UPLOAD_FOLDER"] = os.path.join(TEMP_DIR, "uploads")
app.config["RESULT_FOLDER"] = os.path.join(TEMP_DIR, "results")
app.config["DATASET_FOLDER"] = os.path.join(TEMP_DIR, "dataset")
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB max upload

# Ensure directories exist
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(app.config["RESULT_FOLDER"], exist_ok=True)
os.makedirs(app.config["DATASET_FOLDER"], exist_ok=True)

# Initialize InsightFace
face_app = FaceAnalysis(providers=["CPUExecutionProvider"])
face_app.prepare(ctx_id=0, det_size=(640, 640))

# Initialize DB
init_db()

ATTENDANCE_FILE = os.path.join(TEMP_DIR, "attendance.csv")
COSINE_THRESHOLD = 0.40
MAX_USERS = 10


def mark_attendance(user_id, name):
    """Mark attendance in CSV file with ID, Name, Timestamp."""
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
                    # Format: ID,Name,Timestamp
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

        embeddings = []
        valid_images = []

        # Process each image
        for file in files:
            if file.filename == "":
                continue

            img_bytes = file.read()
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img is None:
                continue

            faces = face_app.get(img)
            if not faces:
                continue

            # Pick largest face
            face = max(
                faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
            )
            embeddings.append(face.embedding)
            valid_images.append(img)

        if not embeddings:
            flash("No faces detected in any of the uploaded images", "error")
            return redirect(request.url)

        # Compute Average Embedding
        avg_embedding = np.mean(embeddings, axis=0)

        # Save to DB
        user_id = add_user(name, avg_embedding)

        # Save Reference Images
        user_dir = os.path.join(app.config["DATASET_FOLDER"], str(user_id))
        if not os.path.exists(user_dir):
            os.makedirs(user_dir, exist_ok=True)

        for idx, img in enumerate(valid_images):
            save_path = os.path.join(user_dir, f"{name}_{idx+1}.jpg")
            cv2.imwrite(save_path, img)

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

        # 1. Load Embeddings
        ids, names, known_embeddings = get_all_embeddings()

        # 2. Read Image
        img_bytes = file.read()
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            flash("Invalid image", "error")
            return redirect(request.url)

        # Save the uploaded file to uploads folder
        if not os.path.exists(app.config["UPLOAD_FOLDER"]):
            os.makedirs(app.config["UPLOAD_FOLDER"])

        upload_filename = f"upload_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
        upload_path = os.path.join(app.config["UPLOAD_FOLDER"], upload_filename)
        cv2.imwrite(upload_path, img)

        # 3. Detect Faces
        faces = face_app.get(img)
        out_img = img.copy()

        results = []

        # Prepare known embeddings if they exist
        known_norm = None
        if known_embeddings.size > 0:
            norms = np.linalg.norm(known_embeddings, axis=1, keepdims=True)
            known_norm = known_embeddings / norms

        for face in faces:
            box = face.bbox.astype(int)
            x1, y1, x2, y2 = box[0], box[1], box[2], box[3]

            name = "Unknown"
            best_score = 0.0
            color = (0, 0, 255)  # Red

            if known_norm is not None:
                emb = face.embedding
                emb_norm = emb / np.linalg.norm(emb)

                # Cosine Similarity
                sims = np.dot(known_norm, emb_norm)
                best_idx = np.argmax(sims)
                best_score = sims[best_idx]

                if best_score >= COSINE_THRESHOLD:
                    user_id = ids[best_idx]
                    name = names[user_id]
                    color = (0, 200, 0)  # Green
                    mark_attendance(user_id, name)

            label = f"{name} ({best_score:.2f})"

            # Draw
            cv2.rectangle(out_img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                out_img,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2,
            )

            results.append({"name": name, "score": float(best_score)})

        # Save Result
        if not os.path.exists(app.config["RESULT_FOLDER"]):
            os.makedirs(app.config["RESULT_FOLDER"])

        filename = f"result_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
        save_path = os.path.join(app.config["RESULT_FOLDER"], filename)
        cv2.imwrite(save_path, out_img)

        return render_template(
            "attendance.html", result_image=filename, results=results
        )

    return render_template("attendance.html")


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

        # 2. Clear Dataset
        if os.path.exists(app.config["DATASET_FOLDER"]):
            shutil.rmtree(app.config["DATASET_FOLDER"])
            os.makedirs(app.config["DATASET_FOLDER"])

        # 3. Clear Results
        if os.path.exists(app.config["RESULT_FOLDER"]):
            shutil.rmtree(app.config["RESULT_FOLDER"])
            os.makedirs(app.config["RESULT_FOLDER"])

        # 4. Clear Uploads
        if os.path.exists(app.config["UPLOAD_FOLDER"]):
            shutil.rmtree(app.config["UPLOAD_FOLDER"])
            os.makedirs(app.config["UPLOAD_FOLDER"])

        # 5. Clear Logs
        if os.path.exists(ATTENDANCE_FILE):
            os.remove(ATTENDANCE_FILE)

        flash("App has been reset successfully.", "success")
    else:
        flash('Invalid confirmation code. Type "RESTART" to reset.', "error")

    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(debug=True, port=5000)
