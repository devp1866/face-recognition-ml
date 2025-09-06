import os
import cv2
import numpy as np
import face_recognition
import json
import argparse

DB_FILE = "db_embeddings.json"

def save_proto(student_id, proto):
    if not os.path.exists(DB_FILE) or os.stat(DB_FILE).st_size == 0:
        db = {}
    else:
        try:
            with open(DB_FILE, "r") as f:
                db = json.load(f)
        except json.JSONDecodeError:
            db = {}

    db[str(student_id)] = proto.tolist()
    with open(DB_FILE, "w") as f:
        json.dump(db, f, indent=2)

def enroll_student(student_id, folder_path):
    encodings = []

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        image = cv2.imread(file_path)

        if image is None:
            print(f"[WARN] Could not read {file_path}")
            continue

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(rgb)
        if len(boxes) == 0:
            print(f"[WARN] No face detected in {file_path}")
            continue

        encoding = face_recognition.face_encodings(rgb, boxes)[0]
        encodings.append(encoding)

    if len(encodings) == 0:
        print("[ERROR] No faces enrolled!")
        return

    proto = np.mean(encodings, axis=0)
    save_proto(student_id, proto)
    print(f"[INFO] Enrolled student {student_id} with {len(encodings)} images.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", required=True, help="Student ID")
    parser.add_argument("--folder", required=True, help="Folder with student images")
    args = parser.parse_args()

    enroll_student(args.id, args.folder)
