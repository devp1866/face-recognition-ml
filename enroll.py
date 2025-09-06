# enroll.py
import os
import cv2
import numpy as np
import json
import argparse
from pathlib import Path
from utils import get_embedding

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

def enroll_student(student_id, folder):
    embeddings = []
    image_files = list(Path(folder).glob("*.jpg")) + list(Path(folder).glob("*.png"))
    
    print(f"Processing {len(image_files)} images for student {student_id}...")
    for img_path in image_files:
        img = cv2.imread(str(img_path))
        # DeepFace expects BGR, which cv2.imread provides, so no color conversion needed here.
        emb = get_embedding(img)
        if emb is not None:
            embeddings.append(emb)

    if len(embeddings) == 0:
        print(f"❌ No valid faces found for {student_id}. Please use clear, frontal images.")
        return

    # Create the average prototype embedding
    proto = np.mean(embeddings, axis=0)
    save_proto(student_id, proto)
    print(f"✅ Enrolled {student_id} with {len(embeddings)} images.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enroll a student using their photos.")
    parser.add_argument("--id", required=True, help="Student ID")
    parser.add_argument("--folder", required=True, help="Folder with student images")
    args = parser.parse_args()

    enroll_student(args.id, args.folder)