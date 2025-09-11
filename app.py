# app.py
# Usage: python app.py test1.jpg [test2.jpg ...]
# Loads known_embeddings_arcface.pkl and runs recognition; outputs attendance.csv and labeled_<image>.jpg

import os
import sys
import pickle
import cv2
import numpy as np
from insightface.app import FaceAnalysis

EMBEDDINGS_FILE = "known_embeddings_arcface.pkl"
ATTENDANCE_FILE = "attendance.csv"
DET_SIZE = (640, 640)
MAX_DIM = 1600
COSINE_THRESHOLD = 0.40   # default: tune between ~0.30 - 0.55 depending on your data

# Prevent duplicate marks within a single run
marked_this_session = set()


def load_embeddings(path=EMBEDDINGS_FILE):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Run enroll.py first.")

    with open(path, "rb") as f:
        data = pickle.load(f)

    # Accept a few possible formats:
    # 1) {"emb_dict": {...}, "ids": [...], "embeddings": np.array}
    # 2) a plain dict {id: embedding}
    if isinstance(data, dict) and "emb_dict" in data:
        emb_dict = data["emb_dict"]
    elif isinstance(data, dict) and all(isinstance(v, (list, np.ndarray)) for v in data.values()):
        emb_dict = data
    else:
        raise ValueError("Invalid embeddings file format. Re-run enroll.py to generate the correct file.")

    # ensure numpy arrays and float32
    for k, v in list(emb_dict.items()):
        emb_dict[k] = np.asarray(v, dtype=np.float32)

    ids = list(emb_dict.keys())
    embeddings = np.vstack([emb_dict[i] for i in ids]).astype(np.float32)

    # pre-normalize known embeddings for cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embeddings_norm = embeddings / norms

    return embeddings, embeddings_norm, ids, emb_dict


def mark_attendance_csv(student_id):
    if student_id in marked_this_session:
        return
    from datetime import datetime
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    write_header = not os.path.exists(ATTENDANCE_FILE)
    with open(ATTENDANCE_FILE, "a", newline="") as f:
        if write_header:
            f.write("ID,Timestamp\n")
        f.write(f"{student_id},{ts}\n")
    marked_this_session.add(student_id)


def process_image(image_path, app, embeddings_norm, ids, emb_dict):
    print(f"\n📸 Processing {image_path} ...")
    img = cv2.imread(image_path)
    if img is None:
        print(f"⚠️ Could not read {image_path}")
        return

    # resize large images for speed/robustness
    h, w = img.shape[:2]
    if max(h, w) > MAX_DIM:
        scale = MAX_DIM / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))

    faces = app.get(img)
    print(f"👥 Detected {len(faces)} faces")

    # if no known embeddings (shouldn't happen), skip quickly
    if embeddings_norm.shape[0] == 0:
        print("⚠️ No enrolled students found in embeddings.")
        return

    # prepare image copy for drawing
    out_img = img.copy()

    for face in faces:
        emb = np.asarray(face.embedding, dtype=np.float32)
        # normalize
        norm = np.linalg.norm(emb)
        if norm == 0:
            emb_norm = emb
        else:
            emb_norm = emb / norm

        # cosine similarity - fast using pre-normalized embeddings
        sims = embeddings_norm.dot(emb_norm)  # (N,)
        best_idx = int(np.argmax(sims))
        best_score = float(sims[best_idx])
        best_id = ids[best_idx]

        if best_score >= COSINE_THRESHOLD:
            label = best_id
            color = (0, 200, 0)
            print(f"✅ Recognized {best_id} (cosine={best_score:.3f})")
            mark_attendance_csv(best_id)
        else:
            label = "Unknown"
            color = (0, 0, 255)
            print(f"❌ Unknown (best={best_id}, cosine={best_score:.3f})")

        # draw bbox & label
        box = face.bbox.astype(int)
        x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
        cv2.rectangle(out_img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(out_img, f"{label} {best_score:.2f}", (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    out_name = f"labeled_{os.path.basename(image_path)}"
    cv2.imwrite(out_name, out_img)
    print(f"🖼️ Saved labeled image: {out_name}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python app.py <image1.jpg> [image2.jpg ...]")
        sys.exit(1)

    embeddings, embeddings_norm, ids, emb_dict = load_embeddings()

    # init insightface once
    app = FaceAnalysis(providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=DET_SIZE)

    for image_path in sys.argv[1:]:
        if not os.path.exists(image_path):
            print(f"Missing file: {image_path} — skipping")
            continue
        process_image(image_path, app, embeddings_norm, ids, emb_dict)

    print("\n✅ Attendance run finished.")


if __name__ == "__main__":
    main()
