# enroll.py
# Usage: python enroll.py
# Output: known_embeddings_arcface.pkl with keys: 'emb_dict', 'ids', 'embeddings'

import os
import pickle
from pathlib import Path
import numpy as np
import cv2
from insightface.app import FaceAnalysis

DATASET_FOLDER = "dataset"
OUT_FILE = "known_embeddings_arcface.pkl"
DET_SIZE = (640, 640)
MAX_DIM = 1600  # if image is very large, resize down (keeps speed & reliability)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def read_image_try(path):
    img = cv2.imread(path)
    if img is None:
        # fallback: try binary read + imdecode (helps with some unicode/encoding path issues)
        try:
            with open(path, "rb") as f:
                arr = np.frombuffer(f.read(), np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        except Exception:
            img = None
    return img


def compute_prototype(encodings):
    """
    Robust prototype from multiple encodings:
    remove outliers (distance to median) then average.
    """
    arr = np.vstack(encodings).astype(np.float32)  # (n, dim)
    median = np.median(arr, axis=0)
    dists = np.linalg.norm(arr - median.reshape(1, -1), axis=1)
    med = np.median(dists)
    mad = np.median(np.abs(dists - med))
    thresh = med + 2.0 * (mad if mad > 1e-6 else 0.01)
    keep = dists <= thresh
    if np.sum(keep) == 0:
        return np.mean(arr, axis=0)
    return np.mean(arr[keep], axis=0)


def main():
    ds = Path(DATASET_FOLDER)
    if not ds.exists():
        raise FileNotFoundError(f"Dataset folder not found: {DATASET_FOLDER}")

    # init insightface app (CPU by default; change providers if GPU available)
    app = FaceAnalysis(providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=DET_SIZE)

    emb_dict = {}
    student_dirs = sorted([p for p in ds.iterdir() if p.is_dir()])

    for student_dir in student_dirs:
        student_id = student_dir.name
        encs = []
        for img_file in sorted(student_dir.iterdir()):
            if img_file.suffix.lower() not in IMAGE_EXTS:
                continue
            img_path = str(img_file)
            img = read_image_try(img_path)
            if img is None:
                print(f"❌ Could not read {img_path} — skipping")
                continue

            # resize if extremely large
            h, w = img.shape[:2]
            if max(h, w) > MAX_DIM:
                scale = MAX_DIM / max(h, w)
                img = cv2.resize(img, (int(w * scale), int(h * scale)))

            try:
                faces = app.get(img)
            except Exception as e:
                print(f"⚠️ Error running detector on {img_path}: {e}")
                continue

            if not faces:
                print(f"⚠️ No face detected in {img_path}")
                continue

            # if multiple faces, pick the largest bbox (likely the subject)
            best_face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
            emb = np.asarray(best_face.embedding, dtype=np.float32)
            encs.append(emb)

        if encs:
            prototype = compute_prototype(encs).astype(np.float32)
            emb_dict[student_id] = prototype
            print(f"✅ Enrolled {student_id} ({len(encs)} images -> prototype ready)")
        else:
            print(f"⚠️ Skipping {student_id}: no valid enrolment images found")

    if not emb_dict:
        raise RuntimeError("No students enrolled. Add images under dataset/<student_id>/")

    ids = list(emb_dict.keys())
    embeddings = np.vstack([emb_dict[i] for i in ids]).astype(np.float32)

    # Save consistent structure so app.py can read it
    with open(OUT_FILE, "wb") as f:
        pickle.dump({"emb_dict": emb_dict, "ids": ids, "embeddings": embeddings}, f)

    print(f"\n📦 Saved embeddings for {len(ids)} students to {OUT_FILE}")


if __name__ == "__main__":
    main()
