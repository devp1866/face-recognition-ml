# calibrate_threshold.py
import json
import numpy as np
import itertools
from utils import cosine_similarity

DB_FILE = "db_embeddings.json"

def load_db():
    with open(DB_FILE, "r") as f:
        return json.load(f)

def calibrate():
    db = load_db()
    student_ids = list(db.keys())

    same_scores = []
    diff_scores = []

    # compare same student with itself (simulate slight variations)
    for sid in student_ids:
        emb = np.array(db[sid])
        # small noise to simulate different capture
        noisy = emb + np.random.normal(0, 0.01, size=emb.shape)
        same_scores.append(cosine_similarity(emb, noisy))

    # compare different students
    for sid1, sid2 in itertools.combinations(student_ids, 2):
        emb1 = np.array(db[sid1])
        emb2 = np.array(db[sid2])
        diff_scores.append(cosine_similarity(emb1, emb2))

    same_mean = np.mean(same_scores)
    diff_mean = np.mean(diff_scores)

    # pick threshold between the two distributions
    threshold = (same_mean + diff_mean) / 2

    print(f"📊 Same-student avg similarity: {same_mean:.3f}")
    print(f"📊 Different-student avg similarity: {diff_mean:.3f}")
    print(f"👉 Suggested threshold: {threshold:.3f}")

if __name__ == "__main__":
    calibrate()
