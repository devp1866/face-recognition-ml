import os
import cv2
import numpy as np
from insightface.app import FaceAnalysis

# Import AntiSpoofing
from core.antispoof import AntiSpoofDet


class FaceEngine:
    def __init__(self, model_name="buffalo_l", ctx_id=0, det_size=(640, 640)):
        """
        Initialize the InsightFace model and Anti-Spoofing model.
        """
        print(f"🧠 Initializing FaceEngine with model: {model_name}")
        # Explicitly pass 'root' to force download/load from resources folder
        resources_path = os.path.join(os.getcwd(), "resources")
        self.app = FaceAnalysis(
            name=model_name, root=resources_path, providers=["CPUExecutionProvider"]
        )
        self.app.prepare(ctx_id=ctx_id, det_size=det_size)

        # Cleanup: Remove the downloaded zip file if it exists to save space
        try:
            zip_path = os.path.join(resources_path, "models", f"{model_name}.zip")
            if os.path.exists(zip_path):
                os.remove(zip_path)
                print(f"🧹 Cleanup: Removed temporary file {model_name}.zip")
        except Exception as e:
            print(f"⚠️ Warning: Could not remove zip file: {e}")

        # Initialize Liveness Detector
        spoof_model_path = os.path.join(
            os.getcwd(), "resources", "models", "minifasv2.onnx"
        )
        self.spoof_det = None
        if os.path.exists(spoof_model_path):
            print("🛡️ Initializing Anti-Spoofing Model...")
            self.spoof_det = AntiSpoofDet(spoof_model_path)
        else:
            print(f"⚠️ Warning: Anti-Spoofing model not found at {spoof_model_path}")

    def process_image(self, image_bytes):
        """Decode image bytes and return image array."""
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img

    def get_faces(self, img):
        """
        Detect faces in the image.
        Returns a list of InsightFace objects (containing bbox, embedding, etc.)
        """
        if img is None:
            return []
        return self.app.get(img)

    def get_best_face_embedding(self, img):
        """
        Detect faces and return the embedding of the largest face.
        Returns: (embedding, face_object) or (None, None)
        """
        faces = self.get_faces(img)
        if not faces:
            return None, None

        # Pick largest face
        face = max(
            faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
        )
        return face.embedding, face

    def check_liveness(self, img, bbox):
        """
        Check if a face is Real or Fake/Spoof.
        Returns: (is_real: bool, score: float, label: str)
        """
        if not self.spoof_det:
            return True, 1.0, "Real (No Model)"

        label, score = self.spoof_det.predict(img, bbox)
        is_real = label == "Real"
        return is_real, score, label

    def recognize_faces(self, img, known_embeddings, ids, names, threshold=0.40):
        """
        Detect and recognize faces in the image against known embeddings.
        Also performs Liveness Detection.
        """
        faces = self.get_faces(img)
        out_img = img.copy()
        results = []

        # Pre-normalize known embeddings if they exist
        known_norm = None
        if known_embeddings is not None and known_embeddings.size > 0:
            norms = np.linalg.norm(known_embeddings, axis=1, keepdims=True)
            known_norm = known_embeddings / norms

        for face in faces:
            box = face.bbox.astype(int)
            x1, y1, x2, y2 = box[0], box[1], box[2], box[3]

            # 1. Check Liveness
            is_real = True
            liveness_score = 0.0
            liveness_label = "Real"

            if self.spoof_det:
                liveness_label, liveness_score = self.spoof_det.predict(img, box)
                if liveness_label == "Fake":
                    is_real = False

            name = "Unknown"
            best_score = 0.0
            color = (0, 0, 255)  # Red default

            # 2. Recognize only if Real (or strict mode)
            # We will still process recognition but label as FAKE if spoofed

            if known_norm is not None:
                emb = face.embedding
                emb_norm = emb / np.linalg.norm(emb)

                sims = np.dot(known_norm, emb_norm)
                best_idx = np.argmax(sims)
                best_score = float(sims[best_idx])

                if best_score >= threshold:
                    user_id = ids[best_idx]
                    name_candidate = names.get(user_id, "Unknown")

                    if is_real:
                        name = name_candidate
                        color = (0, 200, 0)  # Green for Match + Real
                    else:
                        name = f"FAKE: {name_candidate}"
                        color = (0, 165, 255)  # Orange for Spoof Match

            # Override if Spoof
            if not is_real:
                color = (0, 0, 0)  # Black/Warning for Spoof
                # Or keep Orange/Red

            # Display Label
            if is_real:
                label = f"{name} ({best_score:.2f})"
            else:
                label = f"SPOOF ({liveness_score:.2f})"
                color = (0, 0, 255)  # Red for spoof

            results.append(
                {
                    "name": name,
                    "score": best_score,
                    "box": [x1, y1, x2, y2],
                    "is_real": is_real,
                    "liveness": liveness_score,
                }
            )

            # Draw
            cv2.rectangle(out_img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                out_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2
            )

        return out_img, results
