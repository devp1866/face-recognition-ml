import os
import cv2
import numpy as np
from insightface.app import FaceAnalysis

# Import AntiSpoofing
from core.antispoof import AntiSpoofDet


class FaceEngine:

    DEFAULT_THRESHOLD = 0.50

    def __init__(self, model_name="buffalo_l", ctx_id=0, det_size=(640, 640)):
        """
        Initialize the InsightFace model and Anti-Spoofing model.
        """
        print(f"Initializing FaceEngine with model: {model_name}")
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
            print(f" Warning: Could not remove zip file: {e}")

        # Initialize Liveness Detector
        spoof_model_path = os.path.join(
            os.getcwd(), "resources", "models", "minifasv2.onnx"
        )
        self.spoof_det = None
        if os.path.exists(spoof_model_path):
            print(" Initializing Anti-Spoofing Model...")
            self.spoof_det = AntiSpoofDet(spoof_model_path)
        else:
            print(f" Warning: Anti-Spoofing model not found at {spoof_model_path}")

    def process_image(self, image_bytes):
        """Decode image bytes and return image array."""
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img

    def get_faces(self, img):
        """
        Detect faces in the image.
        Returns a list of InsightFace face objects (containing bbox, embedding, etc.)
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

        # Pick the largest face by bounding-box area
        face = max(
            faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
        )
        return face.embedding, face

    def check_liveness(self, img, bbox):
        """
        Check if a face region is Real or Fake/Spoof.

        Args:
            img:   Full BGR image
            bbox:  Face bounding box [x1, y1, x2, y2]

        Returns:
            (is_real: bool, score: float, label: str)
        """
        if not self.spoof_det:
            return True, 1.0, "Real (No Model)"

        label, score = self.spoof_det.predict(img, bbox)
        is_real = label == "Real"
        return is_real, score, label

    def recognize_faces(
        self,
        img,
        known_embeddings,
        ids,
        names,
        threshold=None,
        skip_liveness=False,
    ):
        """
        Detect, recognize, and optionally check liveness for all faces in an image.

        Args:
            img:              BGR image (numpy array)
            known_embeddings: (N, D) array of enrolled face embeddings
            ids:              List of user IDs corresponding to known_embeddings rows
            names:            Dict mapping user_id → name
            threshold:        Cosine similarity threshold. Defaults to DEFAULT_THRESHOLD (0.50).
            skip_liveness:    If True, liveness check is skipped and all faces treated as Real.
                              Use for uploaded images where liveness is meaningless.

        Returns:
            out_img:  Annotated copy of the image with bounding boxes and labels drawn.
            results:  List of dicts, one per detected face:
                        {
                          "user_id":  int | None,
                          "name":     str,
                          "score":    float,   # cosine similarity (0–1)
                          "box":      [x1, y1, x2, y2],
                          "is_real":  bool,
                          "liveness": float,   # real probability from antispoof
                        }
        """
        if threshold is None:
            threshold = self.DEFAULT_THRESHOLD

        faces = self.get_faces(img)
        out_img = img.copy()
        results = []

        # Pre-normalize known embeddings once (avoids repeated division in the loop)
        known_norm = None
        if known_embeddings is not None and known_embeddings.size > 0:
            norms = np.linalg.norm(known_embeddings, axis=1, keepdims=True)
            known_norm = known_embeddings / norms

        for face in faces:
            box = face.bbox.astype(int)
            x1, y1, x2, y2 = box[0], box[1], box[2], box[3]

            #  Liveness check 
            if skip_liveness:
                # TESTING: Liveness check intentionally disabled for this source.
                is_real = True
                liveness_score = 1.0
                liveness_label = "Real (Skipped)"
            else:
                if self.spoof_det:
                    liveness_label, liveness_score = self.spoof_det.predict(img, box)
                    is_real = liveness_label == "Real"
                else:
                    is_real = True
                    liveness_score = 1.0
                    liveness_label = "Real (No Model)"

            # Identity recognition
            matched_user_id = None
            name = "Unknown"
            best_score = 0.0
            color = (0, 0, 255)  # Red default (unknown / spoof)

            if known_norm is not None:
                emb = face.embedding
                emb_norm = emb / np.linalg.norm(emb)

                sims = np.dot(known_norm, emb_norm)
                best_idx = int(np.argmax(sims))
                best_score = float(sims[best_idx])

                if best_score >= threshold:
                    candidate_id = ids[best_idx]
                    name_candidate = names.get(candidate_id, "Unknown")

                    if is_real:
                        # Confirmed: real person + recognised → green
                        matched_user_id = candidate_id
                        name = name_candidate
                        color = (0, 200, 0)
                    else:
                        # Spoof attempt by a known person → orange warning
                        name = f"FAKE: {name_candidate}"
                        color = (0, 165, 255)

            # Override to red if spoofing regardless of match
            if not is_real:
                color = (0, 0, 255)

            # Build display label
            if is_real:
                label_text = f"{name} ({best_score:.2f})"
            else:
                label_text = f"SPOOF ({liveness_score:.2f})"

            # Draw on image 
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

            results.append(
                {
                    "user_id": matched_user_id,
                    "name": name,
                    "score": best_score,
                    "box": [x1, y1, x2, y2],
                    "is_real": is_real,
                    "liveness": liveness_score,
                }
            )

        return out_img, results
