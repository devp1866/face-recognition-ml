import cv2
import numpy as np


class FaceQualityChecker:
    """
    AI Face Quality Gating — Pre-enrollment image quality assessment pipeline.

    Runs 4 independent checks on a detected face before its embedding is
    accepted for enrollment. Each check has a specific threshold and returns
    a human-readable rejection reason so the user knows exactly what to fix.

    Checks (in order):
        1. Blur        — Laplacian variance (sharpness measure)
        2. Brightness  — Mean pixel intensity of face region
        3. Face Size   — Face bounding-box area as % of full image
        4. Face Angle  — Yaw/pitch from InsightFace pose attribute
    """

    # ── Thresholds ─────────────────────────────────────────────────────────────
    BLUR_THRESHOLD       = 20.0    # Laplacian variance; only reject extremely blurry images
    MIN_BRIGHTNESS       = 15.0    # Mean pixel value; only reject near-black images
    MAX_BRIGHTNESS       = 245.0   # Mean pixel value; only reject completely blown out
    MIN_FACE_AREA_RATIO  = 0.01    # Face must be ≥1% of image area (very lenient)
    MAX_POSE_ANGLE       = 55.0    # Degrees; allows fairly angled faces

    def check_blur(self, img, bbox):
        """
        Compute the Laplacian variance of the face crop.
        A sharp face has high variance; a blurry one is near-zero.
        """
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        # Clamp to image bounds
        h, w = img.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        face_crop = img[y1:y2, x1:x2]
        if face_crop.size == 0:
            return 0.0, False, "Face crop is empty"

        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        passed = variance >= self.BLUR_THRESHOLD
        reason = None if passed else f"Image too blurry (score: {variance:.1f}) — hold camera steady or use better lighting"
        return variance, passed, reason

    def check_brightness(self, img, bbox):
        """
        Compute the mean pixel intensity of the face region.
        Too dark or too bright = rejected.
        """
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        h, w = img.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        face_crop = img[y1:y2, x1:x2]
        if face_crop.size == 0:
            return 0.0, False, "Face crop is empty"

        mean_brightness = float(np.mean(face_crop))
        if mean_brightness < self.MIN_BRIGHTNESS:
            return mean_brightness, False, f"Image too dark (brightness: {mean_brightness:.1f}) — improve lighting"
        if mean_brightness > self.MAX_BRIGHTNESS:
            return mean_brightness, False, f"Image overexposed (brightness: {mean_brightness:.1f}) — reduce direct light"
        return mean_brightness, True, None

    def check_face_size(self, img, bbox):
        """
        Check that the face occupies at least MIN_FACE_AREA_RATIO of the image.
        A tiny face (person too far) = poor embedding quality.
        """
        h, w = img.shape[:2]
        image_area = h * w

        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        face_area = (x2 - x1) * (y2 - y1)

        ratio = face_area / image_area if image_area > 0 else 0.0
        passed = ratio >= self.MIN_FACE_AREA_RATIO
        reason = None if passed else f"Face too small ({ratio*100:.1f}% of frame) — move closer to camera"
        return ratio, passed, reason

    def check_face_angle(self, face):
        """
        Use InsightFace's pose attribute (yaw, pitch, roll in degrees).
        Large off-axis angles produce worse embeddings.
        Returns (max_angle, passed, reason).
        """
        pose = getattr(face, "pose", None)
        if pose is None:
            # No pose info — give benefit of the doubt
            return 0.0, True, None

        yaw   = abs(float(pose[1]))  # left/right rotation
        pitch = abs(float(pose[0]))  # up/down tilt
        max_angle = max(yaw, pitch)

        passed = max_angle <= self.MAX_POSE_ANGLE
        if not passed:
            axis = "yaw" if yaw > pitch else "pitch"
            reason = f"Face not frontal ({axis}: {max_angle:.1f}°) — look directly at camera"
        else:
            reason = None
        return max_angle, passed, reason

    def assess(self, img, face):
        """
        Run all 4 quality checks on a detected InsightFace face object.

        Args:
            img:  Full BGR image (numpy array)
            face: InsightFace face object (has .bbox, .pose, .embedding)

        Returns:
            passed (bool):     True only if ALL checks pass
            issues (list[str]):Human-readable rejection reasons (empty if passed)
            scores (dict):     Raw numeric scores for each check
        """
        bbox = face.bbox  # [x1, y1, x2, y2]

        blur_score,       blur_ok,   blur_reason   = self.check_blur(img, bbox)
        brightness_score, bright_ok, bright_reason = self.check_brightness(img, bbox)
        size_ratio,       size_ok,   size_reason   = self.check_face_size(img, bbox)
        angle,            angle_ok,  angle_reason  = self.check_face_angle(face)

        issues = [r for r in [blur_reason, bright_reason, size_reason, angle_reason] if r]
        passed = blur_ok and bright_ok and size_ok and angle_ok

        scores = {
            "blur":       round(blur_score, 1),
            "brightness": round(brightness_score, 1),
            "face_size":  round(size_ratio * 100, 1),   # as %
            "angle":      round(angle, 1),
            "blur_ok":       blur_ok,
            "brightness_ok": bright_ok,
            "size_ok":       size_ok,
            "angle_ok":      angle_ok,
        }

        return passed, issues, scores
