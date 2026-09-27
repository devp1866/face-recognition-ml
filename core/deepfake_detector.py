import cv2
import numpy as np


class DeepfakeDetector:
    """
    Deepfake / AI-Generated Face Detector using Frequency Domain Analysis.

    Real human faces have a characteristic frequency signature: natural
    skin texture produces a specific distribution of energy across the
    FFT spectrum. GAN-generated faces (StyleGAN, DALL-E, Midjourney, etc.)
    contain periodic high-frequency artifacts introduced by:
        - Upsampling/transposed convolutions in GAN decoders
        - Checkerboard artifacts from stride-2 convolutions
        - Spectral peaks at Nyquist frequencies due to nearest-neighbor upsampling

    Algorithm:
        1. Crop and normalize the face region to 128×128 grayscale
        2. Apply 2D FFT and shift zero-frequency to center
        3. Compute log-magnitude spectrum
        4. Extract 3 discriminative features:
            a. High-frequency energy ratio (HF band power / total power)
            b. Spectral entropy (GAN spectra are less uniform → lower entropy)
            c. Azimuthal symmetry score (real faces are more symmetric in freq domain)
        5. Threshold-based decision combining all 3 features

    References:
        - "Detecting and Simulating Artifacts in GAN Fake Images" (Durall et al., 2020)
        - "Watch your Up-Convolution: CNN Based Generative Deep Neural Networks are
          Failing to Reproduce Spectral Distributions" (Durall et al., 2020)
    """

    # Face crop size for FFT analysis
    CROP_SIZE = 128

    # Frequency thresholds (as fraction of max spatial frequency)
    HF_BAND_START = 0.35   # Frequencies above 35% of max = "high frequency"

    # Decision thresholds — set conservatively to avoid false positives.
    # Only extremely structured GAN spectra should trigger these.
    HF_RATIO_THRESHOLD    = 0.93   # Must be very high HF energy (extreme GAN artifact)
    ENTROPY_THRESHOLD     = 1.80   # Must be very low entropy (very structured spectrum)
    SYMMETRY_THRESHOLD    = 0.15   # Must be very asymmetric

    # ALL 3 checks must agree to flag as deepfake.
    # This makes false positives on real photos virtually impossible.
    CONSENSUS_REQUIRED    = 3      # Out of 3 checks

    def _extract_face_crop(self, img: np.ndarray, bbox) -> np.ndarray:
        """
        Crop the face region from the image and resize to CROP_SIZE × CROP_SIZE.
        Converts to grayscale for FFT analysis (color channels add noise).
        """
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        h, w = img.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        face_crop = img[y1:y2, x1:x2]
        if face_crop.size == 0:
            return None

        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (self.CROP_SIZE, self.CROP_SIZE))
        return resized.astype(np.float32)

    def _compute_fft_spectrum(self, gray_crop: np.ndarray) -> np.ndarray:
        """
        Compute the 2D log-magnitude FFT spectrum, shifted so DC is at center.
        Returns a normalized 2D array of the same size as input.
        """
        # Apply Hann window to reduce spectral leakage at edges
        window = np.outer(np.hanning(self.CROP_SIZE), np.hanning(self.CROP_SIZE))
        windowed = gray_crop * window

        # 2D FFT + shift DC to center
        fft = np.fft.fft2(windowed)
        fft_shifted = np.fft.fftshift(fft)

        # Log-magnitude spectrum (avoids numerical issues with very small values)
        magnitude = np.abs(fft_shifted)
        log_magnitude = np.log1p(magnitude)

        return log_magnitude

    def _compute_hf_ratio(self, spectrum: np.ndarray) -> float:
        """
        High-frequency energy ratio: power in the outer ring vs total power.

        GAN upsampling creates spectral peaks at high frequencies (Nyquist).
        Real faces have smoothly decaying high-frequency content.
        A higher ratio than expected → GAN artifact pattern.
        """
        h, w = spectrum.shape
        cy, cx = h // 2, w // 2

        # Create a radial distance map from center
        y, x = np.ogrid[:h, :w]
        radius_map = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        max_radius = np.sqrt(cx ** 2 + cy ** 2)

        # High-frequency band: beyond HF_BAND_START fraction of max radius
        hf_mask = radius_map > (self.HF_BAND_START * max_radius)

        total_energy = np.sum(spectrum ** 2)
        hf_energy    = np.sum((spectrum * hf_mask) ** 2)

        return float(hf_energy / total_energy) if total_energy > 0 else 0.0

    def _compute_spectral_entropy(self, spectrum: np.ndarray) -> float:
        """
        Shannon entropy of the normalized power spectral density.

        A natural face spectrum has higher entropy (more uniform spread of energy).
        GAN artifacts concentrate energy at specific frequencies → lower entropy.
        """
        power = spectrum.flatten() ** 2
        total = power.sum()
        if total == 0:
            return 0.0

        # Normalize to a probability distribution
        prob = power / total
        # Remove zeros to avoid log(0)
        prob = prob[prob > 0]
        entropy = -float(np.sum(prob * np.log(prob)))
        return entropy

    def _compute_symmetry_score(self, spectrum: np.ndarray) -> float:
        """
        Measure azimuthal symmetry of the FFT spectrum.

        Real face spectra are approximately radially symmetric.
        GAN artifacts often introduce directional/grid-like patterns that
        break this symmetry (e.g., from stride-2 transposed convolutions).

        Score: Pearson correlation between the spectrum and its 90°-rotated version.
        Range: [-1, 1]. Real faces → closer to 1.0.
        """
        rotated = np.rot90(spectrum)
        flat1 = spectrum.flatten()
        flat2 = rotated.flatten()

        # Pearson correlation
        mean1, mean2 = flat1.mean(), flat2.mean()
        std1, std2   = flat1.std(), flat2.std()

        if std1 < 1e-8 or std2 < 1e-8:
            return 1.0  # Uniform spectrum — cannot determine, default to real

        corr = float(np.mean((flat1 - mean1) * (flat2 - mean2)) / (std1 * std2))
        return max(-1.0, min(1.0, corr))

    def analyze(self, img: np.ndarray, bbox) -> dict:
        """
        Full deepfake analysis pipeline for a single detected face.

        Args:
            img:   Full BGR image (numpy array)
            bbox:  Face bounding box [x1, y1, x2, y2]

        Returns:
            dict with:
                is_authentic (bool):   True = real face, False = suspected AI-generated
                confidence (float):    0.0–1.0 probability of being authentic
                label (str):           "Authentic" | "Suspected AI-Generated" | "Uncertain"
                hf_ratio (float):      High-frequency energy ratio
                entropy (float):       Spectral entropy value
                symmetry (float):      Azimuthal symmetry score
                flags_triggered (int): Number of GAN-artifact checks triggered (0–3)
        """
        face_crop = self._extract_face_crop(img, bbox)

        if face_crop is None:
            return {
                "is_authentic": True,
                "confidence": 0.5,
                "label": "Uncertain (crop failed)",
                "hf_ratio": 0.0,
                "entropy": 0.0,
                "symmetry": 1.0,
                "flags_triggered": 0,
            }

        spectrum = self._compute_fft_spectrum(face_crop)

        hf_ratio  = self._compute_hf_ratio(spectrum)
        entropy   = self._compute_spectral_entropy(spectrum)
        symmetry  = self._compute_symmetry_score(spectrum)

        # Each check flags a potential GAN artifact
        hf_flag       = hf_ratio  > self.HF_RATIO_THRESHOLD
        entropy_flag  = entropy   < self.ENTROPY_THRESHOLD
        symmetry_flag = symmetry  < self.SYMMETRY_THRESHOLD

        flags = int(hf_flag) + int(entropy_flag) + int(symmetry_flag)

        # Consensus decision: need at least CONSENSUS_REQUIRED flags
        if flags >= self.CONSENSUS_REQUIRED:
            is_authentic = False
            label = "Suspected AI-Generated"
            # Confidence: scale with number of flags
            confidence = round(1.0 - (flags / 3.0 * 0.7), 2)
        elif flags == 1:
            is_authentic = True
            label = "Uncertain"
            confidence = 0.65
        else:
            is_authentic = True
            label = "Authentic"
            confidence = round(0.80 + (symmetry * 0.10), 2)

        return {
            "is_authentic":     is_authentic,
            "confidence":       confidence,
            "label":            label,
            "hf_ratio":         round(hf_ratio, 4),
            "entropy":          round(entropy, 4),
            "symmetry":         round(symmetry, 4),
            "flags_triggered":  flags,
        }
