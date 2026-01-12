import numpy as np
from typing import List

class ZScoreDetector:
    """
    Z-Score based anomaly detection.
    Anomalies are points where |z-score| > threshold.
    """
    def __init__(self, threshold: float = 3.0):
        self.threshold = threshold
        self.mean = None
        self.std = None

    def fit(self, data: np.ndarray):
        """Fit the detector on the data."""
        self.mean = np.mean(data)
        self.std = np.std(data)

    def predict(self, data: np.ndarray) -> List[bool]:
        """Predict anomalies in the data."""
        if self.mean is None or self.std is None:
            raise ValueError("Detector must be fitted before prediction.")
        z_scores = (data - self.mean) / self.std
        return (np.abs(z_scores) > self.threshold).tolist()

    def fit_predict(self, data: np.ndarray) -> List[bool]:
        """Fit and predict in one step."""
        self.fit(data)
        return self.predict(data)