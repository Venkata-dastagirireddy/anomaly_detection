import numpy as np
from typing import List

class IQRDetector:
    """
    IQR (Interquartile Range) based anomaly detection.
    Anomalies are points below Q1 - 1.5*IQR or above Q3 + 1.5*IQR.
    """
    def __init__(self, multiplier: float = 1.5):
        self.multiplier = multiplier
        self.q1 = None
        self.q3 = None
        self.iqr = None

    def fit(self, data: np.ndarray):
        """Fit the detector on the data."""
        self.q1 = np.percentile(data, 25)
        self.q3 = np.percentile(data, 75)
        self.iqr = self.q3 - self.q1

    def predict(self, data: np.ndarray) -> List[bool]:
        """Predict anomalies in the data."""
        if self.q1 is None or self.iqr is None:
            raise ValueError("Detector must be fitted before prediction.")
        lower_bound = self.q1 - self.multiplier * self.iqr
        upper_bound = self.q3 + self.multiplier * self.iqr
        return ((data < lower_bound) | (data > upper_bound)).tolist()

    def fit_predict(self, data: np.ndarray) -> List[bool]:
        """Fit and predict in one step."""
        self.fit(data)
        return self.predict(data)