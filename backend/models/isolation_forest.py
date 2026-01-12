import numpy as np
from typing import List
from sklearn.ensemble import IsolationForest

class IsolationForestDetector:
    """
    Isolation Forest based anomaly detection.
    Uses sklearn's IsolationForest.
    """
    def __init__(self, contamination: float = 0.1, random_state: int = 42):
        self.contamination = contamination
        self.random_state = random_state
        self.model = None

    def fit(self, data: np.ndarray):
        """Fit the detector on the data."""
        # Reshape if 1D
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        self.model = IsolationForest(contamination=self.contamination, random_state=self.random_state)
        self.model.fit(data)

    def predict(self, data: np.ndarray) -> List[bool]:
        """Predict anomalies in the data."""
        if self.model is None:
            raise ValueError("Detector must be fitted before prediction.")
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        predictions = self.model.predict(data)
        # -1 is anomaly, 1 is normal
        return (predictions == -1).tolist()

    def fit_predict(self, data: np.ndarray) -> List[bool]:
        """Fit and predict in one step."""
        self.fit(data)
        return self.predict(data)