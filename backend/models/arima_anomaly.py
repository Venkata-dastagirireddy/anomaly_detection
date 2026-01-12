import numpy as np
from typing import List
from statsmodels.tsa.arima.model import ARIMA

class ARIMAAomalyDetector:
    """
    ARIMA-based anomaly detection.
    Fits ARIMA model, detects anomalies in residuals using z-score.
    """
    def __init__(self, order: tuple = (1, 1, 1), threshold: float = 3.0):
        self.order = order
        self.threshold = threshold
        self.model = None

    def fit(self, data: np.ndarray):
        """Fit the ARIMA model."""
        self.model = ARIMA(data, order=self.order)
        self.model_fit = self.model.fit()

    def predict(self, data: np.ndarray) -> List[bool]:
        """Predict anomalies based on residuals."""
        if self.model is None:
            raise ValueError("Model must be fitted before prediction.")
        residuals = self.model_fit.resid
        mean_resid = np.mean(residuals)
        std_resid = np.std(residuals)
        z_scores = (residuals - mean_resid) / std_resid
        return (np.abs(z_scores) > self.threshold).tolist()

    def fit_predict(self, data: np.ndarray) -> List[bool]:
        """Fit and predict in one step."""
        self.fit(data)
        return self.predict(data)