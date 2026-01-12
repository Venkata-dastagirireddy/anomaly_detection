import numpy as np
from typing import List
from statsmodels.tsa.seasonal import seasonal_decompose

class SeasonalDecompositionAnomalyDetector:
    """
    Seasonal Decomposition based anomaly detection.
    Decomposes time series, detects anomalies in residuals using IQR.
    """
    def __init__(self, model: str = 'additive', period: int = None, multiplier: float = 1.5):
        self.model = model
        self.period = period  # If None, will be inferred
        self.multiplier = multiplier
        self.residuals = None

    def fit(self, data: np.ndarray):
        """Decompose the time series."""
        if self.period is None:
            # Simple heuristic: assume daily if len > 365, etc. But for now, set to 7 or something.
            self.period = min(7, len(data) // 2) if len(data) > 14 else 2
        decomposition = seasonal_decompose(data, model=self.model, period=self.period)
        self.residuals = decomposition.resid

    def predict(self, data: np.ndarray) -> List[bool]:
        """Predict anomalies based on residuals."""
        if self.residuals is None:
            raise ValueError("Detector must be fitted before prediction.")
        # Remove NaN from residuals (edges)
        clean_resid = self.residuals[~np.isnan(self.residuals)]
        q1 = np.percentile(clean_resid, 25)
        q3 = np.percentile(clean_resid, 75)
        iqr = q3 - q1
        lower_bound = q1 - self.multiplier * iqr
        upper_bound = q3 + self.multiplier * iqr
        anomalies = []
        for i in range(len(data)):
            if np.isnan(self.residuals[i]):
                anomalies.append(False)  # Edges are not anomalies
            else:
                anomalies.append(not (lower_bound <= self.residuals[i] <= upper_bound))
        return anomalies

    def fit_predict(self, data: np.ndarray) -> List[bool]:
        """Fit and predict in one step."""
        self.fit(data)
        return self.predict(data)