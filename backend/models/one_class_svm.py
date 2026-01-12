import numpy as np
from typing import List
from sklearn.svm import OneClassSVM

class OneClassSVMDetector:
    """
    One-Class SVM based anomaly detection.
    Uses sklearn's OneClassSVM.
    """
    def __init__(self, nu: float = 0.1, kernel: str = 'rbf', gamma: str = 'scale'):
        self.nu = nu
        self.kernel = kernel
        self.gamma = gamma
        self.model = None

    def fit(self, data: np.ndarray):
        """Fit the detector on the data."""
        # Reshape if 1D
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        self.model = OneClassSVM(nu=self.nu, kernel=self.kernel, gamma=self.gamma)
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