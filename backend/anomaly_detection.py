import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
from .models import (
    ZScoreDetector, IQRDetector, IsolationForestDetector, OneClassSVMDetector,
    ARIMAAomalyDetector, SeasonalDecompositionAnomalyDetector
)

def detect_anomalies(df: pd.DataFrame, value_col: str, algorithm: str, **params) -> Tuple[List[bool], str]:
    """
    Detect anomalies in the value column using specified algorithm.
    Returns (anomalies_list, message)
    """
    data = df[value_col].values

    if algorithm == "Z-Score":
        detector = ZScoreDetector(threshold=params.get('threshold', 3.0))
        anomalies = detector.fit_predict(data)
        message = f"Z-Score detection completed with threshold {detector.threshold}"
    elif algorithm == "IQR":
        detector = IQRDetector(multiplier=params.get('multiplier', 1.5))
        anomalies = detector.fit_predict(data)
        message = f"IQR detection completed with multiplier {detector.multiplier}"
    elif algorithm == "Isolation Forest":
        detector = IsolationForestDetector(contamination=params.get('contamination', 0.1))
        anomalies = detector.fit_predict(data)
        message = f"Isolation Forest detection completed with contamination {detector.contamination}"
    elif algorithm == "One-Class SVM":
        detector = OneClassSVMDetector(nu=params.get('nu', 0.1))
        anomalies = detector.fit_predict(data)
        message = f"One-Class SVM detection completed with nu {detector.nu}"
    elif algorithm == "ARIMA":
        order = params.get('order', (1, 1, 1))
        threshold = params.get('arima_threshold', 3.0)
        detector = ARIMAAomalyDetector(order=order, threshold=threshold)
        anomalies = detector.fit_predict(data)
        message = f"ARIMA detection completed with order {order} and threshold {threshold}"
    elif algorithm == "Seasonal Decomposition":
        model_type = params.get('model_type', 'additive')
        period = params.get('period', None)
        multiplier = params.get('sd_multiplier', 1.5)
        detector = SeasonalDecompositionAnomalyDetector(model=model_type, period=period, multiplier=multiplier)
        anomalies = detector.fit_predict(data)
        message = f"Seasonal Decomposition detection completed with model {model_type}"
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    anomaly_count = sum(anomalies)
    message += f". Detected {anomaly_count} anomalies out of {len(anomalies)} points."

    return anomalies, message


def get_anomaly_summary(anomalies: List[bool]) -> Dict[str, Any]:
    """
    Get summary statistics of anomalies.
    """
    total = len(anomalies)
    anomaly_count = sum(anomalies)
    normal_count = total - anomaly_count
    anomaly_percentage = (anomaly_count / total) * 100 if total > 0 else 0

    return {
        "total_points": total,
        "anomaly_count": anomaly_count,
        "normal_count": normal_count,
        "anomaly_percentage": round(anomaly_percentage, 2)
    }