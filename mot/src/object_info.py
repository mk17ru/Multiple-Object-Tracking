from dataclasses import dataclass

import numpy as np
from filterpy.kalman import KalmanFilter


@dataclass
class ObjectInfo:
    kalman_filter: KalmanFilter
    type: str
    prob: float
    features: np.ndarray = None
    first_detected: int = -1
    last_detected: int = -1
