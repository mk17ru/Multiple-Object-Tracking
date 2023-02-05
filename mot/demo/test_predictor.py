import numpy as np

from src.object_info import ObjectInfo
from src.predictor import Predictor, create_filter

from filterpy.kalman import KalmanFilter

predictor = Predictor()

box1 = np.array([0.5, 0.5, 1., 1., 0., 0., 0.])
obj1 = ObjectInfo(create_filter(box1[:4]), "person", 0.8)
predictor.correct([], [], [0], [obj1])

print(obj1.kalman_filter.x)

box2 = np.array([0.52, 0.52, 1., 1., 0., 0., 0.])
obj2 = ObjectInfo(create_filter(box2[:4]), "person", 0.7)
predictor.correct([0], [obj1], [0], [obj2])

print(obj2.kalman_filter.x)
