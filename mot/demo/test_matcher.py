import numpy as np
from filterpy.kalman import KalmanFilter

from src.matcher import Matcher
from src.object_info import ObjectInfo
from src.predictor import create_filter

box1 = np.array([1., 2., 3., 4., 0., 0., 0.])
box2 = np.array([3., 5., 6., 7., 0., 0., 0.])

obj1 = ObjectInfo(create_filter(box1[:4]), "person", 0.7)
obj2 = ObjectInfo(create_filter(box2[:4]), "person", 0.7)

print(obj1.kalman_filter.x)
print(obj2.kalman_filter.x)

matcher = Matcher()
print(matcher.match([0], [obj1], [obj2]))
