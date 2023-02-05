import copy
import yaml
import numpy as np

from typing import List, Dict

from filterpy.kalman import KalmanFilter

from src.object_info import ObjectInfo


def create_filter(start_pos: np.array):
    kf = KalmanFilter(dim_x=8, dim_z=4)

    kf.F = np.array([[1., 0., 0., 0., 1., 0., 0., 0.],
                     [0., 1., 0., 0., 0., 1., 0., 0.],
                     [0., 0., 1., 0., 0., 0., 1., 0.],
                     [0., 0., 0., 1., 0., 0., 0., 1.],
                     [0., 0., 0., 0., 1., 0., 0., 0.],
                     [0., 0., 0., 0., 0., 1., 0., 0.],
                     [0., 0., 0., 0., 0., 0., 1., 0.],
                     [0., 0., 0., 0., 0., 0., 0., 1.]], np.float32)
    kf.H = np.array([[1., 0., 0., 0., 0., 0., 0., 0.],
                     [0., 1., 0., 0., 0., 0., 0., 0.],
                     [0., 0., 1., 0., 0., 0., 0., 0.],
                     [0., 0., 0., 1., 0., 0., 0., 0.]], np.float32)

    kf.R[2:, 2:] *= 10.
    kf.P[4:, 4:] *= 1000.
    kf.P *= 10.
    kf.Q[6:, 6:] *= 0.01
    kf.Q[4:, 4:] *= 0.01

    kf.x[:4] = start_pos.reshape((4, 1))

    kf.predict()

    return kf


class Predictor:
    def __init__(self):
        with open("configs/constants.yaml", "r") as f:
            self.consts = yaml.safe_load(f)["predictor"]

        pass

    def predict(self, objects: Dict[int, ObjectInfo]) -> None:
        for obj in objects.values():
            if obj.kalman_filter.x[2] + obj.kalman_filter.x[6] <= 0:
                obj.kalman_filter.x[6] = 0.
            if obj.kalman_filter.x[3] + obj.kalman_filter.x[7] <= 0:
                obj.kalman_filter.x[7] = 0.
            obj.kalman_filter.predict()

    def correct(self, actual_objects: Dict[int, ObjectInfo], new_objects: Dict[int, ObjectInfo], step: int) -> None:
        for id, obj in new_objects.items():
            if id in actual_objects:
                actual_objects[id].kalman_filter.update(obj[1].kalman_filter.x[:4])
                actual_objects[id].features = np.vstack([actual_objects[id].features, obj[1].features])
                if actual_objects[id].features.shape[0] == self.consts["features_lifetime"]:
                    actual_objects[id].features = np.delete(actual_objects[id].features, (0), axis=0)
            else:
                actual_objects[id] = obj[1]
                actual_objects[id].first_detected = step
            actual_objects[id].last_detected = step
