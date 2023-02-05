import math
import numpy as np

from src.object_info import ObjectInfo


def clean_file(file_name: str):
    f = open(file_name, "w")
    f.close()


def print_stats(file, i, objects):
    with open(file, "a") as out:
        for obj_id, obj in objects.items():
            rect = box_to_rectangle(obj.kalman_filter.x[:4]).flatten()
            out.write(
                f'{i + 1},{obj_id}.0,{rect[0]},{rect[1]},{rect[2] - rect[0]},{rect[3] - rect[1]},1,-1,-1,-1\n')


def box_to_rectangle(box: np.array) -> np.array:
    lx = math.sqrt(box[2] / box[3])
    ly = math.sqrt(box[2] * box[3])
    return np.array([box[0] - 0.5 * lx, box[1] - 0.5 * ly, box[0] + 0.5 * lx, box[1] + 0.5 * ly])


def area_intersect(a: np.array, b: np.array) -> float:
    dx = min(a[2], b[2]) - max(a[0], b[0])
    dy = min(a[3], b[3]) - max(a[1], b[1])
    if (dx >= 0) and (dy >= 0):
        return dx * dy
    else:
        return None


def calc_cost_area(a: ObjectInfo, b: ObjectInfo) -> float:
    intersection = area_intersect(box_to_rectangle(a.kalman_filter.x[:4]), box_to_rectangle(b.kalman_filter.x[:4]))
    if intersection is None:
        return 1.
    return 1. - ((intersection / (a.kalman_filter.x[2] + b.kalman_filter.x[2] - intersection))[0])


def calc_mahalanobis(a: ObjectInfo, b: np.array) -> float:
    d = a.kalman_filter.x[:4] - b
    return math.sqrt(d.T @ a.kalman_filter.SI @ d)


def calc_feature_cosine(a: np.ndarray, b: np.ndarray) -> float:
    res = np.amax(a @ b.T)
    return 1. - res


def calc_deepsort_metric(a: ObjectInfo, b: ObjectInfo, lmb: float) -> float:
    c1 = calc_cost_area(a, b)
    c2 = calc_feature_cosine(a.features, b.features)
    return lmb * c1 + (1. - lmb) * c2
