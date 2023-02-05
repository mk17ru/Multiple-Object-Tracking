from typing import List, Dict

import numpy as np
import math
import yaml

from src.object_info import ObjectInfo
from src.utils import calc_deepsort_metric, calc_mahalanobis, calc_feature_cosine, calc_cost_area

from scipy.optimize import linear_sum_assignment

INF = 1000.

class Matcher:

    def __init__(self):
        self.counter = 0

        with open("configs/constants.yaml", "r") as f:
            self.consts = yaml.safe_load(f)["matcher"]

    """
        Return (new_ids, new_objects). For new objects updates velocity.
        In case prev_objects is None, enumerate new_objects
    """

    # returns None if rectangles don't intersect

    def match(self, actual_objects: Dict[int, ObjectInfo], new_objects: Dict[int, ObjectInfo], add_new: bool) -> Dict[int, ObjectInfo]:
        if len(actual_objects) == 0:
            return self.make_result([], [], new_objects, [], list(new_objects.keys()), True)

        self.counter = max(self.counter, max(actual_objects) + 1)

        costs = []
        for obj in actual_objects.values():
            row = []
            for new_obj in new_objects.values():
                c1 = calc_cost_area(obj, new_obj)
                c2 = calc_feature_cosine(obj.features, new_obj.features)
                if c1 <= self.consts["sort_threshold"] and c2 <= self.consts["features_threshold"]:
                    row.append(calc_deepsort_metric(obj, new_obj, self.consts["lmb"]))
                else:
                    row.append(INF)
            costs.append(row)
        row_ind, col_ind = linear_sum_assignment(np.array(costs))

        row_new_ind, col_new_ind = [], []

        for i in range(len(row_ind)):
            row = row_ind[i]
            col = col_ind[i]
            if costs[row][col] != INF:
                row_new_ind.append(row)
                col_new_ind.append(col)

        return self.make_result(row_new_ind, col_new_ind, new_objects, list(actual_objects.keys()), list(new_objects.keys()), add_new)

    def make_result(self, row_ind: List[int], col_ind: List[int], new_objects: Dict[int, ObjectInfo], prev_inds: List[int], new_inds: List[int], add_new: bool):
        answer = {}
        for i in range(len(col_ind)):
            answer[prev_inds[row_ind[i]]] = (new_inds[col_ind[i]], new_objects[new_inds[col_ind[i]]])
        index_mn = set(col_ind)

        if add_new:
            for i in new_objects.keys():
                if i not in index_mn:
                    answer[self.counter] = (i, new_objects[i])
                    self.counter += 1
        return answer
