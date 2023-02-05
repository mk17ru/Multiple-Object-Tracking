import yaml
import copy

from src.object_info import ObjectInfo
from typing import List, Dict


class Tracker:
    def __init__(self):
        self.tracks = {}
        self.frames = []
        with open("configs/constants.yaml", "r") as f:
            self.consts = yaml.safe_load(f)["tracker"]

    def get_boxes_in_frame(self, frame_num: int):
        return self.frames[frame_num]

    def get_lask_k_frames(self, start: int, step: int, k: int, prev_ids, prev_objects):
        result = {prev_ids[it]: prev_objects[it] for it in range(len(prev_ids))}
        for it in range(start, start - step * k, -step):
            if it < 0:
                break
            for obj_id in self.frames[it]:
                if obj_id in result:
                    continue
                result[obj_id] = self.frames[it][obj_id]
        return list(result.keys()), list(result.values())

    def register(self, frame_num: int, objects: Dict[int, ObjectInfo]) -> None:
        cur_frame = {}
        for object_id, box_info in objects.items():
            cur_frame[object_id] = copy.deepcopy(box_info[1])
            if object_id not in self.tracks:
                self.tracks[object_id] = [(frame_num, ObjectInfo(copy.deepcopy(box_info[1].kalman_filter), box_info[1].type, box_info[1].prob, None, box_info[1].last_detected))]
            else:
                self.tracks[object_id].append((frame_num, ObjectInfo(copy.deepcopy(box_info[1].kalman_filter), box_info[1].type, box_info[1].prob, None, box_info[1].last_detected)))
        self.frames.append(cur_frame)
