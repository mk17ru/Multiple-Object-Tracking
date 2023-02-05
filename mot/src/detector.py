import yaml
import os
import wget
from typing import List, Dict
import shutil

import numpy as np

import mmcv
from mmcv.runner import load_checkpoint

from mmdet.models import build_detector
from mmdet.apis import inference_detector, show_result_pyplot

from filterpy.kalman import KalmanFilter

from src.object_info import ObjectInfo
from src.predictor import create_filter

class Detector:
    def __init__(self, detector_name):
        with open("configs/paths_to_detectors.yaml", "r") as f:
            detector_config = yaml.safe_load(f)["paths"][detector_name]
            config_path = detector_config["config"]
            checkpoint_path = 'models/' + detector_config["checkpoint_storage"]
            if not os.path.isfile(checkpoint_path):
                shutil.rmtree('models/')
                os.mkdir('models/')
                wget.download(detector_config["checkpoint_url"], out=checkpoint_path)

        with open("configs/constants.yaml", "r") as f:
            self.consts = yaml.safe_load(f)["detector"]

        config = mmcv.Config.fromfile(config_path)
        self.model = build_detector(config.model)

        checkpoint = load_checkpoint(self.model, checkpoint_path)

        self.model.CLASSES = checkpoint['meta']['CLASSES']
        self.model.cfg = config
        self.model.eval()

    def detect(self, img_path: str) -> Dict[int, ObjectInfo]:
        temp_result = inference_detector(self.model, img_path)
        result = {}
        counter = 0
        for class_cnt in range(len(temp_result)):
            if self.model.CLASSES[class_cnt] not in self.consts["classes"]:
                continue
            for box_info in temp_result[class_cnt]:
                if box_info[4] < self.consts["prob_threshold"]:
                    continue

                side1 = abs(box_info[0] - box_info[2])
                side2 = abs(box_info[1] - box_info[3])
                cx = (box_info[0] + box_info[2]) / 2.0
                cy = (box_info[1] + box_info[3]) / 2.0
                s = side1 * side2
                r = side2 / side1
                result[counter] = ObjectInfo(create_filter(np.array([cx, cy, s, r], np.float32)), self.model.CLASSES[class_cnt], box_info[4], None)
                counter += 1
        return result

    # def show_result(self, img: np.ndarray, result: Lis, threshold: float) -> None:
    #     show_result_pyplot(self.model, img, result, score_thr=threshold)
