from typing import List, Dict

import cv2
import numpy as np
import torch
from torchvision import models, transforms

from src.object_info import ObjectInfo
from src.utils import box_to_rectangle, calc_feature_cosine
from src.predictor import create_filter

from torchreid.reid.utils import FeatureExtractor

class FeatureExtractorManager:
    def __init__(self, reid_name):
        self.feature_extractor = FeatureExtractor(
            model_name=reid_name,
            model_path='demo/' + reid_name + '.pth',
            device='cpu'
        )

    def extract_image_features(self, imgs: np.ndarray) -> np.ndarray:
        features = self.feature_extractor(imgs)
        return features

    def extract_features(self, img: np.ndarray, objects: Dict[int, ObjectInfo]):
        boxes = []
        for obj in objects.values():
            bbox = box_to_rectangle(obj.kalman_filter.x[:4]).flatten().astype(np.int64)
            bbox = np.array([max(0, bbox[0]), max(0, bbox[1]), min(img.shape[1], bbox[2]), min(img.shape[0], bbox[3])])
            boxes.append(img[bbox[1]:bbox[3], bbox[0]:bbox[2]])
        features = self.extract_image_features(boxes)

        it = 0
        for k, v in objects.items():
            current_features = features[it].numpy().flatten()
            v.features = current_features / np.linalg.norm(current_features)
            it += 1
