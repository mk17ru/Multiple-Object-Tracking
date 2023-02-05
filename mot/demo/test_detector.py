from src.detector import Detector
import torch
import torchvision


detector_name = 'yolox'
img_path = 'demo/images/Pedestrians.jpeg'

detector = Detector(detector_name)
res = detector.detect(img_path)
print([it.kalman_filter.x for it in res])
