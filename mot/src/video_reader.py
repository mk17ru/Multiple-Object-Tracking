import numpy as np
import os
import cv2


class VideoReader:
    """@param path -- path to folder with frames for tracking"""
    def __init__(self, path):
        self.path = path
        if not os.path.exists(self.path):
            raise Exception("Invalid path to frames directory")

    """yields next frame path and frame itself"""
    def read_frame(self) -> np.ndarray:
        if os.path.isdir(self.path):
            for file_name in sorted(os.listdir(self.path)):
                full_name = self.path + "/" + file_name
                yield cv2.imread(full_name)
        else:
            cap = cv2.VideoCapture(self.path)
            if not cap.isOpened():
                raise Exception("Cannot open video")
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                yield frame
            cap.release()
