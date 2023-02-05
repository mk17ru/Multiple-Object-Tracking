import os
import shutil
import PIL.Image
import cv2
import numpy as np
import yaml

from src.detector import Detector
from src.matcher import Matcher
from src.predictor import Predictor
from src.tracker import Tracker
from src.video_reader import VideoReader
from src.visualizator import Visualizator, create_video
from src.feature_extractor import FeatureExtractorManager

import copy

from src.utils import box_to_rectangle, clean_file, print_stats


class Pipeline:
    def __init__(self, detector_name, reid_name, video_path, stats_file, max_frame=None):
        self.video_reader = VideoReader(video_path)
        self.detector = Detector(detector_name)
        self.matcher = Matcher()
        self.predictor = Predictor()
        self.tracker = Tracker()
        self.visualizator = Visualizator()
        self.feature_extractor = FeatureExtractorManager(reid_name)
        self.max_frame = max_frame
        self.stats_file = stats_file

        with open("configs/constants.yaml", "r") as f:
            self.consts = yaml.safe_load(f)["pipeline"]

        if os.path.exists("test"):
            shutil.rmtree("test")
        os.makedirs("test")

    def draw(self, i, frame, objects):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        p_img = PIL.Image.fromarray(frame)
        self.visualizator.draw_bounding_boxes(p_img, objects).save("test/" + f"{i:04d}.jpeg")

    def create_video(self):
        track_video_file = 'tracking.mp4'
        create_video(frames_patten='test/%04d.jpeg', video_file=track_video_file)

    def get_boxes_of_age(self, actual_objects, age, step):
        return {k: v for k, v in actual_objects.items() if step - v.last_detected == age}

    def run(self):
        actual_objects = {}
        last_detection = -1

        clean_file(self.stats_file)

        for i, frame in enumerate(self.video_reader.read_frame()):
            print(f'Processing frame {i}')
            self.predictor.predict(actual_objects)

            if i % self.consts["detection_frequency"] == 0:
                last_detection = i
                detected_objects = self.detector.detect(frame)
                self.feature_extractor.extract_features(frame, detected_objects)

                for age in range(self.consts["detection_frequency"], self.consts["detection_frequency"] * self.consts["frames_memorized"], self.consts["detection_frequency"]):
                    age_objects = self.get_boxes_of_age(actual_objects, age, i)
                    matched_objects = self.matcher.match(age_objects, detected_objects, False)
                    self.tracker.register(i, matched_objects)
                    self.predictor.correct(actual_objects, matched_objects, i)
                    for v in matched_objects.values():
                        detected_objects.pop(v[0], None)

                matched_objects = self.matcher.match({}, detected_objects, True)
                self.tracker.register(i, matched_objects)
                self.predictor.correct(actual_objects, matched_objects, i)

                actual_objects = {k: v for k, v in actual_objects.items() if
                                  (i - v.last_detected <= self.consts["detection_frequency"] * self.consts["frames_memorized"]) and (v.last_detected - v.first_detected >= self.consts["certainty_window"] * self.consts["detection_frequency"] or v.last_detected == i)}

            objects = {k: v for k, v in actual_objects.items() if v.last_detected == last_detection and v.last_detected - v.first_detected >= self.consts["certainty_window"] * self.consts["detection_frequency"]}
            self.draw(i, frame, objects)
            print_stats(self.stats_file, i, objects)
            if self.max_frame is not None and i == self.max_frame:
                break

        self.create_video()
