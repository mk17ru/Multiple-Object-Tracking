from src.pipeline import Pipeline


detector_name = 'yolox-x'
reid_name = 'osnet_x1_0'
video_path = 'demo/MOT20/train/MOT20-01/img1'

pipeline = Pipeline(detector_name, reid_name, video_path, "MOT20-01.txt")
pipeline.run()