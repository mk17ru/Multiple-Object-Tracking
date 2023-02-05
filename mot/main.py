import argparse
import yaml
from src.pipeline import Pipeline

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    with open("configs/paths_to_detectors.yaml", "r") as stream:
        detectors_list = list(yaml.safe_load(stream)["paths"].keys())
    parser.add_argument("--detector", choices=detectors_list, type=str, metavar="detector", required=True,
                        help=f'Select multi-object detector from the list below {detectors_list}')
    parser.add_argument("--video", type=str, metavar="video", required=True, help='Select video you want to process')
    args = parser.parse_args()

    pipeline = Pipeline(args.detector, args.video, "MOT20-01.txt")
    pipeline.run()
