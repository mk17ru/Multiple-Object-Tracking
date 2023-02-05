import cv2

from src.feature_extractor import FeatureExtractor

from src.utils import calc_feature_cosine

img1 = cv2.imread("demo/Images/savva1.jpg")
img2 = cv2.imread("demo/Images/savva2.jpg")
img3 = cv2.imread("demo/Images/Pedestrians.jpeg")
f = FeatureExtractor()

f1 = f.extract_image_features(img1)
f2 = f.extract_image_features(img2)
f3 = f.extract_image_features(img3)

print(calc_feature_cosine(f1, f2))
print(calc_feature_cosine(f1, f3))
