import numpy as np
from feature_extractor import FeatureExtractor
from PIL import Image
from glob import glob
import cv2
import os

img_paths = []
features = []
fe = FeatureExtractor()
'''Path of Image to Search'''
# All Image after image_detector
path = glob(input('path of image: ')+'\\*.jpg')
for img_path in path:
    feature = fe.extract(img = Image.open(img_path))
    features.append(feature)
    img_paths.append(img_path)
features = np.array(features)

# User에게 Image 전달받음.
'''file of Original Image'''
# Representative image of each breed
o_path = input('image: ')
img = Image.open(o_path)
query = fe.extract(img)
# L2 distance
dists = np.linalg.norm(features - query, axis = 1)
ids = np.argsort(dists)
scores = [(dists[id], img_paths[id]) for id in ids]
# Score가 낮은 사진 n장 추출하여 저장
for d, i in scores:
    if d < 1.39:
        im = Image.open(i)
        im.save(o_path.replace('.jpg','\\') + str(d) +  i.split('\\')[-1])
    else:
        break
# print(scores)
