import cv2
from image_detector import ImageDetector
from glob import glob
import time
import os

if __name__ == '__main__':
    # '''Path of Image to Sorting'''
    # All Image of Users Phones
    start = time.time()

    id = ImageDetector()
    path = glob(input('Path of Image to Classification :') + '/*.jpg')
    print(path[0])
    for i in path:
        vs = cv2.VideoCapture(i)
        ret, frame = vs.read()
        # name: Name of Image, to save with same name
        id.detect(image = frame, name = i.split('/')[-1])
    end = time.time()
    print(end - start)
