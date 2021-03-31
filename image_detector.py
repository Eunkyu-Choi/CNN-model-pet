import numpy as np
import cv2
from PIL import Image
from glob import glob


class ImageDetector:
    def __init__(self):

        self.min_confidence = 0.30
        self.width = 448

        self.net = cv2.dnn.readNet('yolov4-tiny_last.weights','yolov4-tiny.cfg')
        self.classes = ['dog','person','cat', 'face']
        self.layer_names = self.net.getLayerNames()
        self.output_layers = [self.layer_names[i[0] -1] for i in self.net.getUnconnectedOutLayers()]

    def detect(self, image, name):

        h, w = image.shape[:2]
        height = int(h * self.width / w)
        img = cv2.resize(image, (self.width, height))

        blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), swapRB = True, crop = False)
        self.net.setInput(blob)
        outs = self.net.forward(self.output_layers)

        confidences = []
        names = []
        boxes = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > self.min_confidence:
                    center_x = int(detection[0] * self.width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * self.width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    names.append(self.classes[class_id])

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, self.min_confidence, 0.4)
        label = []

        for i in range(len(boxes)):
            if i in indexes:
                label.append(names[i])

        if 'dog' in label:
            folder = 'dog'
            if 'person' in label:
                folder += '_person'
        elif 'cat' in label:
            folder = 'cat'
            if 'person' in label:
                folder += '_person'
        else:
            folder = 'etc'
        # '''Path to Save Image '''

        cv2.imwrite('/home/localley/Desktop/result/{}/{}'.format(folder, name), img)
