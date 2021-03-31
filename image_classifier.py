import tensorflow as tf
import numpy as np
from PIL import Image
import cv2

class ImageClassifier:
    def __init__(self, model_path, f):
        self.model_path = model_path
        self.model = tf.keras.models.load_model(self.model_path)
        self.f = f

    def classify(self, path):
        class_list = self.f.readlines()
        #학습된 모델 불러오기


        image = Image.open(path)

        # CNN Model Input Size
        image = image.resize((224,224))
        image = np.array(image)
        # Convert Image to Gray, and RGB Again
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        #  0 < Pixel Value < 1
        # image = image/255.

        # Convert to tensor shape
        image = np.reshape(image, (1, 224, 224, 3))
        # Prediction
        prediction = self.model.predict(image)
        pred_class = np.argmax(prediction, axis = 1)
        pred_breed = class_list[int(pred_class)]

        # 예측된 pred_breed 값을 해당 반려동물의 종으로 입력.
        print(pred_breed)
