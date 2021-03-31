from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras import models
import numpy as np

class FeatureExtractor:
    def __init__(self):
        base_model = MobileNetV2(weights='imagenet')
        self.model = Model(inputs=base_model.input, outputs=base_model.get_layer('predictions').output)

    def extract(self, img):

        img = img.resize((224, 224))
        img = img.convert('RGB')
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        feature = self.model.predict(x)[0]
        return feature / np.linalg.norm(feature)
