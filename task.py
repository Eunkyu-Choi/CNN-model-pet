import tensorflow as tf
import numpy as np
from PIL import Image
from glob import glob
import cv2

if __name__ == '__main__':
    path = glob(input('path of image: ') + '\\*.jpg')
    model = tf.keras.models.load_model('saved_model_inout/my_model')
    class_list = ['indoor','outdoor']
    for i in path:
        image = Image.open(i)

        name = i.split('\\')[-1]
        cls = i.split('\\')[-2]
        image = image.resize((224, 224))
        image_o = image
        image = np.array(image)
        image = np.reshape(image, (1, 224, 224, 3))
        prediction = model.predict(image)
        pred_class = np.argmax(prediction, axis = 1)
        pred = class_list[int(pred_class)]
        # labeling to image
        image_o.save('C:\\Users\\localley\\Desktop\\result\\{}\\{}\\{}'.format(cls,pred,name))
        # print(pred)
