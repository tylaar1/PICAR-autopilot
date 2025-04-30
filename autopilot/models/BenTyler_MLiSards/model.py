import numpy as np
import tensorflow as tf
import os

class Model:

    saved_speed_model = 'mobnetv3small_classification_model.h5'
    saved_angle_model = 'mobnetv3small_regression_model.h5'
    def __init__(self):
        self.speed_model = tf.keras.models.load_model(os.path.join(os.path.dirname(os.path.abspath(__file__)), self.saved_speed_model))
        self.angle_model = tf.keras.models.load_model(os.path.join(os.path.dirname(os.path.abspath(__file__)), self.saved_angle_model))

    def preprocess(self, image):
        im = tf.image.convert_image_dtype(image, tf.float32)
        im = tf.image.resize(im, [224, 224]) #should be whatever the model input size is
        im = tf.expand_dims(im, axis=0)
        return im

    def predict(self, image):
        angles = np.arange(17)*5+50
        image = self.preprocess(image)
        
        pred_speed = self.speed_model.predict(image)[0]
        speed = pred_speed[0].astype(int)*35
        pred_angle = self.angle_model.predict(image)[0]
        angle = angles[np.argmax(pred_angle)]
        print('angle:', angle,'speed:', speed)
        
        return angle, speed
