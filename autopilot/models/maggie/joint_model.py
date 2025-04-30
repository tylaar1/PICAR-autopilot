'work in progress'

import numpy as np
import tensorflow as tf
import os

class Model:

    saved_joint_model = 'joint_model_filepath'
  
    def __init__(self):
        self.joint_model = tf.keras.models.load_model(os.path.join(os.path.dirname(os.path.abspath(__file__)), self.saved_joint_model))
    def preprocess(self, image):
        im = tf.image.convert_image_dtype(image, tf.float32)
        im = tf.image.resize(im, [100, 100]) #should be whatever the model input size is
        im = tf.expand_dims(im, axis=0)
        return im

    def predict(self, image):
        angles = np.arange(17)*5+50
        image = self.preprocess(image)
        pred_angle, pred_speed = self.joint_model.predict(image)
        speed = int(pred_speed[0][0])*35
        angle = angles[np.argmax(pred_angle[0])]
        print('angle:', angle,'speed:', speed)
        
        return angle, speed
