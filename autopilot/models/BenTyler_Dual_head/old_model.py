'work in progress'

import numpy as np
import tensorflow as tf
import os

class Model:

    
    def __init__(self):
        self.saved_joint_model = 'CNN.tflite'
        # self.joint_model = tf.keras.models.load_model(os.path.join(os.path.dirname(os.path.abspath(__file__)), self.saved_joint_model))
        self.interpreter = tf.lite.Interpreter(
            model_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), self.saved_joint_model)
        )
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def preprocess(self, image):
        im = tf.image.convert_image_dtype(image, tf.float32)
        im /= 255
        im = tf.image.resize(im, [224, 224]) #should be whatever the model input size is
        im = tf.expand_dims(im, axis=0)
        return im

    def predict(self, image):
        image = self.preprocess(image)
        if isinstance(image, tf.tensor):
            image = image.numpy()

        self.interpreter.set_tensor(self.input_details[0]['index'],image)
        self.interpreter.invoke()

        pred_angle = self.interpreter.get_tensor(self.output_details[0]['index'])
        pred_speed = self.interpreter.get_tensor(self.output_details[1]['index'])
        angles = np.arange(17)*5+50
        speed = int(pred_speed[0][0])*35
        angle = angles[np.argmax(pred_angle[0])]
        print('angle:', angle,'speed:', speed)
        
        return angle, speed
