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

    ## ORIGINAL
    # def predict(self, image):
    #     angles = np.arange(17)*5+50
    #     image = self.preprocess(image)
        
    #     pred_speed = self.speed_model.predict(image)[0]
    #     speed = pred_speed[0].astype(int)*35
    #     pred_angle = self.angle_model.predict(image)[0]
    #     angle = angles[np.argmax(pred_angle)]
    #     print('angle:', angle,'speed:', speed)
        
    #     return angle, speed
    
    # FROM TOM
    def predict(self, image):
        processed_image = self.preprocess(image)
        predictions = [self.speed_model(processed_image, training=False),
                       self.angle_model(processed_image, training=False)]
        speed_prediction = predictions[0].numpy()
        angle_prediction = predictions[1].numpy()

        speed_value = float(speed_prediction[0][0])
        angle_value = float(angle_prediction[0][0])

        angle = 80 * angle_value + 50
        speed = 35 * speed_value
        return angle,speed