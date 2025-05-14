import numpy as np
import tensorflow as tf
import os

# ORIGINAL




# #CLAUDE CODE FOR TFLITE
# class Model:
#     def __init__(self):
#         self.saved_speed_model = 'tflite_finetuned_mobnetv3small_classification.tflite'
#         self.saved_angle_model = 'tflite_finetuned_mobnetv3small_regression.tflite'
        
#         # Load TFLite models with just the filenames
#         self.speed_model = tf.lite.Interpreter(model_path=self.saved_speed_model)
#         self.speed_model.allocate_tensors()
        
#         self.angle_model = tf.lite.Interpreter(model_path=self.saved_angle_model)
#         self.angle_model.allocate_tensors()
        
#         # Get input and output tensors
#         self.speed_input_details = self.speed_model.get_input_details()
#         self.speed_output_details = self.speed_model.get_output_details()
        
#         self.angle_input_details = self.angle_model.get_input_details()
#         self.angle_output_details = self.angle_model.get_output_details()
    
#     def preprocess(self, image):
#         im = tf.image.convert_image_dtype(image, tf.float32)
#         im /= 255.0  # Normalize to [0,1]
#         im = tf.image.resize(im, [224, 224])  # Should be whatever the model input size is
#         im = tf.expand_dims(im, axis=0)
#         return im
    
#     def predict(self, image):
#         processed_image = self.preprocess(image)
        
#         # Convert to numpy array since TFLite expects numpy arrays
#         if isinstance(processed_image, tf.Tensor):
#             processed_image = processed_image.numpy()
        
#         # Use speed model
#         self.speed_model.set_tensor(self.speed_input_details[0]['index'], processed_image)
#         self.speed_model.invoke()
#         speed_prediction = self.speed_model.get_tensor(self.speed_output_details[0]['index'])
        
#         # Use angle model
#         self.angle_model.set_tensor(self.angle_input_details[0]['index'], processed_image)
#         self.angle_model.invoke()
#         angle_prediction = self.angle_model.get_tensor(self.angle_output_details[0]['index'])
        
#         # Apply the same logic as in the "FROM TOM" version
#         speed_value = float(speed_prediction[0][0])
#         angle_value = float(angle_prediction[0][0])
        
#         angle = 80 * angle_value + 50
#         speed = 35 * speed_value
        
#         print('angle:', angle, 'speed:', speed)
        
#         return angle, speed


class Model:
 
    
    def __init__(self):
        self.saved_speed_model = 'mobnetv3small_classification_model.h5'
        self.saved_angle_model = 'mobnetv3small_regression_model.h5'
        self.speed_model = tf.keras.models.load_model(os.path.join(os.path.dirname(os.path.abspath(__file__)), self.saved_speed_model))
        self.angle_model = tf.keras.models.load_model(os.path.join(os.path.dirname(os.path.abspath(__file__)), self.saved_angle_model))

    def preprocess(self, image):
        im = tf.image.convert_image_dtype(image, tf.float32)
        im /= 255.0 # Normalize to [0,1]
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
        speed_prediction = np.around(predictions[0].numpy())
        angle_prediction = predictions[1].numpy()

        speed_value = float(speed_prediction[0][0])
        angle_value = float(angle_prediction[0][0])

        angle = 80 * angle_value + 50
        speed = 35 * speed_value
        return angle,speed