# import numpy as np
# import tensorflow as tf
# import os

# class Model:
#     os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices' #enable gpu
#     speed_model = 'tflite_finetuned_mbnetv3small_classification.tflite'
#     angle_model = 'tflite_finetuned_mobnetv3small_regression.tflite'

#     def __init__(self):
        
#         try:
#             delegate = tf.lite.experimental.load_delegate('libedgetpu.so.1') #'libedgetpu.1.dylib' for mac or 'libedgetpu.so.1' for linux
#             print('Using TPU')
            
#             self.speed_interpreter = tf.lite.Interpreter(model_path=os.path.join(os.path.dirname(os.path.abspath(__file__)),
#                                                                              self.speed_model),experimental_delegates=[delegate])
#             self.angle_interpreter = tf.lite.Interpreter(model_path=os.path.join(os.path.dirname(os.path.abspath(__file__)),
#                                                                              self.angle_model), experimental_delegates=[delegate])
          
#         except ValueError:
#             print('Fallback to CPU')

#             self.speed_interpreter = tf.lite.Interpreter(model_path=os.path.join(os.path.dirname(os.path.abspath(__file__)),
#                                                                              self.speed_model))
#             self.angle_interpreter = tf.lite.Interpreter(model_path=os.path.join(os.path.dirname(os.path.abspath(__file__)),
#                                                                              self.angle_model))
            

#         self.speed_interpreter.allocate_tensors()
#         self.angle_interpreter.allocate_tensors()
#         self.speed_input_details = self.speed_interpreter.get_input_details()
#         self.speed_output_details = self.speed_interpreter.get_output_details()
#         self.angle_input_details = self.angle_interpreter.get_input_details()
#         self.angle_output_details = self.angle_interpreter.get_output_details()
#         self.floating_model = self.speed_input_details[0]['dtype'] == np.float32         # check the type of the input tensor

#     def preprocess(self, image):
#         im = tf.image.convert_image_dtype(image, tf.float32)
#         print(im)
#         im = tf.image.resize(im, [224, 224])
#         im = tf.expand_dims(im, axis=0) #add batch dimension
#         return im

#     def predict(self, image):
#         angles = np.arange(17)*5+50
#         image = self.preprocess(image)

#         self.speed_interpreter.set_tensor(self.speed_input_details[0]['index'], image)
#         self.angle_interpreter.set_tensor(self.angle_input_details[0]['index'], image)

#         self.speed_interpreter.invoke()
#         self.angle_interpreter.invoke()

#         pred_speed = self.speed_interpreter.get_tensor(self.speed_output_details[0]['index'])[0]
#         speed = np.around(pred_speed[0]).astype(int)*35

#         pred_angle = self.angle_interpreter.get_tensor(self.angle_output_details[0]['index'])[0]
#         angle = angles[np.argmax(pred_angle)]
        
#         return angle, speed

# CLAUDE

import numpy as np
import tensorflow as tf
import os

class Model:
    os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices' #enable gpu
    speed_model = 'tflite_finetuned_mbnetv3small_classification.tflite'
    angle_model = 'tflite_finetuned_mobnetv3small_regression.tflite'

    def __init__(self):
        
        try:
            delegate = tf.lite.experimental.load_delegate('libedgetpu.so.1') #'libedgetpu.1.dylib' for mac or 'libedgetpu.so.1' for linux
            print('Using TPU')
            
            self.speed_interpreter = tf.lite.Interpreter(model_path=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                                             self.speed_model),experimental_delegates=[delegate])
            self.angle_interpreter = tf.lite.Interpreter(model_path=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                                             self.angle_model), experimental_delegates=[delegate])
          
        except ValueError:
            print('Fallback to CPU')

            self.speed_interpreter = tf.lite.Interpreter(model_path=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                                             self.speed_model))
            self.angle_interpreter = tf.lite.Interpreter(model_path=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                                             self.angle_model))
            

        self.speed_interpreter.allocate_tensors()
        self.angle_interpreter.allocate_tensors()
        self.speed_input_details = self.speed_interpreter.get_input_details()
        self.speed_output_details = self.speed_interpreter.get_output_details()
        self.angle_input_details = self.angle_interpreter.get_input_details()
        self.angle_output_details = self.angle_interpreter.get_output_details()
        self.floating_model = self.speed_input_details[0]['dtype'] == np.float32         # check the type of the input tensor

    def preprocess(self, image):
        # CHANGED: Updated preprocessing to match training pipeline
        # First resize the image
        im = tf.image.resize(image, [224, 224])
        # Then normalize to [0,1] range - matching training exactly
        im = im / 255.0
        # Add debugging to see values
        print("Image min/max after normalization:", tf.reduce_min(im).numpy(), tf.reduce_max(im).numpy())
        # Add batch dimension
        im = tf.expand_dims(im, axis=0)
        return im

    def predict(self, image):
        # angles = np.arange(17)*5+50
        image = self.preprocess(image)

        # ADDED: Additional debugging to see raw model outputs
        print("Input shape:", image.shape)
        
        self.speed_interpreter.set_tensor(self.speed_input_details[0]['index'], image)
        self.angle_interpreter.set_tensor(self.angle_input_details[0]['index'], image)

        self.speed_interpreter.invoke()
        self.angle_interpreter.invoke()

        pred_speed = self.speed_interpreter.get_tensor(self.speed_output_details[0]['index'])[0]
        # ADDED: Print raw predictions for debugging
        print("Raw speed prediction:", pred_speed)
        speed = np.around(pred_speed[0]).astype(int)*35

        pred_angle = self.angle_interpreter.get_tensor(self.angle_output_details[0]['index'])[0]
        # ADDED: Print raw predictions for debugging
        print("Raw angle prediction:", pred_angle)
        # angle = angles[np.argmax(pred_angle)]
        angle = pred_angle[0]*80 + 50
        
        return angle, speed