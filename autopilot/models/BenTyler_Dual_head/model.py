# import os
# import sys
# # from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
# import numpy as np
# import tensorflow as tf

# class Model:
#     os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices' #enable gpu
#     model_name = 'mobnet.tflite'

#     def __init__(self):
#         print('init model')

#         self.interpreter = tf.lite.Interpreter(
#             model_path = os.path.join(
#                 os.path.dirname(os.path.abspath(__file__)), self.model_name
#             )
#         )
#         self.interpreter.allocate_tensors()
#         self.input_details = self.interpreter.get_input_details()
#         self.output_details = self.interpreter.get_output_details()

#     def predict(self, image):
#         print('make prediction')


#         input_index = self.input_details[0]['index']

#         processed_image = self.preprocess(image)
#         self.interpreter.set_tensor(input_index, processed_image)

#         self.interpreter.invoke()

#         print()
#         print(self.output_details)
#         print()

#         angle_index = self.output_details[1]['index']
#         speed_index = self.output_details[0]['index']

#         angle_output = self.interpreter.get_tensor(angle_index)
#         speed_output = self.interpreter.get_tensor(speed_index)

#         angle = angle_output[0]
#         speed = speed_output[0]
#         print(f'raw angle: {angle}, raw speed: {speed}')
#         angle, speed = self.postprocess(angle, speed)

#         return angle, speed
    
#     # @staticmethod
#     def postprocess(self, angle, speed):
#         possible_angles = np.arange(0,1.01,0.0625)
#         possible_speeds = np.array([0,1])

#         def find_closest_value(x, candidate_list):
#             return min(candidate_list, key=lambda val: abs(val - x))
        
#         angle = find_closest_value(angle, possible_angles)
#         speed = find_closest_value(speed, possible_speeds)

#         angle = angle * 80 + 50
#         speed = speed * 35

#         print(f'angle: {angle}, speed: {speed}')
#         return angle, speed
    
#     # @staticmethod
#     def preprocess(self, image):
#         im = tf.image.resize(image, [224, 224])
#         # Then normalize to [0,1] range - matching training exactly
#         im = im / 255.0
#         # Add debugging to see values
#         print("Image min/max after normalization:", tf.reduce_min(im).numpy(), tf.reduce_max(im).numpy())
#         # Add batch dimension
#         im = tf.expand_dims(im, axis=0)
#         return im

import numpy as np
import tensorflow as tf
import os

class Model:
    os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'  # enable gpu
    model_path = 'mobnet.tflite'
    
    def __init__(self):
        try:
            # Try to use Edge TPU if available
            delegate = tf.lite.experimental.load_delegate('libedgetpu.so.1')  # 'libedgetpu.1.dylib' for mac or 'libedgetpu.so.1' for linux
            print('Using TPU')
            
            self.interpreter = tf.lite.Interpreter(
                model_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), self.model_path),
                experimental_delegates=[delegate]
            )
        except ValueError:
            print('Fallback to CPU')
            
            self.interpreter = tf.lite.Interpreter(
                model_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), self.model_path)
            )
        
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.floating_model = self.input_details[0]['dtype'] == np.float32
    
    def preprocess(self, image):
        # im = tf.image.convert_image_dtype(image, tf.float32)
        im = tf.image.resize(image, [224, 224])
        im /= 255.0
        im = tf.expand_dims(im, axis=0)  # add batch dimension
        return im
    
    def predict(self, image):
        image = self.preprocess(image)
        
        self.interpreter.set_tensor(self.input_details[0]['index'], image)
        self.interpreter.invoke()
        
        # Get output in format [speed, angle]
        pred_speed = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        pred_angle = self.interpreter.get_tensor(self.output_details[1]['index'])[0]
        print(f'raw angle: {pred_angle}, raw speed: {pred_speed}')

        speed = np.around(pred_speed[0]).astype(int)*35
        angle = pred_angle[0]*80+50
            
        return angle, speed