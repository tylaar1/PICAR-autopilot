import numpy as np
import tensorflow as tf
import os

class Model:
    os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'  # enable gpu
    model_path = 'unified_model.tflite'
    
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
        im = tf.image.convert_image_dtype(image, tf.float32)
        im = tf.image.resize(im, [100, 100])
        im = tf.expand_dims(im, axis=0)  # add batch dimension
        return im
    
    def predict(self, image):
        image = self.preprocess(image)
        
        self.interpreter.set_tensor(self.input_details[0]['index'], image)
        self.interpreter.invoke()
        
        # Get output in format [speed, angle]
        prediction = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        
        # Process speed and angle based on model output
        # Assuming the model directly outputs speed and angle values
        speed = np.around(prediction[0]).astype(int) * 35
        
        # If your angle is still categorical, you can adapt this section
        # Example: if the second value represents an angle index
        angles = np.arange(17) * 5 + 50
        if len(prediction) > 2:  # If second value is a distribution over angles
            angle = angles[np.argmax(prediction[1:])]
        else:  # If second value is the direct angle
            angle = prediction[1]
            
        return angle, speed