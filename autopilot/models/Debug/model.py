import numpy as np
import tensorflow as tf
import os

# ============= Diagnostic Functions =============

def inspect_tflite_model(model_path):
    """Print detailed info about a TFLite model"""
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    # Get input details
    input_details = interpreter.get_input_details()
    print("\n===== INPUT DETAILS =====")
    for input_detail in input_details:
        print(f"Name: {input_detail['name']}")
        print(f"Shape: {input_detail['shape']}")
        print(f"Type: {input_detail['dtype']}")
        print(f"Quantization: {input_detail.get('quantization', 'None')}")
        print()
    
    # Get output details
    output_details = interpreter.get_output_details()
    print("\n===== OUTPUT DETAILS =====")
    for output_detail in output_details:
        print(f"Name: {output_detail['name']}")
        print(f"Shape: {output_detail['shape']}")
        print(f"Type: {output_detail['dtype']}")
        print(f"Quantization: {output_detail.get('quantization', 'None')}")
        print()
    
    return interpreter, input_details, output_details

def test_random_inputs(interpreter, input_details, output_details, num_tests=5):
    """Test model with random inputs to see if outputs vary"""
    input_shape = input_details[0]['shape']
    input_dtype = input_details[0]['dtype']
    
    print(f"\nTesting {num_tests} random inputs...")
    
    all_outputs = []
    for i in range(num_tests):
        # Create random input (values between 0-1 for normalized images)
        if input_dtype == np.float32:
            random_input = np.random.random(input_shape).astype(np.float32)
        else:
            # For uint8 quantized models
            random_input = np.random.randint(0, 256, input_shape).astype(np.uint8)
        
        # Set the input tensor
        interpreter.set_tensor(input_details[0]['index'], random_input)
        
        # Run inference
        interpreter.invoke()
        
        # Get the output
        output = interpreter.get_tensor(output_details[0]['index'])
        print(f"Test {i+1} output: {output}")
        all_outputs.append(output)
    
    # Check if all outputs are the same
    first_output = all_outputs[0]
    all_same = all(np.array_equal(output, first_output) for output in all_outputs)
    
    if all_same:
        print("\n⚠️ WARNING: All outputs are identical despite different inputs!")
    else:
        print("\n✓ Model produces different outputs for different inputs.")
    
    return all_outputs

def compare_with_original_keras(keras_model_path, tflite_model_path, test_image):
    """Compare predictions between original Keras model and TFLite model"""
    # Load Keras model if available
    try:
        keras_model = tf.keras.models.load_model(keras_model_path)
        
        # Preprocess for Keras (assuming same preprocessing)
        keras_input = test_image.copy()
        keras_input = tf.image.resize(keras_input, [224, 224])
        keras_input = keras_input / 255.0
        keras_input = tf.expand_dims(keras_input, axis=0)
        
        # Get Keras prediction
        keras_pred = keras_model.predict(keras_input)
        print("\n===== KERAS PREDICTION =====")
        print(keras_pred)
        
        # Now get TFLite prediction
        interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # Prepare input based on required data type
        if input_details[0]['dtype'] == np.float32:
            tflite_input = keras_input.numpy()
        else:
            # For uint8 quantized models, scale appropriately
            tflite_input = (test_image * 255).astype(np.uint8)
            tflite_input = tf.image.resize(tflite_input, [224, 224])
            tflite_input = tf.expand_dims(tflite_input, axis=0)
        
        interpreter.set_tensor(input_details[0]['index'], tflite_input)
        interpreter.invoke()
        tflite_pred = interpreter.get_tensor(output_details[0]['index'])
        
        print("\n===== TFLITE PREDICTION =====")
        print(tflite_pred)
        
        # Compare results
        if keras_pred.shape != tflite_pred.shape:
            print(f"\n⚠️ Shape mismatch: Keras {keras_pred.shape} vs TFLite {tflite_pred.shape}")
        
        # Calculate difference
        try:
            diff = np.abs(keras_pred - tflite_pred)
            avg_diff = np.mean(diff)
            max_diff = np.max(diff)
            print(f"\nAverage difference: {avg_diff}")
            print(f"Maximum difference: {max_diff}")
        except:
            print("\nCouldn't directly compare outputs due to shape/type differences")
        
        return keras_pred, tflite_pred
    
    except Exception as e:
        print(f"Error comparing models: {e}")
        return None, None

# ============= Modified Model Class =============

class ImprovedModel:
    def __init__(self):
        # Model paths
        self.speed_model_path = 'tflite_finetunedstratifiedsplit_extradata_mobnetv3small_speedclassif.tflite'
        self.angle_model_path = 'tflite_extradata_angleregression_finetuned_mobnetv3small.tflite'
        
        # Original Keras models if available (update paths as needed)
        self.keras_speed_model_path = 'original_speed_model.h5'  # Update with your path
        self.keras_angle_model_path = 'original_angle_model.h5'  # Update with your path
        
        # Initialize interpreters
        try:
            print("Loading TFLite models...")
            self.speed_interpreter = tf.lite.Interpreter(model_path=os.path.join(
                os.path.dirname(os.path.abspath(__file__)), self.speed_model_path))
            self.angle_interpreter = tf.lite.Interpreter(model_path=os.path.join(
                os.path.dirname(os.path.abspath(__file__)), self.angle_model_path))
            
            # Allocate tensors
            self.speed_interpreter.allocate_tensors()
            self.angle_interpreter.allocate_tensors()
            
            # Get details
            self.speed_input_details = self.speed_interpreter.get_input_details()
            self.speed_output_details = self.speed_interpreter.get_output_details()
            self.angle_input_details = self.angle_interpreter.get_input_details()
            self.angle_output_details = self.angle_interpreter.get_output_details()
            
            # Check input types
            self.speed_floating_model = self.speed_input_details[0]['dtype'] == np.float32
            self.angle_floating_model = self.angle_input_details[0]['dtype'] == np.float32
            
            print(f"Speed model is floating point: {self.speed_floating_model}")
            print(f"Angle model is floating point: {self.angle_floating_model}")
            
        except Exception as e:
            print(f"Error initializing model: {e}")
            raise
    
    def preprocess(self, image, for_model='both'):
        """Preprocess image according to model requirements"""
        # Resize image
        im = tf.image.resize(image, [224, 224])
        
        if for_model == 'speed' and not self.speed_floating_model:
            # For uint8 quantized model
            im_uint8 = tf.cast(im, tf.uint8)
            im_batched = tf.expand_dims(im_uint8, axis=0)
            print("Preprocessed for quantized speed model")
            return im_batched
        
        elif for_model == 'angle' and not self.angle_floating_model:
            # For uint8 quantized model
            im_uint8 = tf.cast(im, tf.uint8)
            im_batched = tf.expand_dims(im_uint8, axis=0)
            print("Preprocessed for quantized angle model")
            return im_batched
        
        else:
            # For floating point models
            im_normalized = im / 255.0
            im_batched = tf.expand_dims(im_normalized, axis=0)
            print("Preprocessed for floating point model")
            print(f"Image shape: {im_batched.shape}, min: {tf.reduce_min(im_batched).numpy():.4f}, max: {tf.reduce_max(im_batched).numpy():.4f}")
            return im_batched
    
    def predict(self, image):
        """Predict angle and speed with detailed debugging"""
        # Process for speed model
        speed_input = self.preprocess(image, for_model='speed')
        print(f"Speed input shape: {speed_input.shape}")
        
        # Process for angle model
        angle_input = self.preprocess(image, for_model='angle')
        print(f"Angle input shape: {angle_input.shape}")
        
        # Run speed model
        print("\nRunning speed model inference...")
        self.speed_interpreter.set_tensor(self.speed_input_details[0]['index'], speed_input)
        self.speed_interpreter.invoke()
        
        # Get speed output and apply post-processing
        raw_speed = self.speed_interpreter.get_tensor(self.speed_output_details[0]['index'])
        print(f"Raw speed model output shape: {raw_speed.shape}, values: {raw_speed}")
        
        # Properly handle the output based on its shape
        if len(raw_speed.shape) > 1 and raw_speed.shape[1] > 1:
            # It's likely a classification model with multiple classes
            predicted_class = np.argmax(raw_speed, axis=1)[0]
            speed = predicted_class * 35  # Your speed scaling
            print(f"Speed classification model: class {predicted_class} → {speed}")
        else:
            # It's likely a regression model
            speed = np.around(raw_speed[0][0]).astype(int) * 35
            print(f"Speed regression model: {raw_speed[0][0]} → {speed}")
        
        # Run angle model
        print("\nRunning angle model inference...")
        self.angle_interpreter.set_tensor(self.angle_input_details[0]['index'], angle_input)
        self.angle_interpreter.invoke()
        
        # Get angle output and apply post-processing
        raw_angle = self.angle_interpreter.get_tensor(self.angle_output_details[0]['index'])
        print(f"Raw angle model output shape: {raw_angle.shape}, values: {raw_angle}")
        
        # Properly handle the output based on its shape
        if len(raw_angle.shape) > 1 and raw_angle.shape[1] > 1:
            # It's likely a classification model with multiple classes
            angles = np.arange(raw_angle.shape[1]) * 5 + 50
            predicted_class = np.argmax(raw_angle, axis=1)[0]
            angle = angles[predicted_class]
            print(f"Angle classification model: class {predicted_class} → {angle}")
        else:
            # It's likely a regression model
            angle = raw_angle[0][0] * 80 + 50
            print(f"Angle regression model: {raw_angle[0][0]} → {angle}")
        
        return angle, speed


# ============= Usage Example =============

def run_diagnostics():
    """Run comprehensive diagnostics on the TFLite models"""
    # Model paths
    speed_model_path = 'tflite_finetunedstratifiedsplit_extradata_mobnetv3small_speedclassif.tflite'
    angle_model_path = 'tflite_extradata_angleregression_finetuned_mobnetv3small.tflite'
    
    # 1. Inspect models
    print("\n========== SPEED MODEL INSPECTION ==========")
    speed_interpreter, speed_input_details, speed_output_details = inspect_tflite_model(speed_model_path)
    
    print("\n========== ANGLE MODEL INSPECTION ==========")
    angle_interpreter, angle_input_details, angle_output_details = inspect_tflite_model(angle_model_path)
    
    # 2. Test with random inputs
    print("\n========== SPEED MODEL RANDOM INPUT TEST ==========")
    test_random_inputs(speed_interpreter, speed_input_details, speed_output_details)
    
    print("\n========== ANGLE MODEL RANDOM INPUT TEST ==========")
    test_random_inputs(angle_interpreter, angle_input_details, angle_output_details)
    
    # 3. Test the improved model with a sample image
    try:
        # Create a random test image (replace with your actual image loading)
        test_image = np.random.random((480, 640, 3)).astype(np.float32)
        
        print("\n========== TESTING IMPROVED MODEL ==========")
        improved_model = ImprovedModel()
        angle, speed = improved_model.predict(test_image)
        print(f"\nFinal prediction: Angle = {angle:.2f}, Speed = {speed}")
        
    except Exception as e:
        print(f"Error testing improved model: {e}")


if __name__ == "__main__":
    run_diagnostics()