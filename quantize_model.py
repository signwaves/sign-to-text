import tensorflow as tf
import numpy as np

def quantize_tflite_model(tflite_model_path, quantized_model_path):
    # Load the TFLite model
    with open(tflite_model_path, 'rb') as f:
        tflite_model = f.read()

    # Convert the model to a TensorFlow Lite model with quantization
    converter = tf.lite.TFLiteConverter.from_saved_model(tflite_model_path)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    quantized_tflite_model = converter.convert()

    # Save the quantized model to a file
    with open(quantized_model_path, 'wb') as f:
        f.write(quantized_tflite_model)

def main():
    tflite_model_path = "sign2text_transformer_hand_landmarks.tflite"
    quantized_model_path = "sign2text_transformer_hand_landmarks_quantized.tflite"
    quantize_tflite_model(tflite_model_path, quantized_model_path)
    print(f"Quantized model saved to {quantized_model_path}")

if __name__ == "__main__":
    main()
