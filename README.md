# Instructions for Building, Converting, and Quantizing the Model

## Building the Model

1. **Prepare the Environment:**
   - Ensure you have Python 3.7 or higher installed.
   - Install the required Python packages using pip:
     ```bash
     pip install torch tensorflow fairseq
     ```

2. **Load Configuration:**
   - Load the configuration from the YAML file using Fairseq's options and parser.

3. **Load Model:**
   - Load the Fairseq task and the `Sign2TextTransformerModel`.
   - Load the trained model's weights from the checkpoint.

## Converting the Model to TensorFlow

1. **Convert PyTorch Model to ONNX:**
   - Create a dummy input tensor with the correct shape.
   - Use `torch.onnx.export` to convert the PyTorch model to ONNX format.

2. **Convert ONNX Model to TFLite:**
   - Load the ONNX model.
   - Convert the ONNX model to TensorFlow using `onnx_tf.backend.prepare`.
   - Convert the TensorFlow model to TFLite with quantization using `tf.lite.TFLiteConverter`.

3. **Verification:**
   - Load the TFLite model for verification.
   - Compare the outputs of the TFLite and PyTorch models to ensure they match within an acceptable tolerance.

## Quantizing the Model

1. **Perform Post-Training Quantization:**
   - Load the TFLite model.
   - Convert the model to a TensorFlow Lite model with quantization using `tf.lite.TFLiteConverter`.
   - Save the quantized model to a file.

2. **Save Quantized Model:**
   - Save the quantized TFLite model to a file.

3. **Verification:**
   - Verify the quantized model to ensure it meets the desired size and performance requirements.

## CLI Commands

1. **Convert PyTorch Model to ONNX:**
   ```bash
   python convert_to_tflite.py
   ```

2. **Quantize the TFLite Model:**
   ```bash
   python quantize_model.py
   ```
