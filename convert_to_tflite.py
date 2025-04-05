import torch
import tensorflow as tf
import numpy as np
from fairseq.models.sign_to_text.sign2text_transformer import Sign2TextTransformerModel
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq import checkpoint_utils, options, tasks
import os, time

# Configuration
checkpoint_path = "final_models/baseline_6_3_dp03_wd_2/ckpts/checkpoint.best_reduced_sacrebleu_3.5401.pt"  # Path to your trained PyTorch model checkpoint
config_path = "examples/sign_language/config/wicv_cvpr23/i3d_best/baseline_6_3_dp03_wd_2.yaml"  # Path to your model's configuration file
output_path = "sign2text_transformer_hand_landmarks.tflite"  # Path to save the TFLite model
sequence_length = 64
feat_dim = 63  # Hand landmark features

# --- 1. Load Configuration ---
# Load the configuration from the YAML file using Fairseq's options and parser
parser = options.get_training_parser()
args = options.parse_args_and_arch(
    parser,
    input_args=[
        "--config-dir",
        os.path.dirname(config_path),
        "--config-name",
        os.path.basename(config_path).replace(".yaml", ""),
    ],
)
cfg = convert_namespace_to_omegaconf(args)

# --- 2. Load Model ---
# Load the Fairseq task and the Sign2TextTransformerModel
task = tasks.setup_task(cfg.task)
model = Sign2TextTransformerModel.build_model(cfg.model, task)

# Load the trained model's weights from the checkpoint
state = checkpoint_utils.load_checkpoint_to_cpu(checkpoint_path)
model.load_state_dict(state["model"], strict=True)
model.eval()


# --- 3. Adapt Model for TFLite (if needed) ---
# In our case, we have already modified the model code in sign2text_transformer.py.
# Further adaptations (like changing activation functions) can be done here if needed.


# --- 4. Create Dummy Input ---
# Create a dummy input tensor with the correct shape (batch_size, sequence_length, feat_dim)
dummy_input = torch.randn(1, sequence_length, feat_dim)  # Batch size 1

# 5. Convert to TFLite
class TFLiteModel(tf.Module):
    def __init__(self, model):
        super(TFLiteModel, self).__init__()
        self.model = model

    @tf.function(input_signature=[tf.TensorSpec(shape=[1, sequence_length, feat_dim], dtype=tf.float32)])
    def __call__(self, x):
        # Convert TensorFlow tensor to PyTorch tensor
        x = torch.from_numpy(x.numpy())
        # Run inference with the PyTorch model's encoder
        encoder_out = self.model.encoder(
            src_tokens=x, encoder_padding_mask=torch.zeros(1, sequence_length).bool()
        )
        # Return the encoder output as a NumPy array
        return encoder_out["encoder_out"][0].numpy()

# Create an instance of the TFLiteModel wrapper
tflite_model = TFLiteModel(model)

# --- 5. Convert to TFLite ---
# Create a TFLite converter from the concrete function of the TFLiteModel
converter = tf.lite.TFLiteConverter.from_concrete_functions(
    [tflite_model.__call__.get_concrete_function()], tflite_model
)

# Set the inference input type to float32 explicitly
converter.inference_input_type = tf.float32

# Apply default optimizations, including quantization
converter.optimizations = [tf.lite.Optimize.DEFAULT]
# Convert the model
tflite_model = converter.convert()

# --- 6. Save TFLite Model ---
# Save the converted TFLite model to a file
with open(output_path, "wb") as f:
    f.write(tflite_model)

print(f"TFLite model saved to: {output_path}")

# --- 7. Verification ---
# Load the TFLite model for verification
interpreter = tf.lite.Interpreter(model_path=output_path)
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Create a dummy input for the TFLite model
tflite_dummy_input = np.random.rand(1, sequence_length, feat_dim).astype(np.float32)

# Set the input tensor
interpreter.set_tensor(input_details[0]["index"], tflite_dummy_input)

# Run inference
start_time = time.time()
interpreter.invoke()
end_time = time.time()

# Get the output tensor
tflite_output = interpreter.get_tensor(output_details[0]["index"])

print(f"TFLite model inference time: {end_time - start_time:.4f} seconds")

# Run the PyTorch model for comparison
with torch.no_grad():
    pytorch_output = model.encoder(
        src_tokens=dummy_input, encoder_padding_mask=torch.zeros(1, sequence_length).bool()
    )["encoder_out"][0].numpy()

# Check if the outputs are close (within a certain tolerance)
diff = np.abs(tflite_output - pytorch_output)
max_diff = np.max(diff)
tolerance = 1e-3  # Adjust as needed

print(f"Max difference between TFLite and PyTorch outputs: {max_diff:.6f}")

if max_diff < tolerance:
    print("Verification successful: TFLite and PyTorch model outputs are close.")
else:
    print(
        "Verification failed: TFLite and PyTorch model outputs differ significantly."
    )