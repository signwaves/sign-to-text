import torch
import tensorflow as tf
import numpy as np
from fairseq.models.sign_to_text.sign2text_transformer import Sign2TextTransformerModel
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq import checkpoint_utils, options, tasks
import os, time
import torch.onnx

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

# --- 3. Convert PyTorch model to ONNX ---
def convert_to_onnx(model, dummy_input, onnx_path):
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size", 1: "sequence_length"}, "output": {0: "batch_size", 1: "sequence_length"}},
    )

# --- 4. Convert ONNX model to TFLite ---
def convert_to_tflite(onnx_path, tflite_path):
    import onnx
    from onnx_tf.backend import prepare

    # Load ONNX model
    onnx_model = onnx.load(onnx_path)

    # Convert ONNX model to TensorFlow
    tf_rep = prepare(onnx_model)

    # Convert TensorFlow model to TFLite with quantization
    converter = tf.lite.TFLiteConverter.from_concrete_functions([tf_rep.tf_module.__call__.get_concrete_function()])
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    # Save the TFLite model
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)

# --- 5. Create Dummy Input ---
# Create a dummy input tensor with the correct shape (batch_size, sequence_length, feat_dim)
dummy_input = torch.randn(1, sequence_length, feat_dim)  # Batch size 1

# --- 6. Convert to ONNX ---
onnx_path = "sign2text_transformer.onnx"
convert_to_onnx(model, dummy_input, onnx_path)

# --- 7. Convert to TFLite ---
convert_to_tflite(onnx_path, output_path)

# --- 8. Verification ---
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
