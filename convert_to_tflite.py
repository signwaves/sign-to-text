import torch
import tensorflow as tf
import numpy as np
from fairseq.models.sign_to_text.sign2text_transformer import Sign2TextTransformerModel
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq import checkpoint_utils, options, tasks, utils
import os, time
import argparse
import onnx
from onnx_tf.backend import prepare

# --- Argument Parser ---
def get_parser():
    parser = argparse.ArgumentParser(
        description="Convert Fairseq Sign2TextTransformer Encoder to TFLite"
    )
    # fmt: off
    parser.add_argument('--checkpoint', required=True, metavar='FILE', help='Path to Fairseq model checkpoint (.pt)')
    parser.add_argument('--output-onnx', default='sign_encoder.onnx', metavar='FILE', help='Path to save the intermediate ONNX model')
    parser.add_argument('--output-tflite', default='sign_encoder.tflite', metavar='FILE', help='Path to save the final TFLite model')
    parser.add_argument('--seq-len', type=int, default=128, metavar='N', help='Maximum sequence length for dummy input')
    parser.add_argument('--feat-dim', type=int, required=True, metavar='N', help='Feature dimension of the input (e.g., MediaPipe landmarks)')
    parser.add_argument('--batch-size', type=int, default=1, metavar='N', help='Batch size for dummy input')
    parser.add_argument('--fp16', action='store_true', help='Enable FP16 quantization for TFLite')
    parser.add_argument('--int8', action='store_true', help='Enable INT8 quantization for TFLite (requires representative dataset)')
    parser.add_argument('--int8-dataset-npy', metavar='FILE', default=None, help='Path to a .npy file with representative data for INT8 calibration (shape: N, seq_len, feat_dim)')
    # Add Fairseq config overrides if necessary (often needed to find the task)
    parser.add_argument('--data-bin', metavar='DIR', help='Path to the binarized data directory (needed for task setup)')
    parser.add_argument('--task', default='sign_to_text', help='Fairseq task type')
    parser.add_argument('--source-lang', default='sign', help='Source language tag')
    parser.add_argument('--target-lang', default='en', help='Target language tag')

    # fmt: on
    return parser

def main(args):
    print("--- Configuration ---")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output ONNX: {args.output_onnx}")
    print(f"Output TFLite: {args.output_tflite}")
    print(f"Sequence Length: {args.seq_len}")
    print(f"Feature Dimension: {args.feat_dim}")
    print(f"Batch Size: {args.batch_size}")
    print(f"FP16 Quantization: {args.fp16}")
    print(f"INT8 Quantization: {args.int8}")
    if args.int8:
        print(f"INT8 Dataset: {args.int8_dataset_npy}")

    # --- 1. Load Fairseq Model ---
    print("\n--- 1. Loading Fairseq Model ---")
    # Use Fairseq's utility functions to load model and config
    # Need to provide enough task args for setup_task to work
    models, cfg, task = checkpoint_utils.load_model_ensemble_and_task(
        [args.checkpoint],
        arg_overrides={
            "data": args.data_bin, # Crucial if task relies on data dir
            "task": args.task,
            "source_lang": args.source_lang,
            "target_lang": args.target_lang,
        } if args.data_bin else {}, # Only override if data_bin is provided
    )
    model = models[0]

    # Ensure model is in eval mode and on CPU for consistency
    model.eval()
    model.cpu()
    print("Model loaded successfully.")
    # print("Model Config:", cfg) # Optional: Print loaded config

    # --- 2. Prepare Dummy Inputs for Encoder ---
    print("\n--- 2. Preparing Dummy Inputs for Encoder ---")
    # The encoder typically needs:
    # - src_tokens: The input features (batch, seq_len, feat_dim)
    # - src_lengths: The actual length of each sequence (batch,)
    dummy_src_tokens = torch.randn(args.batch_size, args.seq_len, args.feat_dim, dtype=torch.float32)
    # Assume sequences are full length for dummy input, can be adjusted
    dummy_src_lengths = torch.full((args.batch_size,), args.seq_len, dtype=torch.long)

    encoder_inputs = (dummy_src_tokens, dummy_src_lengths)
    input_names = ["src_tokens", "src_lengths"]
    output_names = ["encoder_out"] # The primary output from the encoder

    print(f"Dummy src_tokens shape: {dummy_src_tokens.shape}")
    print(f"Dummy src_lengths shape: {dummy_src_lengths.shape}")

    # --- 3. Convert PyTorch Encoder to ONNX ---
    print("\n--- 3. Converting Encoder to ONNX ---")
    # We specifically target the 'encoder' part of the model
    encoder_model = model.encoder

    # Define dynamic axes for variable batch size and sequence length
    dynamic_axes = {
        "src_tokens": {0: "batch_size", 1: "sequence_length"},
        "src_lengths": {0: "batch_size"},
        "encoder_out": {0: "sequence_length", 1: "batch_size"}, # Note: Fairseq encoder output is often (T, B, C)
    }

    try:
        torch.onnx.export(
            encoder_model,
            encoder_inputs,
            args.output_onnx,
            export_params=True,
            opset_version=12, # Increased opset version, 11 or higher recommended
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            verbose=False # Set to True for detailed debugging
        )
        print(f"ONNX model saved successfully to {args.output_onnx}")
        # Verify the ONNX model
        onnx_model = onnx.load(args.output_onnx)
        onnx.checker.check_model(onnx_model)
        print("ONNX model check passed.")

    except Exception as e:
        print(f"Error during ONNX export: {e}")
        print("Please check model compatibility, input shapes, and opset version.")
        return # Exit if ONNX export fails

    # --- 4. Convert ONNX model to TFLite ---
    print("\n--- 4. Converting ONNX to TFLite ---")
    try:
        # Load ONNX model
        onnx_model = onnx.load(args.output_onnx)

        # Convert ONNX model to TensorFlow representation
        tf_rep = prepare(onnx_model) # onnx_tf backend

        # Get the concrete function signature
        # Important: Must match the input names used during ONNX export!
        input_signature = []
        for name in input_names:
            for detail in tf_rep.inputs: # tf_rep.inputs lists actual TF input names
                if name in detail: # Match based on name used in ONNX export
                    # Create spec based on ONNX model's input info
                    onnx_input = next(i for i in onnx_model.graph.input if i.name == name)
                    dtype = tf.float32 if onnx_input.type.tensor_type.elem_type == onnx.TensorProto.FLOAT else tf.int64
                    # Handle dynamic axes in shape (None)
                    shape = [d.dim_value if d.dim_value > 0 else None for d in onnx_input.type.tensor_type.shape.dim]
                    input_signature.append(tf.TensorSpec(shape=shape, dtype=dtype, name=name))
                    break
        if len(input_signature) != len(input_names):
             print("Warning: Could not perfectly match ONNX input names to TF signature. Using generic signature.")
             # Fallback if matching fails (less reliable)
             concrete_func = tf_rep.tf_module.__call__.get_concrete_function()
        else:
             print(f"Input Signature: {input_signature}")
             concrete_func = tf_rep.tf_module.__call__.get_concrete_function(*input_signature)


        # Convert TensorFlow graph function to TFLite
        converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func], tf_rep.tf_module)

        # Apply Quantization
        if args.fp16:
            print("Applying FP16 quantization...")
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]
        elif args.int8:
            print("Applying INT8 quantization...")
            if args.int8_dataset_npy is None:
                print("Error: --int8 requires --int8-dataset-npy for calibration.")
                return
            try:
                print(f"Loading INT8 calibration data from {args.int8_dataset_npy}...")
                calibration_data = np.load(args.int8_dataset_npy).astype(np.float32)
                # Ensure calibration data matches expected input dimensions
                if calibration_data.ndim != 3 or calibration_data.shape[2] != args.feat_dim:
                     raise ValueError(f"Calibration data shape mismatch. Expected (N, seq_len, {args.feat_dim}), Got {calibration_data.shape}")
                print(f"Calibration data shape: {calibration_data.shape}")

                def representative_dataset_gen():
                    # Adapt this based on how many inputs your *encoder* takes
                    # Here we assume src_tokens is the primary input needed for calibration
                    count = 0
                    max_samples = 100 # Limit number of calibration samples
                    for i in range(min(max_samples, calibration_data.shape[0])):
                        # Need representative lengths too if model uses them heavily
                        # For simplicity, using max length here. Adjust if needed.
                        rep_len = np.array([min(args.seq_len, calibration_data.shape[1])], dtype=np.int64)
                        rep_tok = calibration_data[i:i+1, :args.seq_len, :] # Ensure correct seq len
                        # Yield a list matching the order of concrete_func.inputs
                        # THIS ORDER IS CRITICAL and must match concrete_func.inputs!
                        # Check concrete_func.structured_input_signature to be sure.
                        # Typically it's [src_tokens, src_lengths] if input_names was correct.
                        yield [rep_tok, rep_len]
                        count += 1
                    print(f"Generated {count} calibration samples.")


                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                converter.representative_dataset = representative_dataset_gen
                # Force INT8 input/output if desired, otherwise uses FLOAT32
                # converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
                # converter.inference_input_type = tf.int8 # or tf.uint8
                # converter.inference_output_type = tf.int8 # or tf.uint8
            except Exception as e:
                print(f"Error during INT8 setup: {e}")
                return
        else:
            # Default: No specific quantization beyond standard optimizations
             converter.optimizations = [tf.lite.Optimize.DEFAULT]


        # Allow custom ops if needed (less common for standard layers)
        converter.allow_custom_ops = True
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,  # Enable TensorFlow Lite ops.
            tf.lite.OpsSet.SELECT_TF_OPS,  # Enable TensorFlow ops (if needed).
        ]


        tflite_model = converter.convert()

        # Save the TFLite model
        with open(args.output_tflite, "wb") as f:
            f.write(tflite_model)
        print(f"TFLite model saved successfully to {args.output_tflite}")

    except Exception as e:
        print(f"Error during TFLite conversion: {e}")
        print("Check TensorFlow/ONNX compatibility, model ops, and quantization settings.")
        return # Exit if TFLite conversion fails

    # --- 5. Verification (Optional but Recommended) ---
    print("\n--- 5. Verification ---")
    try:
        # Load the TFLite model
        interpreter = tf.lite.Interpreter(model_path=args.output_tflite)
        interpreter.allocate_tensors()

        # Get input and output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print(f"TFLite Input Details: {input_details}")
        print(f"TFLite Output Details: {output_details}")

        # Prepare dummy input for TFLite (match expected types and shapes)
        tflite_dummy_inputs = {}
        # Ensure numpy inputs match TFLite expected dtypes
        tflite_dummy_inputs[input_details[0]['index']] = dummy_src_tokens.numpy().astype(input_details[0]['dtype'])
        if len(input_details) > 1: # If src_lengths is also an input
             tflite_dummy_inputs[input_details[1]['index']] = dummy_src_lengths.numpy().astype(input_details[1]['dtype'])

        # Set the input tensors
        for index, tensor_data in tflite_dummy_inputs.items():
            interpreter.set_tensor(index, tensor_data)

        # Run inference
        start_time = time.time()
        interpreter.invoke()
        end_time = time.time()
        tflite_inference_time = end_time - start_time

        # Get the output tensor (assuming the main encoder output is the first one)
        tflite_output = interpreter.get_tensor(output_details[0]["index"])

        print(f"TFLite model inference time: {tflite_inference_time:.4f} seconds")
        print(f"TFLite Output Shape: {tflite_output.shape}")


        # Run the original PyTorch encoder for comparison
        with torch.no_grad():
            start_time = time.time()
            # Pass the *same* dummy inputs used for export
            pytorch_encoder_output_dict = model.encoder(
                src_tokens=dummy_src_tokens, src_lengths=dummy_src_lengths
            )
            end_time = time.time()
            pytorch_inference_time = end_time - start_time

            # Extract the primary encoder output tensor - often under 'encoder_out' key
            # The output is typically [T, B, C]. TFLite might transpose this.
            # Check the structure of pytorch_encoder_output_dict if unsure.
            # Assuming 'encoder_out' is a tuple/list containing the tensor [T, B, C]
            pytorch_output = pytorch_encoder_output_dict["encoder_out"][0].numpy()

        print(f"PyTorch Encoder inference time: {pytorch_inference_time:.4f} seconds")
        print(f"PyTorch Output Shape: {pytorch_output.shape}")


        # Compare outputs (handle potential transpose)
        # TFLite output might be (B, T, C) while PyTorch is (T, B, C)
        if tflite_output.shape != pytorch_output.shape:
            try:
                 # Attempt transpose if shapes mismatch in a standard way (B,T,C vs T,B,C)
                 if (tflite_output.shape[0] == pytorch_output.shape[1] and
                     tflite_output.shape[1] == pytorch_output.shape[0] and
                     tflite_output.shape[2] == pytorch_output.shape[2]):
                     print("Transposing PyTorch output (T, B, C) -> (B, T, C) for comparison.")
                     pytorch_output = np.transpose(pytorch_output, (1, 0, 2))
                 else:
                     print("Warning: Output shape mismatch is not a simple transpose.")
            except:
                 print("Warning: Could not determine transpose for shape mismatch.")


        if tflite_output.shape == pytorch_output.shape:
            diff = np.abs(tflite_output - pytorch_output)
            max_diff = np.max(diff)
            mean_diff = np.mean(diff)
            tolerance = 1e-2 if args.fp16 or args.int8 else 1e-4 # Looser tolerance for quantized models

            print(f"\nMax difference between TFLite and PyTorch outputs: {max_diff:.6f}")
            print(f"Mean difference between TFLite and PyTorch outputs: {mean_diff:.6f}")

            if max_diff < tolerance:
                print("Verification successful: TFLite and PyTorch Encoder outputs are close.")
            else:
                print("Verification WARNING: TFLite and PyTorch Encoder outputs differ significantly.")
        else:
             print(f"Verification FAILED: Output shapes mismatch! TFLite={tflite_output.shape}, PyTorch={pytorch_output.shape}")


    except Exception as e:
        print(f"Error during verification: {e}")

    print("\n--- Conversion process finished. ---")


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)