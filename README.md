# Fairseq Sign Language Translation (how2sign Focus)

This project uses the Fairseq toolkit to train and evaluate sign language translation models, with a primary focus on the How2Sign dataset. The end goal of this project is to make a react native app that uses this tflite model to do live sign language translation. This README provides instructions for setting up the environment (especially on Google Colab), preparing data, training, evaluating, and converting models to TensorFlow Lite (TFLite) for potential mobile deployment (e.g., in React Native).

## Overview of the end goal of this project

You're building an end-to-end sign language translation app in React Native Expo:

    Input: User performs sign language in front of the device camera (using Expo Camera).
    Processing:
        Extract key points/features from the camera feed (likely using a library like MediaPipe within React Native, or potentially sending frames to a backend).
        Feed these features into the locally running TFLite sign language translation model.
    Output:
        The TFLite model outputs the translated text.
        A Text-to-Speech module speaks the translated text.

The process to build the model is:

    Data Preparation (How2Sign): This involves getting the How2Sign dataset, extracting features (e.g., using MediaPipe via the scripts in examples/sign_language/scripts/), creating TSV manifests, and finally using fairseq-preprocess.
    Training: Using fairseq-train (or fairseq-hydra-train) with the preprocessed data and a suitable config file.
    Conversion: Adapting and running convert_to_tflite.py on the trained model.


## Building the Model on Google Colab

Here's how you can set up your Google Colab notebook to build the model, assuming you've already followed the Setup steps in the README.md (cloning the repo, installing dependencies):

1. Download How2Sign Dataset (User Action):

    You need to obtain the How2Sign dataset (videos and transcriptions) yourself. The scripts expect a certain directory structure, so refer to examples/sign_language/README.md for details on where to place the downloaded files.
    Upload the dataset to your Google Drive and mount it in Colab, or download it directly within the Colab environment using wget or gdown if you have direct URLs. Let's assume you place it in a directory accessible at /content/how2sign-data.

2. Extract MediaPipe Features:

    Navigate to the scripts directory:
```
    cd examples/sign_language/scripts
```
Run the MediaPipe extraction script. This processes the videos to extract pose and hand keypoints. Note: This can be very time-consuming, especially on a large dataset like How2Sign. You might need a persistent Colab session or consider running this part locally if Colab times out.
```
# --- In Colab Cell ---
# Adjust input/output paths as needed
!python extract_mediapipe.py --video-dir /content/how2sign-data/videos --output-dir /content/how2sign-data/mediapipe_output --processes 4
```
Convert the extracted JSON files to NumPy format:
```
# --- In Colab Cell ---
# Adjust paths as needed
!python mediapipe_json2npy.py --json-dir /content/how2sign-data/mediapipe_output --output-dir /content/how2sign-data/mediapipe_npy
```
Go back to the project root:
```
cd ../../..
```
3. Generate TSV Manifests:

    Use the script to create the .tsv files that Fairseq needs, linking the extracted features (NumPy files) to the transcriptions.
```
    # --- In Colab Cell ---
    # Adjust paths based on your How2Sign download structure and output from previous step
    !python examples/sign_language/scripts/generate_tsv.py 
        --feature-dir /content/how2sign-data/mediapipe_npy 
        --video-ids-path /content/how2sign-data/splits/train.txt 
        --transcription-path /content/how2sign-data/transcriptions/train.en 
        --output-path /content/how2sign-data/processed/train.tsv 
        --feature-type mediapipe # Make sure feature type matches extraction
    # Repeat for validation (dev) and test splits
    !python examples/sign_language/scripts/generate_tsv.py 
        --feature-dir /content/how2sign-data/mediapipe_npy 
        --video-ids-path /content/how2sign-data/splits/dev.txt 
        --transcription-path /content/how2sign-data/transcriptions/dev.en 
        --output-path /content/how2sign-data/processed/dev.tsv 
        --feature-type mediapipe

    !python examples/sign_language/scripts/generate_tsv.py 
        --feature-dir /content/how2sign-data/mediapipe_npy 
        --video-ids-path /content/how2sign-data/splits/test.txt 
        --transcription-path /content/how2sign-data/transcriptions/test.en 
        --output-path /content/how2sign-data/processed/test.tsv 
        --feature-type mediapipe
```

4. Train SentencePiece Model:

    Train a tokenizer on your target language data (English transcriptions).
```
    # --- In Colab Cell ---
    # Combine all transcriptions (or just use training split)
    !cat /content/how2sign-data/transcriptions/*.en > /content/how2sign-data/processed/all_transcriptions.en

    !python examples/sign_language/scripts/train_spm.py 
        --input /content/how2sign-data/processed/all_transcriptions.en 
        --model-prefix /content/how2sign-data/processed/spm_en 
        --vocab-size 4000 # Adjust vocab size as needed
```
5. Fairseq Preprocessing (Binarization):

    Prepare the data in Fairseq's binary format. You'll need a feature configuration YAML. Let's assume there's one like examples/sign_language/config/mediapipe_feature_config.yaml (you might need to create or adapt one based on your exact MediaPipe features).
```
    # --- In Colab Cell ---
    # Create destination directory
    !mkdir -p data-bin/how2sign_mediapipe

    # Ensure the feature config YAML exists and is correct
    # You might need to copy/adapt one from examples/sign_language/config/
    # Let's assume it's at: examples/sign_language/config/mediapipe_feature_config.yaml

    !fairseq-preprocess --task sign_to_text 
        --source-lang sign --target-lang en 
        --trainpref /content/how2sign-data/processed/train.tsv 
        --validpref /content/how2sign-data/processed/dev.tsv 
        --testpref /content/how2sign-data/processed/test.tsv 
        --destdir data-bin/how2sign_mediapipe 
        --config examples/sign_language/config/mediapipe_feature_config.yaml 
        --srcdict examples/sign_language/datasets/dummy_dict.txt 
        --tgtdict /content/how2sign-data/processed/spm_en.vocab 
        --workers 2 # Adjust based on Colab resources
```
    (Note: Using a dummy source dictionary is common for non-text source modalities like sign features)

6. Training:

    Select a training configuration file suited for MediaPipe features, e.g., examples/sign_language/config/wmt-slt/srf_4k.yaml (you might need to adjust paths or parameters within it).
```
    # --- In Colab Cell ---
    # Option 1: Using fairseq-train (might need manual parameter setting)
    # !fairseq-train data-bin/how2sign_mediapipe 
    #    --config-yaml examples/sign_language/config/mediapipe_feature_config.yaml 
    #    --user-dir fairseq/models/sign_to_text 
    #    --task sign_to_text 
    #    --arch sign2text_transformer 
    #    --save-dir checkpoints/how2sign_mediapipe 
    #    --optimizer adam --lr 0.0005 ... (Add other params from a config file)

    # Option 2: Using fairseq-hydra-train (Recommended - uses full config file)
    # Ensure the chosen config file (e.g., srf_4k.yaml) has correct paths/params
    !fairseq-hydra-train 
        task.data=data-bin/how2sign_mediapipe 
        task.sentencepiece_model=/content/how2sign-data/processed/spm_en.model 
        hydra.run.dir=checkpoints/how2sign_mediapipe 
        distributed_training.distributed_world_size=1 `# Use 1 for single Colab GPU` 
        --config-dir examples/sign_language/config/wmt-slt 
        --config-name srf_4k # Or another relevant config
```
7. TFLite Conversion (Adaptation Required):

    After training finishes and you have a checkpoint file (e.g., checkpoints/how2sign_mediapipe/checkpoint_best.pt), adapt convert_to_tflite.py.
        Key changes: Load the sign2text_transformer model, use your checkpoint, define dummy inputs matching the shape/type of your processed MediaPipe features and SentencePiece outputs.
```
    # --- In Colab Cell ---
    # Make necessary modifications to convert_to_tflite.py first!
    !python convert_to_tflite.py 
        --checkpoint checkpoints/how2sign_mediapipe/checkpoint_best.pt 
        --output-dir tflite_models/how2sign_mediapipe
```
This detailed workflow should prepare you to run the model building process on Google Colab. Remember to adjust paths according to where you store the data and outputs. The most critical parts are ensuring the How2Sign data is correctly downloaded/placed and that the configuration files (.yaml) match your data and desired model architecture. Good luck!

## 1. Setup (Google Colab Recommended)

### Environment Setup

1.  **Clone the Repository:**
    ```bash
    git clone <your-repository-url>
    cd <repository-directory>
    ```
    *(Or upload the project folder directly to your Colab environment)*

2.  **Select GPU Runtime:**
    *   In Google Colab, go to `Runtime` -> `Change runtime type` and select `GPU` as the hardware accelerator.

3.  **Install Dependencies:**
    *   Install PyTorch matching your Colab's CUDA version (check `!nvcc --version`). See [PyTorch installation instructions](https://pytorch.org/get-started/locally/). Example:
        ```bash
        # Example for CUDA 11.8 (check your Colab version!)
        !pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
        ```
    *   Install Fairseq and other required packages:
        ```bash
        !pip install --upgrade pip setuptools  # Ensure pip and setuptools are up-to-date
        !pip install fairseq sentencepiece # Core Fairseq
        !pip install tensorflow onnx onnx-tf # For TFLite conversion
        !pip install mediapipe iopath av # Dependencies for sign language examples
        # Install other potential dependencies based on errors or specific feature needs
        # !pip install editdistance sacrebleu tensorboardX hydra-core omegaconf
        ```
    *   **(Optional) Build Fairseq from source if needed (for C++ extensions):**
        ```bash
        # !python setup.py build_ext --inplace
        ```

### Verify Installation

```python
import fairseq
import torch
import tensorflow as tf
import onnx
import onnx_tf
import mediapipe
import iopath
import av

print("Fairseq version:", fairseq.__version__)
print("PyTorch version:", torch.__version__)
print("TensorFlow version:", tf.__version__)
print("Torch CUDA available:", torch.cuda.is_available())
```

## 2. Data Preparation (How2Sign)

Detailed instructions for downloading and preparing the How2Sign dataset are provided within the sign language example directory:

*   **Main Guide:** `examples/sign_language/README.md`
*   **Scripts:** `examples/sign_language/scripts/`

**General Steps:**

1.  **Download:** Obtain the How2Sign dataset videos and annotations.
2.  **Feature Extraction:** Extract relevant features (e.g., pose using MediaPipe, I3D features) using scripts like `extract_mediapipe.py`, `i3d_formatting.py`, `mediapipe_json2npy.py`.
3.  **Create TSV Manifests:** Generate Tab-Separated Value (TSV) files mapping features to text transcriptions using `generate_tsv.py`.
4.  **SentencePiece Model:** Train a SentencePiece model on the target text data using `train_spm.py`.
5.  **Fairseq Preprocessing:** Binarize the data for Fairseq training using `fairseq-preprocess`.

Refer to `examples/sign_language/README.md` for specific commands and configurations for these steps. An example `fairseq-preprocess` command might look like this (adapt paths and filenames based on your setup):

```bash
# Example - Adapt based on examples/sign_language/README.md
fairseq-preprocess --task sign_to_text \
    --source-lang sign --target-lang en \
    --trainpref path/to/your/train.tsv \
    --validpref path/to/your/dev.tsv \
    --testpref path/to/your/test.tsv \
    --destdir data-bin/how2sign_features \
    --config config.yaml # Your feature config
    --workers 20
```

## 3. Training

Training commands and configurations are provided in:

*   **Guide & Scripts:** `examples/sign_language/README.md`, `examples/sign_language/train.sh`
*   **Hydra Configs:** `examples/sign_language/config/`

Use `fairseq-train` or `fairseq-hydra-train` to start training.

**Example using `fairseq-train` (adapt based on configs):**

```bash
# Example - Adapt based on examples/sign_language/train.sh and configs
fairseq-train data-bin/how2sign_features \
    --config-yaml config.yaml # Feature config \
    --user-dir fairseq/models/sign_to_text \
    --task sign_to_text \
    --arch sign2text_transformer # Or other model from configs \
    --save-dir checkpoints/sign2text_how2sign \
    # Add other training parameters (optimizer, LR, batch size, etc.) from configs
    # e.g., --optimizer adam --lr 0.0005 --max-tokens 4096 ...
    --log-interval 100 \
    --max-update 200000 \
    --patience 10
```

**Example using `fairseq-hydra-train` (adapt based on configs):**

```bash
# Example - Adapt based on examples/sign_language/train.sh and configs
fairseq-hydra-train \
    task.data=data-bin/how2sign_features \
    distributed_training.distributed_world_size=1 \
    --config-dir examples/sign_language/config \
    --config-name <your_config_file_name> # e.g., fn_4k.yaml
```

Monitor training progress using logs or TensorBoard if configured.

## 4. Generation / Inference

Generation/inference commands are provided in:

*   **Guide & Scripts:** `examples/sign_language/README.md`, `examples/sign_language/generate.sh`

Use `fairseq-generate` or `fairseq-interactive`.

**Example using `fairseq-generate` (adapt based on configs):**

```bash
# Example - Adapt based on examples/sign_language/generate.sh and configs
fairseq-generate data-bin/how2sign_features \
    --config-yaml config.yaml # Feature config \
    --user-dir fairseq/models/sign_to_text \
    --task sign_to_text \
    --path checkpoints/sign2text_how2sign/checkpoint_best.pt \
    --max-tokens 50000 \
    --beam 5 \
    --scoring sacrebleu \
    --results-path results/sign2text_how2sign
```

## 5. TensorFlow Lite (TFLite) Conversion for React Native

The goal is to convert the trained PyTorch model into a TFLite format suitable for deployment in mobile applications, such as those built with React Native.

**Script:** `convert_to_tflite.py`

This script provides a basic framework for the conversion process, which typically involves:

1.  **PyTorch -> ONNX:** Exporting the Fairseq PyTorch model (`Sign2TextTransformerModel` or similar) to the ONNX (Open Neural Network Exchange) format.
2.  **ONNX -> TensorFlow:** Converting the ONNX model to a TensorFlow SavedModel format.
3.  **TensorFlow -> TFLite:** Converting the TensorFlow model to the TensorFlow Lite format. This step often includes quantization (e.g., FP16 or INT8) to reduce model size and potentially improve inference speed on mobile devices.

**Important Considerations:**

*   **Script Adaptation:** The provided `convert_to_tflite.py` is **generic**. You will likely **need to modify it significantly** to work with the specific sign language model architecture (`Sign2TextTransformerModel`) trained on How2Sign data. This includes:
    *   Correctly loading your trained Fairseq model and checkpoint.
    *   Providing the correct dummy input tensors (`dummy_src_tokens`, `dummy_prev_output_tokens`, etc.) that match the expected input shapes and types of your model. This might involve understanding how features (like pose data) are preprocessed and fed into the model.
    *   Ensuring all model operations (layers, activations) are supported by the ONNX and TFLite conversion processes. Custom or complex operations might require workarounds or model simplification.
*   **Quantization:** Quantization can significantly reduce model size but might impact accuracy. Post-training quantization (as shown in the script) is common, but may require a representative dataset for calibration, especially for INT8 quantization, to maintain performance. Evaluate the quantized model carefully.
*   **React Native Integration:** Once you have a `.tflite` file, integrating it into React Native requires using a suitable library (e.g., `react-native-tensorflow-lite` or similar wrappers) to load the model and run inference within your mobile app. Pre- and post-processing steps (matching those used during training/conversion) will also need to be implemented in your React Native application.

**Steps:**

1.  **Adapt `convert_to_tflite.py`:** Modify the script to load your specific model checkpoint and define correct dummy inputs.
2.  **Run Conversion:** Execute the adapted script. Debug any errors related to unsupported operations or shape mismatches.
    ```bash
    python convert_to_tflite.py --checkpoint path/to/your/checkpoint_best.pt --output-dir tflite_models
    ```
3.  **Test TFLite Model:** Verify the generated TFLite model's predictions against the original PyTorch model using sample data.
4.  **Integrate into React Native:** Use a TFLite library in React Native to load and run the model.
