# Fairseq Sign Language Translation (How2Sign Focus for React Native App)

This project utilizes the Fairseq toolkit to train sign language translation models, specifically targeting the How2Sign dataset. The ultimate objective is to develop a React Native Expo application that performs live sign language translation using a TFLite model generated from this workflow.

This README provides a comprehensive guide covering environment setup (optimized for Google Colab), downloading and preparing the How2Sign dataset, training the translation model, evaluating its performance, and converting the final model to TensorFlow Lite (TFLite) for mobile deployment.

## Project Goal: End-to-End React Native Sign Language Translator

The final application aims to achieve the following pipeline within a React Native Expo app:

1.  **Input:** The user performs sign language in front of the device's camera (captured using Expo Camera).
2.  **Feature Extraction:** Key points (pose, hands, face) are extracted from the live camera feed in real-time. This will likely be done directly within the React Native app using libraries like MediaPipe's JavaScript/WASM solutions, or potentially offloaded if necessary.
3.  **Translation:** The extracted features are fed into the locally deployed TFLite sign language translation model.
4.  **Output:**
    *   The TFLite model outputs the corresponding translated text.
    *   A Text-to-Speech (TTS) module within the app voices the translation.

## Model Building Workflow Overview

The process to create the TFLite model involves these key stages:

1.  **Setup:** Configure the environment (Python, PyTorch, Fairseq, TFLite dependencies), preferably on Google Colab with GPU acceleration.
2.  **Data Acquisition (How2Sign):** Download the necessary How2Sign dataset components (videos, text) using the provided script.
3.  **Data Preparation:**
    *   Extract features from videos (e.g., MediaPipe keypoints).
    *   Generate TSV manifest files linking features to text.
    *   Train a SentencePiece model for text tokenization.
    *   Preprocess/binarize the data for Fairseq training.
4.  **Training:** Train the sign-to-text model using `fairseq-train` or `fairseq-hydra-train`.
5.  **Evaluation:** Assess model performance using `fairseq-generate`.
6.  **TFLite Conversion:** Adapt and run `convert_to_tflite.py` to convert the trained PyTorch model to the TFLite format.

---

## 1. Setup (Google Colab Recommended)

Using Google Colab provides free access to GPUs, which significantly speeds up training.

### Environment Setup

1.  **Clone the Repository (if applicable):**
    *   If you have this project on GitHub:
        ```bash
        # --- In Colab Cell ---
        !git clone <your-repository-url>
        %cd <repository-directory>
        ```
    *   Alternatively, upload the project folder directly to your Colab environment and navigate into it using the file browser and `%cd`.

2.  **Select GPU Runtime:**
    *   In Google Colab: `Runtime` -> `Change runtime type` -> Select `GPU`.

3.  **Install Dependencies:**
    *   **PyTorch:** Install the version matching Colab's CUDA. Check CUDA version first:
        ```bash
        # --- In Colab Cell ---
        !nvcc --version
        ```
        Then install PyTorch (example for CUDA 11.8, adjust if needed):
        ```bash
        # --- In Colab Cell ---
        # Find the correct command for your CUDA version at: https://pytorch.org/get-started/locally/
        !pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
        ```
    *   **Fairseq & Other Libraries:**
        ```bash
        # --- In Colab Cell ---
        !pip install --upgrade pip setuptools # Ensure pip/setuptools are up-to-date
        !pip install fairseq sentencepiece # Core Fairseq & tokenizer
        !pip install tensorflow onnx onnx-tf # For TFLite conversion
        !pip install mediapipe iopath av # Dependencies for sign language features/data handling
        # Optional but often useful dependencies
        !pip install editdistance sacrebleu tensorboardX hydra-core omegaconf pandas tqdm
        ```
    *   **(Optional) Build Fairseq from source:** Sometimes needed for C++ extensions if pip install has issues.
        ```bash
        # --- In Colab Cell ---
        # !python setup.py build_ext --inplace
        ```

4.  **(Optional but Recommended) Mount Google Drive:** To persist downloaded data, checkpoints, and outputs across Colab sessions.
    ```python
    # --- In Colab Cell ---
    from google.colab import drive
    drive.mount('/content/drive')
    # Create a base directory for the project on your Drive
    %mkdir -p /content/drive/MyDrive/fairseq_slt_how2sign
    %cd /content/drive/MyDrive/fairseq_slt_how2sign
    # If you cloned the repo elsewhere, move/copy it here now.
    ```
    *Note: Adjust subsequent paths in this README if using Google Drive.*

### Verify Installation

Run this cell to ensure key libraries are installed and accessible.

```python
# --- In Colab Cell ---
import sys
import fairseq
import torch
import tensorflow as tf
import onnx
import onnx_tf
import mediapipe
import iopath
import av
import sentencepiece
import pandas

print("--- Versions ---")
print("Python:", sys.version)
print("Fairseq:", fairseq.__version__)
print("PyTorch:", torch.__version__)
print("TensorFlow:", tf.__version__)
print("ONNX:", onnx.__version__)
print("ONNX-TF:", onnx_tf.__version__)
print("MediaPipe:", mediapipe.__version__)
print("SentencePiece:", sentencepiece.__version__)
print("Pandas:", pandas.__version__)

print("\n--- CUDA ---")
print("Torch CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA Device:", torch.cuda.get_device_name(0))

print("\nSetup complete!")
```

---

## 2. Data Acquisition (How2Sign)

We will use the official `download_how2sign.sh` script to download the required dataset components.

1.  **Save the Download Script:** Copy the content of the `how2sign.sh` script provided at the end of the original prompt into a file named `download_how2sign.sh` in your project directory (e.g., in your Colab environment or mounted Drive).

2.  **Make the Script Executable:**
    ```bash
    # --- In Colab Cell ---
    !chmod +x download_how2sign.sh
    ```

3.  **Run the Download Script:** Execute the script, specifying the modalities you need. For this project, you'll likely need the **video data** (choose *one* of the video types, e.g., `rgb_front_clips` which are pre-segmented) and the **re-aligned English translations**.
    *   Downloading **clips** (`rgb_front_clips`) is generally recommended over full videos if available, as they are already segmented by sentence.
    *   Using **re-aligned** text (`english_translation_re-aligned`) is often better as the alignment between signs and text is improved.

    ```bash
    # --- In Colab Cell ---
    # Example: Download front-view clips and re-aligned English text
    # This will create a ./How2Sign directory structure
    # This can take a VERY long time and requires significant disk space! Run this in a persistent environment (like mounted Drive).
    !./download_how2sign.sh rgb_front_clips english_translation_re-aligned

    # --- Alternatively, if you prefer full videos (requires more processing later): ---
    # !./download_how2sign.sh rgb_front_videos english_translation_re-aligned
    ```

4.  **Expected Directory Structure:** After running the script, you should have a structure like this (relative to where you ran the script):
    ```
    How2Sign/
    ├── sentence_level/
    │   ├── train/
    │   │   ├── rgb_front/          # Or rgb_front_videos if you chose that
    │   │   │   └── ... (video clip files .mp4)
    │   │   └── text/
    │   │       └── en/
    │   │           └── raw_text/
    │   │               └── re_aligned/
    │   │                   └── how2sign_realigned_train.csv
    │   ├── val/
    │   │   ├── rgb_front/
    │   │   └── text/
    │   │       └── en/
    │   │           └── raw_text/
    │   │               └── re_aligned/
    │   │                   └── how2sign_realigned_val.csv
    │   └── test/
    │       ├── rgb_front/
    │       └── text/
    │           └── en/
    │               └── raw_text/
    │                   └── re_aligned/
    │                       └── how2sign_realigned_test.csv
    # (Other directories like video_level might exist if you downloaded full videos)
    ```
    *Confirm this structure exists before proceeding.*

---

## 3. Data Preparation

This stage involves extracting features, creating manifests, tokenizing text, and preparing data for Fairseq. We'll assume you downloaded `rgb_front_clips` and `english_translation_re-aligned` into the `./How2Sign` directory.

**Create Processing Directories:**
```bash
# --- In Colab Cell ---
# Directory for extracted features
!mkdir -p ./How2Sign/features/mediapipe_npy
# Directory for processed manifests and SPM model
!mkdir -p ./How2Sign/processed
# Directory for Fairseq binary data
!mkdir -p ./data-bin/how2sign_mediapipe
```

**1. Extract MediaPipe Features:**
Process the downloaded video clips (`.mp4` files) to extract pose and hand keypoints using MediaPipe.

*   Navigate to the scripts directory (assuming the fairseq repo structure):
    ```bash
    # --- In Colab Cell ---
    %cd examples/sign_language/scripts
    ```
*   Run extraction (this can be **extremely time-consuming** for the full dataset):
    ```bash
    # --- In Colab Cell ---
    # Adjust paths based on your download location and desired output
    # Use a high number of processes if Colab allows, but monitor memory usage
    !python extract_mediapipe.py \
        --video-dir ../../../How2Sign/sentence_level/train/rgb_front \
        --output-dir ../../../How2Sign/features/mediapipe_json_train \
        --processes 4 # Adjust based on Colab CPU cores

    !python extract_mediapipe.py \
        --video-dir ../../../How2Sign/sentence_level/val/rgb_front \
        --output-dir ../../../How2Sign/features/mediapipe_json_val \
        --processes 4

    !python extract_mediapipe.py \
        --video-dir ../../../How2Sign/sentence_level/test/rgb_front \
        --output-dir ../../../How2Sign/features/mediapipe_json_test \
        --processes 4
    ```
*   Convert the extracted JSON files to NumPy format (`.npy`):
    ```bash
    # --- In Colab Cell ---
    # Adjust paths as needed
    !python mediapipe_json2npy.py \
        --json-dir ../../../How2Sign/features/mediapipe_json_train \
        --output-dir ../../../How2Sign/features/mediapipe_npy \
        --split train

    !python mediapipe_json2npy.py \
        --json-dir ../../../How2Sign/features/mediapipe_json_val \
        --output-dir ../../../How2Sign/features/mediapipe_npy \
        --split val

    !python mediapipe_json2npy.py \
        --json-dir ../../../How2Sign/features/mediapipe_json_test \
        --output-dir ../../../How2Sign/features/mediapipe_npy \
        --split test
    ```
*   Go back to the project root directory:
    ```bash
    # --- In Colab Cell ---
    %cd ../../..
    ```

**2. Generate TSV Manifests:**
Create `.tsv` files linking the extracted MediaPipe features (`.npy`) to the corresponding English transcriptions from the `.csv` files.

*   First, extract the raw text from the CSVs:
    ```bash
    # --- In Colab Cell ---
    # Ensure pandas is installed: !pip install pandas
    import pandas as pd

    def extract_text(csv_path, output_txt_path, text_column='TEXT'):
        df = pd.read_csv(csv_path, sep=',') # Adjust sep if needed
        # Ensure text is string and handle potential NaN
        texts = df[text_column].fillna('').astype(str).tolist()
        with open(output_txt_path, 'w', encoding='utf-8') as f:
            for text in texts:
                f.write(text + '\n')
        print(f"Extracted text to {output_txt_path}")

    # Define paths (adjust if your structure differs)
    base_path = './How2Sign/sentence_level/'
    text_base_path = base_path + '{split}/text/en/raw_text/re_aligned/how2sign_realigned_{split}.csv'
    output_base_path = './How2Sign/processed/{split}.en' # Output directory created earlier

    # Process train, val, test
    for split in ['train', 'val', 'test']:
        csv_file = text_base_path.format(split=split)
        txt_file = output_base_path.format(split=split)
        extract_text(csv_file, txt_file)

    # Also create video ID lists (needed for generate_tsv.py)
    # Assuming video filenames are the IDs (e.g., VIDEO_ID.mp4)
    import os
    def create_id_list(video_dir, output_id_path):
        ids = sorted([os.path.splitext(f)[0] for f in os.listdir(video_dir) if f.endswith('.mp4')])
        with open(output_id_path, 'w') as f:
            for vid_id in ids:
                f.write(vid_id + '\n')
        print(f"Created ID list at {output_id_path}")

    id_output_base_path = './How2Sign/processed/{split}.ids'
    video_base_path = base_path + '{split}/rgb_front/'

    for split in ['train', 'val', 'test']:
        video_dir = video_base_path.format(split=split)
        id_file = id_output_base_path.format(split=split)
        create_id_list(video_dir, id_file)
    ```
*   Now generate the TSV files:
    ```bash
    # --- In Colab Cell ---
    # Adjust paths based on your structure and feature output
    !python examples/sign_language/scripts/generate_tsv.py \
        --feature-dir ./How2Sign/features/mediapipe_npy \
        --video-ids-path ./How2Sign/processed/train.ids \
        --transcription-path ./How2Sign/processed/train.en \
        --output-path ./How2Sign/processed/train.tsv \
        --feature-type mediapipe # Important: matches extraction type

    !python examples/sign_language/scripts/generate_tsv.py \
        --feature-dir ./How2Sign/features/mediapipe_npy \
        --video-ids-path ./How2Sign/processed/val.ids \
        --transcription-path ./How2Sign/processed/val.en \
        --output-path ./How2Sign/processed/dev.tsv \
        --feature-type mediapipe

    !python examples/sign_language/scripts/generate_tsv.py \
        --feature-dir ./How2Sign/features/mediapipe_npy \
        --video-ids-path ./How2Sign/processed/test.ids \
        --transcription-path ./How2Sign/processed/test.en \
        --output-path ./How2Sign/processed/test.tsv \
        --feature-type mediapipe
    ```

**3. Train SentencePiece Model:**
Train a subword tokenizer on the target language (English) transcriptions.

```bash
# --- In Colab Cell ---
# Combine transcriptions for training SPM (using only train is also common)
!cat ./How2Sign/processed/train.en ./How2Sign/processed/dev.en > ./How2Sign/processed/all_transcriptions.en

# Train the SPM model
!python examples/sign_language/scripts/train_spm.py \
    --input ./How2Sign/processed/all_transcriptions.en \
    --model-prefix ./How2Sign/processed/spm_en_bpe4000 \
    --vocab-size 4000 \
    --character-coverage 1.0 \
    --model-type bpe # BPE is common for translation tasks
```
This will create `spm_en_bpe4000.model` and `spm_en_bpe4000.vocab`.

**4. Fairseq Preprocessing (Binarization):**
Convert the TSV manifests and text data into Fairseq's binary format for efficient training.

*   **Feature Configuration YAML:** You need a YAML file defining how features are processed. Create one, e.g., `mediapipe_config.yaml`, based on examples in `examples/sign_language/config/`. A minimal example for MediaPipe might look like:

    ```yaml
    # Create this file: mediapipe_config.yaml
    modality: MEDIAPIPE # Should match --feature-type in generate_tsv
    process_steps:
      # Example: Normalize pose (adjust based on your actual features/needs)
      # - type: normalize
      #   mean_path: path/to/mean.npy # Optional: pre-calculated mean
      #   std_path: path/to/std.npy   # Optional: pre-calculated std
      # Or just use instance normalization if mean/std aren't pre-calculated
      - type: instance_normalize
      # Add other steps like subsampling if desired
      # - type: subsample
      #   rate: 2 # Keep every 2nd frame
    ```
    *Note: You might need to adapt `examples/sign_language/config/mediapipe_feature_config.yaml` or create your own based on the exact structure of your `.npy` files.*

*   **Dummy Source Dictionary:** Sign language features don't have a "source dictionary" like text. Create a dummy one:
    ```bash
    # --- In Colab Cell ---
    !echo "<UNUSED> 0" > dummy_dict.txt
    !echo "<PAD> 1" >> dummy_dict.txt
    !echo "<EOS> 2" >> dummy_dict.txt
    !echo "<UNK> 3" >> dummy_dict.txt
    ```

*   Run `fairseq-preprocess`:
    ```bash
    # --- In Colab Cell ---
    # Ensure the feature config YAML (mediapipe_config.yaml) exists and is correct
    # Ensure the SPM vocab (spm_en_bpe4000.vocab) exists

    !fairseq-preprocess --task sign_to_text \
        --source-lang sign --target-lang en \
        --trainpref ./How2Sign/processed/train.tsv \
        --validpref ./How2Sign/processed/dev.tsv \
        --testpref ./How2Sign/processed/test.tsv \
        --destdir ./data-bin/how2sign_mediapipe \
        --config mediapipe_config.yaml \
        --srcdict dummy_dict.txt \
        --tgtdict ./How2Sign/processed/spm_en_bpe4000.vocab \
        --workers 2 # Adjust based on Colab resources
    ```

Data is now prepared in `./data-bin/how2sign_mediapipe`.

---

## 4. Training

Train the sign-to-text translation model using the preprocessed data. Using `fairseq-hydra-train` is recommended as it handles configuration more robustly via YAML files.

*   **Choose a Training Configuration:** Select or adapt a configuration file from `examples/sign_language/config/wmt-slt/`. For instance, `srf_4k.yaml` might be a starting point, but you **must edit it** to:
    *   Point `task.data` to your binarized data path (`./data-bin/how2sign_mediapipe`).
    *   Point `task.sentencepiece_model` to your trained SPM model (`./How2Sign/processed/spm_en_bpe4000.model`).
    *   Adjust `model` parameters (e.g., `encoder_embed_dim`, `decoder_embed_dim`) based on your feature dimensions and desired model size.
    *   Set `distributed_training.distributed_world_size=1` for single-GPU Colab training.
    *   Adjust batch sizes (`dataset.max_tokens`, `dataset.batch_size`) based on GPU memory.
    *   Set `checkpoint.save_dir` to where you want checkpoints saved (e.g., on Google Drive).

*   **Run Training:**
    ```bash
    # --- In Colab Cell ---
    # Make sure you have edited the config file (e.g., srf_4k.yaml) appropriately!
    # Example assuming the config is saved as 'my_how2sign_config.yaml' in the config dir

    !fairseq-hydra-train \
        --config-dir examples/sign_language/config/wmt-slt \
        --config-name srf_4k # Or your adapted config file name \
        task.data=./data-bin/how2sign_mediapipe \
        task.sentencepiece_model=./How2Sign/processed/spm_en_bpe4000.model \
        hydra.run.dir=./checkpoints/how2sign_mediapipe_srf4k \
        checkpoint.save_dir=./checkpoints/how2sign_mediapipe_srf4k \
        distributed_training.distributed_world_size=1 \
        dataset.num_workers=2 # Adjust based on Colab
        # Add overrides for specific params if needed, e.g.:
        # dataset.max_tokens=2048 model.encoder_embed_dim=256 ...
    ```

Training will save checkpoints (e.g., `checkpoint_best.pt`, `checkpoint_last.pt`) in the specified directory (`./checkpoints/how2sign_mediapipe_srf4k`). Monitor progress via the console output or TensorBoard logs if configured.

---

## 5. Evaluation / Inference

Generate translations for the test set to evaluate the trained model using BLEU or other metrics.

```bash
# --- In Colab Cell ---
# Adapt paths and config name as needed
CHECKPOINT_PATH="./checkpoints/how2sign_mediapipe_srf4k/checkpoint_best.pt"
CONFIG_DIR="examples/sign_language/config/wmt-slt"
CONFIG_NAME="srf_4k" # Use the same config name as training
RESULTS_PATH="./results/how2sign_mediapipe_srf4k_test"

!fairseq-generate ./data-bin/how2sign_mediapipe \
    --task sign_to_text \
    --source-lang sign --target-lang en \
    --gen-subset test \
    --path $CHECKPOINT_PATH \
    --config-dir $CONFIG_DIR \
    --config-name $CONFIG_NAME \
    --sentencepiece-model ./How2Sign/processed/spm_en_bpe4000.model \
    --scoring sacrebleu \
    --beam 5 \
    --max-tokens 30000 `# Adjust based on GPU memory` \
    --results-path $RESULTS_PATH
    # Add --user-dir if your model needs custom code from fairseq/models/sign_to_text

# The BLEU score will be printed, and translations saved in $RESULTS_PATH
```

---

## 6. TensorFlow Lite (TFLite) Conversion

This is the crucial step for deploying the model in the React Native app. The provided `convert_to_tflite.py` script needs significant adaptation for sign language models.

**Key Challenges & Adaptations:**

1.  **Model Loading:** Modify the script to load your specific Fairseq sign language model architecture (e.g., `Sign2TextTransformerModel` from `fairseq.models.sign_to_text`) and your trained checkpoint (`checkpoint_best.pt`).
2.  **Dummy Inputs:** This is critical. You need to create dummy input tensors that precisely match the **shape and data type** expected by your model's `forward` method *after* preprocessing. This includes:
    *   `src_tokens`: The processed feature tensor (e.g., from MediaPipe `.npy` files, potentially normalized and subsampled as defined in your `mediapipe_config.yaml`). The shape will likely be `(batch_size, sequence_length, feature_dimension)`.
    *   `src_lengths`: A tensor indicating the actual length of each sequence in the batch, shape `(batch_size,)`.
    *   `prev_output_tokens`: The input to the decoder (usually starts with an EOS token), shape `(batch_size, target_sequence_length)`.
3.  **ONNX Export:** The `torch.onnx.export` function needs the model, dummy inputs, and potentially `dynamic_axes` specified if your input/output sequences have variable lengths (which they almost certainly will).
4.  **Unsupported Operations:** Fairseq models, especially custom ones like sign language transformers, might use PyTorch operations not directly supported by ONNX or TFLite. You might encounter errors during conversion. Solutions involve:
    *   Simplifying the model architecture.
    *   Implementing custom ONNX/TFLite operators (advanced).
    *   Finding equivalent supported operations.
5.  **Quantization:** To reduce model size for mobile, apply quantization (e.g., FP16 or INT8) during the TFLite conversion step. Post-training quantization is common, but INT8 often requires a representative dataset for calibration to minimize accuracy loss.

**Running the Adapted Script (Conceptual):**

```bash
# --- In Colab Cell ---
# 1. SIGNIFICANTLY MODIFY convert_to_tflite.py FIRST!
#    - Update model loading logic for Sign2TextTransformerModel
#    - Define correct dummy_src_tokens, src_lengths, prev_output_tokens
#    - Add dynamic_axes to torch.onnx.export for sequence dimensions
#    - Handle potential unsupported ops

# 2. Run the adapted script
!python convert_to_tflite.py \
    --checkpoint ./checkpoints/how2sign_mediapipe_srf4k/checkpoint_best.pt \
    --output-dir ./tflite_models/how2sign_mediapipe_srf4k \
    # Add any other arguments your adapted script requires (e.g., path to config)
```

**Testing:** After conversion, rigorously test the `.tflite` model's output against the original PyTorch model using sample feature inputs.

---

## 7. React Native Integration (Next Steps)

Once you have a working and tested `.tflite` model:

1.  **Choose a TFLite Library:** Select a library for running TFLite models in React Native (e.g., `react-native-tensorflow-lite`, wrappers around TensorFlow.js with TFLite backend, etc.).
2.  **Implement Preprocessing:** Replicate the *exact* same feature preprocessing steps (MediaPipe extraction, normalization, subsampling from your `mediapipe_config.yaml`) within your React Native app using JavaScript/WASM libraries (like MediaPipe Tasks) before feeding data to the TFLite model.
3.  **Implement Postprocessing:** Decode the model's output tensor (likely token IDs) back into text using the SentencePiece model/vocabulary. You might need to port the SPM decoding logic or use a JavaScript SPM library.
4.  **Integrate Camera & UI:** Use Expo Camera to get video frames, process them, run inference, and display/speak the results.

This workflow provides a path from the How2Sign dataset to a TFLite model suitable for your React Native sign language translation app. Remember that the TFLite conversion and mobile integration steps often require significant debugging and adaptation. Good luck!