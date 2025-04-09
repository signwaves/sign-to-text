# Fairseq Sign Language Translation (How2Sign Focus for React Native App)

This project utilizes the Fairseq toolkit to train sign language translation models, specifically targeting the How2Sign dataset. The ultimate objective is to develop a React Native Expo application that performs live sign language translation using a TFLite model generated from this workflow.

This README provides a comprehensive guide covering environment setup (optimized for Google Colab with Google Drive), downloading and preparing the How2Sign dataset, training the translation model, evaluating its performance, and converting the final model to TensorFlow Lite (TFLite) for mobile deployment.

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

1.  **Setup:** Configure the environment (Python, PyTorch, Fairseq, TFLite dependencies) on Google Colab, integrating with Google Drive for persistence.
2.  **Data Acquisition (How2Sign):** Download the necessary How2Sign dataset components (videos, text) using the provided script into your Google Drive project folder.
3.  **Data Preparation:**
    *   Extract features from videos (e.g., MediaPipe keypoints).
    *   Generate TSV manifest files linking features to text.
    *   Train a SentencePiece model for text tokenization.
    *   Preprocess/binarize the data for Fairseq training.
4.  **Training:** Train the sign-to-text model using `fairseq-train` or `fairseq-hydra-train`, saving checkpoints to Google Drive.
5.  **Evaluation:** Assess model performance using `fairseq-generate`.
6.  **TFLite Conversion:** Adapt and run `convert_to_tflite.py` to convert the trained PyTorch model to the TFLite format.

---

## 1. Setup (Google Colab with Google Drive Recommended)

Using Google Colab provides free access to GPUs, and Google Drive ensures your data, code, and models persist across sessions.

### Environment Setup

1.  **Mount Google Drive:**
    ```python
    # --- In Colab Cell ---
    from google.colab import drive
    drive.mount('/content/drive')
    ```

2.  **Create Project Directory & Navigate:** Create a dedicated folder on your Google Drive for this project and make it the current working directory. **All subsequent relative paths in this README assume you are running commands from this directory.**
    ```bash
    # --- In Colab Cell ---
    # Adjust the path if you prefer a different location/name
    %mkdir -p /content/drive/MyDrive/sign_to_text_project
    %cd /content/drive/MyDrive/sign_to_text_project

    # Verify current directory
    !pwd
    ```

3.  **Clone Your Repository:** Clone your project repository *into* this Google Drive directory.
    ```bash
    # --- In Colab Cell ---
    # Replace <your-repository-url> with your actual repo URL
    !git clone <your-repository-url> .
    # Or if you cloned it outside, move/copy the contents here.
    # Make sure the 'examples', 'fairseq', etc. directories from the
    # aquaticcalf/sign-to-text repo are now inside your Drive project folder.
    ```

4.  **Select GPU Runtime:**
    *   In Google Colab: `Runtime` -> `Change runtime type` -> Select `GPU`.

5.  **Install Dependencies:**
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
    *   **(Optional) Build Fairseq from source:** Sometimes needed for C++ extensions if pip install has issues. Run this from your project root directory on Drive.
        ```bash
        # --- In Colab Cell ---
        # !python setup.py build_ext --inplace
        ```

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

print("\nSetup complete! Current directory should be your project folder on Google Drive.")
!pwd
```

---

## 2. Data Acquisition (How2Sign)

We will use the official `download_how2sign.sh` script to download the required dataset components directly into your Google Drive project folder.

1.  **Save the Download Script:** Ensure the `download_how2sign.sh` script (provided separately or from the original repo) exists in your current project directory (`/content/drive/MyDrive/sign_to_text_project/`).

2.  **Make the Script Executable:**
    ```bash
    # --- In Colab Cell ---
    # Run this from your project root on Drive
    !chmod +x download_how2sign.sh
    ```

3.  **Run the Download Script:** Execute the script from your project root. It will create a `How2Sign` subdirectory within your Drive project folder.
    *   Downloading **clips** (`rgb_front_clips`) is generally recommended.
    *   Using **re-aligned** text (`english_translation_re-aligned`) is often better.

    ```bash
    # --- In Colab Cell ---
    # Run this from your project root on Drive (e.g., /content/drive/MyDrive/sign_to_text_project/)
    # Example: Download front-view clips and re-aligned English text
    # This will create a ./How2Sign directory structure INSIDE your project folder
    # This can take a VERY long time and requires significant disk space!
    !./download_how2sign.sh rgb_front_clips english_translation_re-aligned

    # --- Alternatively, if you prefer full videos (requires more processing later): ---
    # !./download_how2sign.sh rgb_front_videos english_translation_re-aligned
    ```

4.  **Expected Directory Structure:** After running the script, you should have a structure like this within your project directory on Drive:
    ```
    ./  # Your project root, e.g., /content/drive/MyDrive/sign_to_text_project/
    ├── How2Sign/
    │   ├── sentence_level/
    │   │   ├── train/
    │   │   │   ├── rgb_front/          # Or rgb_front_videos if you chose that
    │   │   │   │   └── ... (video clip files .mp4)
    │   │   │   └── text/
    │   │   │       └── en/
    │   │   │           └── raw_text/
    │   │   │               └── re_aligned/
    │   │   │                   └── how2sign_realigned_train.csv
    │   │   ├── val/
    │   │   │   ├── rgb_front/
    │   │   │   └── text/ (...)
    │   │   └── test/
    │   │       ├── rgb_front/
    │   │       └── text/ (...)
    │   # (Other directories like video_level might exist)
    ├── examples/
    ├── fairseq/
    ├── download_how2sign.sh
    └── ... (other project files)
    ```
    *Confirm this structure exists before proceeding.*

---

## 3. Data Preparation

This stage involves extracting features, creating manifests, tokenizing text, and preparing data for Fairseq. All paths assume you are running commands from your project root directory on Google Drive (`/content/drive/MyDrive/sign_to_text_project/`).

**Create Processing Directories:**
```bash
# --- In Colab Cell ---
# Run from project root
# Directory for extracted features
!mkdir -p ./How2Sign/features/mediapipe_json_train
!mkdir -p ./How2Sign/features/mediapipe_json_val
!mkdir -p ./How2Sign/features/mediapipe_json_test
!mkdir -p ./How2Sign/features/mediapipe_npy
# Directory for processed manifests and SPM model
!mkdir -p ./How2Sign/processed
# Directory for Fairseq binary data
!mkdir -p ./data-bin/how2sign_mediapipe
```

**1. Extract MediaPipe Features:**
Process the downloaded video clips (`.mp4` files) to extract keypoints.

*   Navigate to the scripts directory, run extraction, and navigate back. Paths inside the command are relative to the script's location.
    ```bash
    # --- In Colab Cell ---
    %cd examples/sign_language/scripts

    # Run extraction (this can be extremely time-consuming)
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

    # Convert the extracted JSON files to NumPy format (.npy)
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

    # Go back to the project root directory
    %cd ../../..
    !pwd # Should be back at your project root on Drive
    ```

**2. Generate TSV Manifests:**
Create `.tsv` files linking features (`.npy`) to transcriptions (`.csv`).

*   First, extract the raw text from the CSVs (run from project root):
    ```python
    # --- In Colab Cell ---
    import pandas as pd
    import os

    # Make sure you are in the project root directory first!
    # %cd /content/drive/MyDrive/sign_to_text_project

    def extract_text(csv_path, output_txt_path, text_column='TEXT'):
        df = pd.read_csv(csv_path, sep=',') # Adjust sep if needed
        texts = df[text_column].fillna('').astype(str).tolist()
        with open(output_txt_path, 'w', encoding='utf-8') as f:
            for text in texts:
                f.write(text + '\n')
        print(f"Extracted text to {output_txt_path}")

    # Define paths relative to project root
    base_path = './How2Sign/sentence_level/'
    text_base_path = base_path + '{split}/text/en/raw_text/re_aligned/how2sign_realigned_{split}.csv'
    output_base_path = './How2Sign/processed/{split}.en' # Output directory created earlier

    # Process train, val, test
    for split in ['train', 'val', 'test']:
        csv_file = text_base_path.format(split=split)
        txt_file = output_base_path.format(split=split)
        extract_text(csv_file, txt_file)

    # Also create video ID lists
    def create_id_list(video_dir, output_id_path):
        # Ensure video_dir exists before listing
        if not os.path.isdir(video_dir):
            print(f"Warning: Video directory not found: {video_dir}")
            return
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
*   Now generate the TSV files (run from project root):
    ```bash
    # --- In Colab Cell ---
    # Ensure you are in the project root
    !pwd
    # Adjust paths based on your structure and feature output
    !python examples/sign_language/scripts/generate_tsv.py \
        --feature-dir ./How2Sign/features/mediapipe_npy \
        --video-ids-path ./How2Sign/processed/train.ids \
        --transcription-path ./How2Sign/processed/train.en \
        --output-path ./How2Sign/processed/train.tsv \
        --feature-type mediapipe

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
Train a subword tokenizer (run from project root).

```bash
# --- In Colab Cell ---
# Ensure you are in the project root
!pwd
# Combine transcriptions for training SPM
!cat ./How2Sign/processed/train.en ./How2Sign/processed/dev.en > ./How2Sign/processed/all_transcriptions.en

# Train the SPM model
!python examples/sign_language/scripts/train_spm.py \
    --input ./How2Sign/processed/all_transcriptions.en \
    --model-prefix ./How2Sign/processed/spm_en_bpe4000 \
    --vocab-size 4000 \
    --character-coverage 1.0 \
    --model-type bpe # BPE is common for translation tasks
```
This will create `spm_en_bpe4000.model` and `spm_en_bpe4000.vocab` in `./How2Sign/processed/`.

**4. Fairseq Preprocessing (Binarization):**
Convert data into Fairseq's binary format (run from project root).

*   **Feature Configuration YAML:** Create a file named `mediapipe_config.yaml` **in your project root directory** (`/content/drive/MyDrive/sign_to_text_project/`). Adapt its contents based on `examples/sign_language/config/` and your feature processing needs. A minimal example:

    ```yaml
    # Create this file: ./mediapipe_config.yaml (in project root)
    modality: MEDIAPIPE # Should match --feature-type in generate_tsv
    process_steps:
      - type: instance_normalize
      # Add other steps like subsampling if desired
      # - type: subsample
      #   rate: 2
    ```
    *Note: You might need normalization mean/std files depending on your chosen steps.*

*   **Dummy Source Dictionary:** Create a dummy dictionary in your project root.
    ```bash
    # --- In Colab Cell ---
    # Run from project root
    !echo "<UNUSED> 0" > ./dummy_dict.txt
    !echo "<PAD> 1" >> ./dummy_dict.txt
    !echo "<EOS> 2" >> ./dummy_dict.txt
    !echo "<UNK> 3" >> ./dummy_dict.txt
    ```

*   Run `fairseq-preprocess` (from project root):
    ```bash
    # --- In Colab Cell ---
    # Ensure you are in the project root
    !pwd
    # Ensure the feature config YAML (./mediapipe_config.yaml) exists and is correct
    # Ensure the SPM vocab (./How2Sign/processed/spm_en_bpe4000.vocab) exists

    !fairseq-preprocess --task sign_to_text \
        --source-lang sign --target-lang en \
        --trainpref ./How2Sign/processed/train.tsv \
        --validpref ./How2Sign/processed/dev.tsv \
        --testpref ./How2Sign/processed/test.tsv \
        --destdir ./data-bin/how2sign_mediapipe \
        --config ./mediapipe_config.yaml \
        --srcdict ./dummy_dict.txt \
        --tgtdict ./How2Sign/processed/spm_en_bpe4000.vocab \
        --workers 2 # Adjust based on Colab resources
    ```

Data is now prepared in `./data-bin/how2sign_mediapipe` within your Drive project folder.

---

## 4. Training

Train the model using the preprocessed data (run from project root). Using `fairseq-hydra-train` is recommended.

*   **Choose & Adapt Training Configuration:** Select a base config from `examples/sign_language/config/wmt-slt/` (e.g., `srf_4k.yaml`). **You must edit this file OR override parameters on the command line**. Key parameters to set/check:
    *   `task.data`: Must point to `./data-bin/how2sign_mediapipe` (relative to project root).
    *   `task.sentencepiece_model`: Must point to `./How2Sign/processed/spm_en_bpe4000.model`.
    *   `model`: Adjust dimensions based on features/size.
    *   `distributed_training.distributed_world_size=1` (for single GPU).
    *   `dataset.max_tokens`, `dataset.batch_size`: Adjust for GPU memory.
    *   `checkpoint.save_dir`: Set to a relative path like `./checkpoints/how2sign_mediapipe_srf4k` (will save to your Drive).
    *   `hydra.run.dir`: Set similarly, e.g., `./checkpoints/how2sign_mediapipe_srf4k/hydra_run`.

*   **Run Training (from project root):**
    ```bash
    # --- In Colab Cell ---
    # Ensure you are in the project root
    !pwd
    # Example assumes using srf_4k config and overriding key paths.
    # Alternatively, edit the srf_4k.yaml file directly.

    !fairseq-hydra-train \
        --config-dir examples/sign_language/config/wmt-slt \
        --config-name srf_4k \
        task.data=./data-bin/how2sign_mediapipe \
        task.sentencepiece_model=./How2Sign/processed/spm_en_bpe4000.model \
        checkpoint.save_dir=./checkpoints/how2sign_mediapipe_srf4k \
        hydra.run.dir=./checkpoints/how2sign_mediapipe_srf4k/hydra_run \
        distributed_training.distributed_world_size=1 \
        dataset.num_workers=2 `# Adjust based on Colab` \
        dataset.batch_size=16 `# Example: Adjust based on GPU RAM and max_tokens` \
        dataset.max_tokens=4096 `# Example: Adjust based on GPU RAM`
        # Add other overrides if needed: model.encoder_embed_dim=256 ...
    ```

Checkpoints (`checkpoint_best.pt`, `checkpoint_last.pt`) will be saved in `./checkpoints/how2sign_mediapipe_srf4k` on your Google Drive.

---

## 5. Evaluation / Inference

Generate translations for the test set (run from project root).

```bash
# --- In Colab Cell ---
# Ensure you are in the project root
!pwd
# Adapt paths and config name as needed
CHECKPOINT_PATH="./checkpoints/how2sign_mediapipe_srf4k/checkpoint_best.pt"
CONFIG_DIR="examples/sign_language/config/wmt-slt"
CONFIG_NAME="srf_4k" # Use the same config name as training
RESULTS_PATH="./results/how2sign_mediapipe_srf4k_test"
!mkdir -p ./results # Ensure results directory exists

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
    # Note: May need path overrides for data/spm model if not picked up from config
    # task.data=./data-bin/how2sign_mediapipe \
    # task.sentencepiece_model=./How2Sign/processed/spm_en_bpe4000.model

# The BLEU score will be printed, and translations saved in $RESULTS_PATH
```

---

## 6. TensorFlow Lite (TFLite) Conversion (Encoder Only)

This is a crucial step for deploying the model in the React Native app. The provided `convert_to_tflite.py` script focuses on converting **only the encoder** part of the trained Sign-to-Text Transformer model to TFLite.

**Why Export Only the Encoder?**

*   Exporting the full autoregressive decoder (which generates text token by token) directly to ONNX/TFLite is complex due to its dynamic, looping nature.
*   The standard practice for deploying transformer translation models is to:
    1.  Run the **encoder** once on the input features (sign language keypoints) using the TFLite model.
    2.  Implement the **decoder logic** (the loop that generates text tokens using the encoder's output) separately in the application code (e.g., JavaScript in React Native).

The `convert_to_tflite.py` script handles step 1.

**Using the `convert_to_tflite.py` Script:**

The script takes several arguments to configure the conversion process. Ensure the script is located in your project root directory on Google Drive before running.

**Key Arguments:**

*   `--checkpoint`: **Required.** Path to your trained Fairseq checkpoint (e.g., `./checkpoints/how2sign_mediapipe_srf4k/checkpoint_best.pt`).
*   `--feat-dim`: **Required.** The dimensionality of your input features *after* preprocessing (e.g., the number of columns in your `.npy` files). Calculate this carefully based on your MediaPipe extraction and `mediapipe_json2npy.py` processing.
*   `--data-bin`: **Required.** Path to the Fairseq binarized data directory (e.g., `./data-bin/how2sign_mediapipe`). This is needed for the script to load the task configuration correctly.
*   `--output-onnx`: Path to save the intermediate ONNX model (e.g., `./tflite_models/encoder.onnx`).
*   `--output-tflite`: Path to save the final TFLite model (e.g., `./tflite_models/encoder.tflite`).
*   `--seq-len`: Maximum sequence length the model should handle during conversion (affects dummy input size).
*   `--fp16`: Add this flag to enable FP16 quantization (smaller model, potentially faster).
*   `--int8`: Add this flag to enable INT8 quantization (smallest model, potentially fastest, requires calibration data).
*   `--int8-dataset-npy`: Required if using `--int8`. Path to a `.npy` file containing a representative sample of your *preprocessed* training features (shape: `num_samples, seq_len, feat_dim`) used for calibration.

**Running the Conversion Script (Examples):**

Make sure you are in your project root directory (`/content/drive/MyDrive/sign_to_text_project/`) in Colab.

1.  **Create Output Directory:**
    ```bash
    # --- In Colab Cell ---
    !mkdir -p ./tflite_models
    ```

2.  **Run Conversion (No Quantization):**
    *Replace `--feat-dim` value with your actual feature dimension!*
    ```bash
    # --- In Colab Cell ---
    !python ./convert_to_tflite.py \
        --checkpoint ./checkpoints/how2sign_mediapipe_srf4k/checkpoint_best.pt \
        --data-bin ./data-bin/how2sign_mediapipe \
        --feat-dim 258 `# <-- IMPORTANT: Set your actual feature dimension` \
        --seq-len 256 \
        --output-onnx ./tflite_models/encoder.onnx \
        --output-tflite ./tflite_models/encoder.tflite
    ```

3.  **Run Conversion (FP16 Quantization):**
    *Replace `--feat-dim` value.*
    ```bash
    # --- In Colab Cell ---
    !python ./convert_to_tflite.py \
        --checkpoint ./checkpoints/how2sign_mediapipe_srf4k/checkpoint_best.pt \
        --data-bin ./data-bin/how2sign_mediapipe \
        --feat-dim 258 `# <-- IMPORTANT: Set your actual feature dimension` \
        --seq-len 256 \
        --output-onnx ./tflite_models/encoder_fp16.onnx \
        --output-tflite ./tflite_models/encoder_fp16.tflite \
        --fp16
    ```

4.  **Run Conversion (INT8 Quantization):**
    *Replace `--feat-dim` value. Create and provide path to `calibration_data.npy`.*
    *You need to generate `calibration_data.npy` yourself, containing ~100-1000 samples of your preprocessed training features.*
    ```bash
    # --- In Colab Cell ---
    # Example assumes you have created 'calibration_data.npy' in the project root
    !python ./convert_to_tflite.py \
        --checkpoint ./checkpoints/how2sign_mediapipe_srf4k/checkpoint_best.pt \
        --data-bin ./data-bin/how2sign_mediapipe \
        --feat-dim 258 `# <-- IMPORTANT: Set your actual feature dimension` \
        --seq-len 256 \
        --output-onnx ./tflite_models/encoder_int8.onnx \
        --output-tflite ./tflite_models/encoder_int8.tflite \
        --int8 \
        --int8-dataset-npy ./calibration_data.npy
    ```

**Verification:** The script includes a verification step that compares the output of the TFLite encoder model against the original PyTorch encoder using dummy data. Check the output logs for messages indicating whether the verification was successful or if the outputs differ significantly (especially important after quantization).

The resulting `.tflite` file (e.g., `encoder.tflite` or `encoder_fp16.tflite`) contains **only the encoder** and is ready for the next stage.

---

## 7. React Native Integration (Next Steps)

Once you have a working and verified `encoder.tflite` model:

1.  **TFLite Library:** Choose and integrate a TFLite runtime library into your React Native Expo app.
2.  **Preprocessing:** Implement the *exact* feature preprocessing steps (MediaPipe extraction, normalization, subsampling from `./mediapipe_config.yaml`) within your app using JavaScript/WASM libraries.
3.  **Run Encoder:** Load the `encoder.tflite` model. Feed the preprocessed feature tensor (`src_tokens`) and sequence lengths (`src_lengths`) to the model to get the `encoder_out` tensor.
4.  **Implement Decoder:** This is a significant step. You need to implement the text generation logic in JavaScript:
    *   Load your SentencePiece model vocabulary (`./How2Sign/processed/spm_en_bpe4000.model`) or use a JS library that can handle it.
    *   Start the decoding process with a Begin-Of-Sentence (BOS) token.
    *   Create a loop that:
        *   Takes the `encoder_out` tensor and the currently generated sequence of tokens.
        *   Performs the decoder's attention mechanism (cross-attention to encoder output, self-attention to previously generated tokens) and feed-forward layers to predict the *next* token's logits. **This part needs careful implementation, potentially by converting the decoder separately or reimplementing its core logic.**
        *   Selects the next token ID (e.g., using argmax for greedy search).
        *   Appends the ID to the sequence.
        *   Stops when an End-Of-Sentence (EOS) token is predicted or max length is reached.
5.  **Postprocessing:** Convert the final sequence of generated token IDs back into human-readable text using your SentencePiece model/vocabulary.
6.  **Integrate Camera & UI:** Use Expo Camera, run the full pipeline (preprocessing -> TFLite encoder -> JS decoder -> postprocessing), and display/speak the results.

This workflow provides a path from the How2Sign dataset to an *encoder* TFLite model suitable for your React Native app. Remember that implementing the decoder logic within the app is a non-trivial task. Good luck!