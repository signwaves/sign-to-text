#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import multiprocessing
import os
import traceback # Import traceback for better error reporting
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np # Although not directly used for landmarks, good practice
from tqdm import tqdm

# Define expected landmark counts directly from MediaPipe Holistic standard output
MP_LMKS = {
    'face': 468,
    'pose': 33,
    'left_hand': 21,
    'right_hand': 21,
}

# Initialize MediaPipe Holistic solution model
mp_holistic = mp.solutions.holistic

def format_landmarks(landmark_list, expected_count):
    """
    Formats MediaPipe landmarks into 'x,y,z' strings.
    If landmarks are missing, pads with '0.0,0.0,0.0'.
    Returns the number of landmarks corresponding to standard MediaPipe output.
    """
    output_lms = []
    if landmark_list:
        for lm in landmark_list.landmark:
            # Store as x,y,z (visibility/presence ignored for target format compatibility)
            output_lms.append(f"{lm.x},{lm.y},{lm.z}")

    # Pad with placeholders if fewer landmarks were detected than expected by MediaPipe standard
    while len(output_lms) < expected_count:
         # Using "0,0,0" as placeholder, consistent with missing data often being at origin
        output_lms.append("0.0,0.0,0.0")

    # Return exactly the number of landmarks MediaPipe provides for this part
    return output_lms[:expected_count] # Ensure we don't exceed expected count (safety)

def process_video(video_path: Path, output_dir: Path, video_dir_root: Path):
    """
    Processes a single video file, extracts landmarks from the first valid frame,
    and saves them to a JSON file in the target structure.
    Requires video_dir_root to calculate relative path correctly.
    """
    try:
        # Determine output path based on relative structure
        relative_path = video_path.relative_to(video_dir_root)
        output_json_path = (output_dir / relative_path).with_suffix('.json')

        # Skip if already processed
        if output_json_path.exists():
            return f"Skipped: {video_path.name}"

        output_json_path.parent.mkdir(parents=True, exist_ok=True)

        # --- MediaPipe Processing ---
        # Initialize Holistic model inside the worker process
        holistic = mp_holistic.Holistic(
            static_image_mode=False, # Process video
            model_complexity=1,      # 0, 1, or 2. Higher = more accurate but slower.
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3)

        cap = cv2.VideoCapture(str(video_path)) # Use string path for cv2
        if not cap.isOpened():
            holistic.close()
            return f"Error opening video: {video_path.name}"

        processed_data = {
            # Initialize keys based on MediaPipe names used in mediapipe_json2npy.py
            "face_landmarks": {"landmarks": []},
            "pose_landmarks": {"landmarks": []},
            "left_hand_landmarks": {"landmarks": []},
            "right_hand_landmarks": {"landmarks": []}
        }
        found_landmarks = False

        while cap.isOpened() and not found_landmarks:
            success, image = cap.read()
            if not success:
                break # End of video or error reading frame

            # Convert the BGR image to RGB for MediaPipe
            image.flags.writeable = False # Performance optimization
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = holistic.process(image_rgb)
            image.flags.writeable = True # Reset writeable flag

            # --- Extract landmarks IF DETECTED in this frame ---
            # We take landmarks from the *first* frame where detection occurs
            # Using pose landmarks as the primary check, but ensuring at least one type is found
            if results.pose_landmarks or results.face_landmarks or results.left_hand_landmarks or results.right_hand_landmarks:

                processed_data["face_landmarks"]["landmarks"] = format_landmarks(results.face_landmarks, MP_LMKS['face'])
                processed_data["pose_landmarks"]["landmarks"] = format_landmarks(results.pose_landmarks, MP_LMKS['pose'])
                processed_data["left_hand_landmarks"]["landmarks"] = format_landmarks(results.left_hand_landmarks, MP_LMKS['left_hand'])
                processed_data["right_hand_landmarks"]["landmarks"] = format_landmarks(results.right_hand_landmarks, MP_LMKS['right_hand'])
                found_landmarks = True # Stop after finding the first frame with any landmarks

        # If no landmarks were found in any frame, fill structure with placeholders
        if not found_landmarks:
             print(f"Warning: No landmarks detected in {video_path.name}. Creating placeholder file.")
             processed_data["face_landmarks"]["landmarks"] = ["0.0,0.0,0.0"] * MP_LMKS['face']
             processed_data["pose_landmarks"]["landmarks"] = ["0.0,0.0,0.0"] * MP_LMKS['pose']
             processed_data["left_hand_landmarks"]["landmarks"] = ["0.0,0.0,0.0"] * MP_LMKS['left_hand']
             processed_data["right_hand_landmarks"]["landmarks"] = ["0.0,0.0,0.0"] * MP_LMKS['right_hand']

        # Save the structured data to JSON
        with open(output_json_path, 'w') as f:
            # Use separators=(',', ':') for most compact JSON
            json.dump(processed_data, f, separators=(',', ':'))

        cap.release()
        holistic.close()
        return f"Processed: {video_path.name}"

    except Exception as e:
        # Print traceback for detailed error info
        cap.release() # Ensure cap is released on error
        holistic.close() # Ensure model is closed on error
        tb_str = traceback.format_exc()
        return f"FAILED: {video_path.name} | Error: {e}\nTraceback:\n{tb_str}"


def parse_args():
    parser = argparse.ArgumentParser(description="Extract MediaPipe landmarks from videos in a directory and save as JSON.")
    parser.add_argument('--video-dir', type=str, required=True, help="Directory containing input video files (e.g., MP4). Search is recursive.")
    parser.add_argument('--output-dir', type=str, required=True, help="Directory where the output JSON files will be saved, mirroring the input structure.")
    # Use os.cpu_count() for a potentially better default
    parser.add_argument('--processes', type=int, default=max(1, os.cpu_count() // 2 if os.cpu_count() else 1), help="Number of parallel processes to use.")
    return parser.parse_args()

def main():
    args = parse_args()

    video_dir = Path(args.video_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    num_processes = args.processes

    if not video_dir.is_dir():
        print(f"Error: Input video directory not found: {video_dir}")
        return

    print(f"Input video directory: {video_dir}")
    print(f"Output JSON directory: {output_dir}")
    print(f"Using {num_processes} processes.")

    # Find all video files (add/remove extensions as needed)
    video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.webm']
    video_files = []
    for ext in video_extensions:
        video_files.extend(list(video_dir.rglob(ext)))

    if not video_files:
        print(f"Error: No video files ({', '.join(video_extensions)}) found in the input directory.")
        return

    print(f"Found {len(video_files)} video files.")

    # Create argument list for multiprocessing pool, passing the root video_dir
    pool_args = [(vf, output_dir, video_dir) for vf in video_files]

    # Run processing in parallel
    print("Starting parallel processing...")
    # Use starmap as process_video takes multiple arguments
    with multiprocessing.Pool(processes=num_processes) as pool:
         # Wrap starmap with tqdm for a progress bar
        results = list(tqdm(pool.starmap(process_video, pool_args), total=len(video_files)))

    print("Processing finished.")

    # Optional: Summarize results / print failures
    failures = [r for r in results if "FAILED" in r]
    skips = [r for r in results if "Skipped" in r]
    processed_count = len(results) - len(failures) - len(skips)
    print(f"Summary: Processed={processed_count}, Skipped={len(skips)}, Failed={len(failures)}")
    if failures:
        print("\n--- Failures ---")
        for fail in failures:
            print(fail)
        print("--- End Failures ---")


if __name__ == '__main__':
    main()
