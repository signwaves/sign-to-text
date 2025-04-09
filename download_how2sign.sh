#!/bin/bash
#
# This script will create the folder structure and download the How2Sign dataset using gdown.
# It requires the 'gdown' tool to be installed (pip install gdown).
# For any questions about the data please contact: amanda.duarte[at]upc.edu
#
# To use this script, first choose the modalities that you would like to download and pass it as an argument to the command.
# For example, to download the "rgb_front_videos", the "rgb_side_videos" and the "english_translation_re-aligned" you can use the following command:
#
# ./download_how2sign_gdown.sh rgb_front_videos rgb_side_videos english_translation_re-aligned
#
# The names of the modalities avaliable for download can be found at the botton of this document
################################################################################

# --- Check for gdown dependency ---
if ! command -v gdown &> /dev/null
then
    echo "ERROR: 'gdown' command not found." >&2
    echo "Please install it first: pip install gdown" >&2
    exit 1
fi

# --- Check for required tools ---
if ! command -v unzip &> /dev/null; then echo "ERROR: 'unzip' command not found. Please install it." >&2; exit 1; fi
if ! command -v cat &> /dev/null; then echo "ERROR: 'cat' command not found. Please install it (usually part of coreutils)." >&2; exit 1; fi
if ! command -v tar &> /dev/null; then echo "ERROR: 'tar' command not found. Please install it." >&2; exit 1; fi


# --- Check for arguments ---
if (( $# < 1 ))
then
    echo "USAGE: $0 <argument1> <argument2> ..."
    echo "Example: $0 rgb_front_videos english_translation"
    echo "Available arguments:"
    echo "  rgb_front_videos"
    echo "  rgb_side_videos"
    echo "  rgb_front_clips"
    echo "  rgb_side_clips"
    echo "  rgb_front_2D_keypoints"
    # echo "  rgb_side_2D_keypoints" # Still commented out as in original
    echo "  english_translation"
    echo "  english_translation_re-aligned"
    exit 1
fi

echo "Downloading the How2Sign dataset using gdown"

#############################################
# Create folder structure and download data #
#############################################

# Updated Helper function for gdown with retry logic and verification
download_gdrive() {
    local file_id="$1"
    local output_file="$2"
    # Tunable parameters for retry mechanism
    local max_retries=10
    local sleep_time=15 # seconds to wait between retries

    local retry_count=0

    echo "Downloading GDrive ID: ${file_id} to ${output_file}"

    # Loop for retries
    while (( retry_count < max_retries )); do
        # Attempt download
        gdown "${file_id}" -O "${output_file}"
        local gdown_exit_status=$?

        # Check 1: gdown exit status.
        # Check 2: If gdown exit status is 0, verify file exists and is not empty (-s flag)
        if [[ $gdown_exit_status -eq 0 ]] && [[ -s "${output_file}" ]]; then
            echo "Successfully downloaded and verified ${output_file}"
            return 0 # Success! Exit the function with status 0.
        fi

        # If download failed (either gdown error or empty file)...
        retry_count=$((retry_count+1))
        echo "Download attempt ${retry_count}/${max_retries} failed for ${output_file} (gdown exit: ${gdown_exit_status})." >&2

        # Check if we have more retries left
        if (( retry_count < max_retries )); then
            echo "Retrying in ${sleep_time} seconds..."
            sleep ${sleep_time}
            # Optional: You could increase sleep_time here for exponential backoff
            # sleep_time=$((sleep_time * 2))
        else
            echo "ERROR: Download failed permanently after ${max_retries} attempts for GDrive ID ${file_id} -> ${output_file}." >&2
            echo "This might be due to persistent Google Drive quota issues or network problems." >&2
            echo "Consider trying the 'Make a Personal Copy' or 'Manual Browser Download' methods described previously." >&2
            # Clean up potentially empty/corrupted file from the last attempt
            rm -f "${output_file}"
            return 1 # Failure! Exit the function with status 1.
        fi
    done
}


#------------------------- Green Screen RGB videos - Frontal View -------------------------#
rgb_front_videos()
{
    local work_dir="." # Directory where temporary downloads happen
    local dest_train="./How2Sign/video_level/train/rgb_front"
    local dest_val="./How2Sign/video_level/val/rgb_front"
    local dest_test="./How2Sign/video_level/test/rgb_front"
    mkdir -p "$dest_train" "$dest_val" "$dest_test"

    echo "***** Downloading Green Screen RGB videos (Frontal View)... This might take a while!*****"

    ## Train Parts
    download_gdrive "1xWlMM2O3Gbp_8LK5FefoH0TVEmae6jIf" "${work_dir}/train_raw_videos.z01" || return 1
    download_gdrive "1krtYdpK_LQFgEUCnHxoYAW7EyhLMLWq0" "${work_dir}/train_raw_videos.z02" || return 1
    download_gdrive "1fXpWRNFhpuVm3ym7lT9vF_bnDjHkvP_K" "${work_dir}/train_raw_videos.z03" || return 1
    download_gdrive "1IFetFt4AzsxNCMZ0VVpX7YRgFAm58X48" "${work_dir}/train_raw_videos.z04" || return 1
    download_gdrive "1ZHuuun6Ae-AOLBns3LmuH7w8C9YCB4gH" "${work_dir}/train_raw_videos.z05" || return 1
    download_gdrive "1FQQIPblk-oLH_vu7h2tDO0oJaZ3xkp5N" "${work_dir}/train_raw_videos.z06" || return 1
    download_gdrive "19XNgERcolGAMPPgX-Gx_GebSTx3W4o0r" "${work_dir}/train_raw_videos.z07" || return 1
    download_gdrive "1YN-SA9uzrogEdKeT6UdQUIcuGEyYJILg" "${work_dir}/train_raw_videos.z08" || return 1
    download_gdrive "1SZQ2GzPLCkRqvsImAjULAPBiuAKi9DE9" "${work_dir}/train_raw_videos.z09" || return 1
    download_gdrive "1Xe1T5okJiopMXUiH3sc0mdCWNDYSBopd" "${work_dir}/train_raw_videos.zip" || return 1

    ## Val
    download_gdrive "1fCkyuKSsc7gauljuL9sx_jBomf3N6i0g" "${work_dir}/val_raw_videos.zip" || return 1

    ## Test
    download_gdrive "1z0i6BBGHQ12ChY63hZH56QnczvQ0JfTb" "${work_dir}/test_raw_videos.zip" || return 1

    # Merge all train zip files
    echo "***** Preparing the downloaded files... this might take some time! *****"
    echo "Combining split train archives..."
    # Use absolute or relative paths consistently
    cat "${work_dir}/train_raw_videos.z"* > "${work_dir}/train_raw_videos_all.zip"
    if [[ $? -ne 0 ]]; then echo "Failed to combine train archives! Check if all .z* parts downloaded correctly." >&2; return 1; fi
    echo "Cleaning up split train archives..."
    rm -f "${work_dir}/train_raw_videos.z"? # Removes .z01 through .z09
    rm -f "${work_dir}/train_raw_videos.z"?? # Removes .z10 through .z99 (just in case)
    rm -f "${work_dir}/train_raw_videos.zip" # Remove the last downloaded part (.zip)


    echo "Unzipping train set..."
    unzip -q "${work_dir}/train_raw_videos_all.zip" -d "$dest_train"
    if [[ $? -ne 0 ]]; then echo "Failed to unzip train set!" >&2; return 1; fi
    rm -f "${work_dir}/train_raw_videos_all.zip"

    echo "Unzipping validation set..."
    unzip -q "${work_dir}/val_raw_videos.zip" -d "$dest_val"
    if [[ $? -ne 0 ]]; then echo "Failed to unzip validation set!" >&2; return 1; fi
    rm -f "${work_dir}/val_raw_videos.zip"

    echo "Unzipping test set..."
    unzip -q "${work_dir}/test_raw_videos.zip" -d "$dest_test"
    if [[ $? -ne 0 ]]; then echo "Failed to unzip test set!" >&2; return 1; fi
    rm -f "${work_dir}/test_raw_videos.zip"

    echo "Frontal videos downloaded and extracted."
}

#------------------------- Green Screen RGB videos - Side View -------------------------#
rgb_side_videos()
{
    local work_dir="."
    local dest_train="./How2Sign/video_level/train/rgb_side"
    local dest_val="./How2Sign/video_level/val/rgb_side"
    local dest_test="./How2Sign/video_level/test/rgb_side"
    mkdir -p "$dest_train" "$dest_val" "$dest_test"

    echo "***** Downloading Green Screen RGB videos (Side View)... This might take a while! *****"

    ## Train
    download_gdrive "1Rmf6LfNWn6lWkAz6Iuj5pMOI2I5p4j1U" "${work_dir}/train_side_raw_videos.z01" || return 1
    download_gdrive "1FytIYIRYrBgAeNWIAhO5vnI2mYOvYC9i" "${work_dir}/train_side_raw_videos.z02" || return 1
    download_gdrive "1kC24jgNgjYYiIYhCRE-gGR28H_2xBBbP" "${work_dir}/train_side_raw_videos.z03" || return 1
    download_gdrive "1JunkM-ImFYao_MwDW9zeqe-6Th6rOLhR" "${work_dir}/train_side_raw_videos.z04" || return 1
    download_gdrive "1-vMckelz9fy4GVNYXRCcy7cJ12X4P3KZ" "${work_dir}/train_side_raw_videos.z05" || return 1
    download_gdrive "1uV413eKsihkNzquN2bwtIQG-OZZMz6sh" "${work_dir}/train_side_raw_videos.z06" || return 1
    download_gdrive "1sU8xrneFJHBzT_PFz4iRPqI8A7HGilhW" "${work_dir}/train_side_raw_videos.z07" || return 1
    download_gdrive "1RPLxeZ54uSZUJSXdPFhXOgeIXziOwTW9" "${work_dir}/train_side_raw_videos.z08" || return 1
    download_gdrive "1tClhr98PszBvFpo9ELKuhbTZZgTGGQqh" "${work_dir}/train_side_raw_videos.z09" || return 1
    download_gdrive "10xrXWgH7iW3E6sgJZDPRwlIhIaDLfHQm" "${work_dir}/train_side_raw_videos.zip" || return 1

    ## Val
    download_gdrive "1Z2H96JT68o7eTChEXPI9z3xyx7zUJPl5" "${work_dir}/val_rgb_side_raw_videos.zip" || return 1

    ## Test
    download_gdrive "1tCQ8KIuuiirXHsh29w0XAMNB3HLIGqgA" "${work_dir}/test_rgb_side_raw_videos.zip" || return 1

    # Merge all train zip files
    echo "***** Preparing the downloaded files... this might take some time! *****"
    echo "Combining split train archives..."
    cat "${work_dir}/train_side_raw_videos.z"* > "${work_dir}/train_side_raw_videos_all.zip"
    if [[ $? -ne 0 ]]; then echo "Failed to combine train archives! Check if all .z* parts downloaded correctly." >&2; return 1; fi
    echo "Cleaning up split train archives..."
    rm -f "${work_dir}/train_side_raw_videos.z"?
    rm -f "${work_dir}/train_side_raw_videos.z"??
    rm -f "${work_dir}/train_side_raw_videos.zip" # Remove the last downloaded part (.zip)

    echo "Unzipping train set..."
    unzip -q "${work_dir}/train_side_raw_videos_all.zip" -d "$dest_train"
    if [[ $? -ne 0 ]]; then echo "Failed to unzip train set!" >&2; return 1; fi
    rm -f "${work_dir}/train_side_raw_videos_all.zip"

    echo "Unzipping validation set..."
    unzip -q "${work_dir}/val_rgb_side_raw_videos.zip" -d "$dest_val"
    if [[ $? -ne 0 ]]; then echo "Failed to unzip validation set!" >&2; return 1; fi
    rm -f "${work_dir}/val_rgb_side_raw_videos.zip"

    echo "Unzipping test set..."
    unzip -q "${work_dir}/test_rgb_side_raw_videos.zip" -d "$dest_test"
    if [[ $? -ne 0 ]]; then echo "Failed to unzip test set!" >&2; return 1; fi
    rm -f "${work_dir}/test_rgb_side_raw_videos.zip"

    echo "Side videos downloaded and extracted."
}

#------------------------- Green Screen RGB clips -- Frontal view -------------------------#
rgb_front_clips()
{
    local work_dir="."
    local dest_train="./How2Sign/sentence_level/train/rgb_front"
    local dest_val="./How2Sign/sentence_level/val/rgb_front"
    local dest_test="./How2Sign/sentence_level/test/rgb_front"
    mkdir -p "$dest_train" "$dest_val" "$dest_test"

    echo "***** Downloading and preparing the Green Screen RGB clips (Frontal view) *****"

    ## Train
    download_gdrive "1VX7n0jjW0pW3GEdgOks3z8nqE6iI6EnW" "${work_dir}/train_rgb_front_clips.zip" || return 1
    ## Val
    download_gdrive "1DhLH8tIBn9HsTzUJUfsEOGcP4l9EvOiO" "${work_dir}/val_rgb_front_clips.zip" || return 1
    ## Test
    download_gdrive "1qTIXFsu8M55HrCiaGv7vZ7GkdB3ubjaG" "${work_dir}/test_rgb_front_clips.zip" || return 1

    echo "Unzipping train clips..."
    unzip -q "${work_dir}/train_rgb_front_clips.zip" -d "$dest_train"
    if [[ $? -ne 0 ]]; then echo "Failed to unzip train clips!" >&2; return 1; fi
    rm -f "${work_dir}/train_rgb_front_clips.zip"

    echo "Unzipping validation clips..."
    unzip -q "${work_dir}/val_rgb_front_clips.zip"   -d "$dest_val"
    if [[ $? -ne 0 ]]; then echo "Failed to unzip validation clips!" >&2; return 1; fi
    rm -f "${work_dir}/val_rgb_front_clips.zip"

    echo "Unzipping test clips..."
    unzip -q "${work_dir}/test_rgb_front_clips.zip"  -d "$dest_test"
    if [[ $? -ne 0 ]]; then echo "Failed to unzip test clips!" >&2; return 1; fi
    rm -f "${work_dir}/test_rgb_front_clips.zip"

    echo "Frontal clips downloaded and extracted."
}

#------------------------- Green Screen RGB clips -- Side view -------------------------#
rgb_side_clips()
{
    local work_dir="."
    local dest_train="./How2Sign/sentence_level/train/rgb_side"
    local dest_val="./How2Sign/sentence_level/val/rgb_side"
    local dest_test="./How2Sign/sentence_level/test/rgb_side"
    mkdir -p "$dest_train" "$dest_val" "$dest_test"

    echo "***** Downloading and preparing the Green Screen RGB clips (Side view) *****"

    ## Train
    download_gdrive "1oiw861NGp4CKKFO3iuHGSCgTyQ-DXHW7" "${work_dir}/train_rgb_side_clips.zip" || return 1
    ## Val
    download_gdrive "1mxL7kJPNUzJ6zoaqJyxF1Krnjo4F-eQG" "${work_dir}/val_rgb_side_clips.zip" || return 1
    ## Test
    download_gdrive "1j9v9P7UdMJ0_FVWg8H95cqx4DMSsrdbH" "${work_dir}/test_rgb_side_clips.zip" || return 1

    echo "Unzipping train clips..."
    unzip -q "${work_dir}/train_rgb_side_clips.zip" -d "$dest_train"
    if [[ $? -ne 0 ]]; then echo "Failed to unzip train clips!" >&2; return 1; fi
    rm -f "${work_dir}/train_rgb_side_clips.zip"

    echo "Unzipping validation clips..."
    unzip -q "${work_dir}/val_rgb_side_clips.zip"   -d "$dest_val"
    if [[ $? -ne 0 ]]; then echo "Failed to unzip validation clips!" >&2; return 1; fi
    rm -f "${work_dir}/val_rgb_side_clips.zip"

    echo "Unzipping test clips..."
    unzip -q "${work_dir}/test_rgb_side_clips.zip"  -d "$dest_test"
    if [[ $? -ne 0 ]]; then echo "Failed to unzip test clips!" >&2; return 1; fi
    rm -f "${work_dir}/test_rgb_side_clips.zip"

    echo "Side clips downloaded and extracted."
}

#------------------------- B-F-H 2D Keypoints clips -- Frontal view -------------------------#
rgb_front_2D_keypoints()
{
    local work_dir="."
    local dest_train="./How2Sign/sentence_level/train/rgb_front/features"
    local dest_val="./How2Sign/sentence_level/val/rgb_front/features"
    local dest_test="./How2Sign/sentence_level/test/rgb_front/features"
    mkdir -p "$dest_train" "$dest_val" "$dest_test"

    echo "***** Downloading B-F-H 2D Keypoints clips (Frontal view) files... This might take a while! *****"
    ## Train
    download_gdrive "1TBX7hLraMiiLucknM1mhblNVomO9-Y0r" "${work_dir}/train_2D_keypoints.tar.gz" || return 1
    ## Val
    download_gdrive "1JmEsU0GYUD5iVdefMOZpeWa_iYnmK_7w" "${work_dir}/val_2D_keypoints.tar.gz" || return 1
    ## Test
    download_gdrive "1g8tzzW5BNPzHXlamuMQOvdwlHRa-29Vp" "${work_dir}/test_2D_keypoints.tar.gz" || return 1

    echo "***** Preparing the downloaded files... this might take some time! *****"
    echo "Extracting train keypoints..."
    tar -xzf "${work_dir}/train_2D_keypoints.tar.gz" -C "$dest_train" # Added -z for .gz
    if [[ $? -ne 0 ]]; then echo "Failed to extract train keypoints!" >&2; return 1; fi
    rm -f "${work_dir}/train_2D_keypoints.tar.gz"

    echo "Extracting validation keypoints..."
    tar -xzf "${work_dir}/val_2D_keypoints.tar.gz"   -C "$dest_val" # Added -z for .gz
    if [[ $? -ne 0 ]]; then echo "Failed to extract validation keypoints!" >&2; return 1; fi
    rm -f "${work_dir}/val_2D_keypoints.tar.gz"

    echo "Extracting test keypoints..."
    tar -xzf "${work_dir}/test_2D_keypoints.tar.gz"  -C "$dest_test" # Added -z for .gz
    if [[ $? -ne 0 ]]; then echo "Failed to extract test keypoints!" >&2; return 1; fi
    rm -f "${work_dir}/test_2D_keypoints.tar.gz"

    echo "Frontal 2D keypoints downloaded and extracted."
}

# # B-F-H 2D Keypoints clips -- Side view
# rgb_side_2D_keypoints()
# {
#     # This section remains commented out as in the original script.
#     # If enabled, the gdown lines would need converting similar to above.
#     echo "Creating B-F-H 2D Keypoints clips -- Side view folders"
#     mkdir -p "./How2Sign/sentence_level/train/rgb_side/features/openpose_output"
#     mkdir -p "./How2Sign/sentence_level/val/rgb_side/features/openpose_output"
#     mkdir -p "./How2Sign/sentence_level/test/rgb_side/features/openpose_output"
#     # Download and extract logic would go here if needed
# }

#------------------------- English Translation -------------------------#
english_translation()
{
    local work_dir="."
    local dest_train="./How2Sign/sentence_level/train/text/en/raw_text"
    local dest_val="./How2Sign/sentence_level/val/text/en/raw_text"
    local dest_test="./How2Sign/sentence_level/test/text/en/raw_text"
    mkdir -p "$dest_train" "$dest_val" "$dest_test"

    echo "***** Downloading and preparing the English Translation text files *****"
    ## Train
    download_gdrive "1lq7ksWeD3FzaIwowRbe_BvCmSmOG12-f" "${work_dir}/how2sign_train.csv" || return 1
    ## Val
    download_gdrive "1aBQUClTlZB504JtDISJ0DJlbuYUZCGu3" "${work_dir}/how2sign_val.csv" || return 1
    ## Test
    download_gdrive "1ScxYnEjILZMn22qKjQj8Wyr_F0nha7kG" "${work_dir}/how2sign_test.csv" || return 1

    # Move files only if download was successful (implied by reaching here)
    mv "${work_dir}/how2sign_train.csv" "$dest_train/" || { echo "Failed to move train text file!" >&2; return 1; }
    mv "${work_dir}/how2sign_val.csv" "$dest_val/" || { echo "Failed to move val text file!" >&2; return 1; }
    mv "${work_dir}/how2sign_test.csv" "$dest_test/" || { echo "Failed to move test text file!" >&2; return 1; }

    echo "English translation files downloaded and moved."
}

#------------------------- English Translation re-aligned -------------------------#
english_translation_re-aligned()
{
    local work_dir="."
    local dest_train="./How2Sign/sentence_level/train/text/en/raw_text/re_aligned"
    local dest_val="./How2Sign/sentence_level/val/text/en/raw_text/re_aligned"
    local dest_test="./How2Sign/sentence_level/test/text/en/raw_text/re_aligned"
    mkdir -p "$dest_train" "$dest_val" "$dest_test"

    echo "***** Downloading and preparing the re-aligned English Translation text files *****"
    ## Train
    download_gdrive "1dUHSoefk9OxKJnHrHPX--I4tpm9QD0ok" "${work_dir}/how2sign_realigned_train.csv" || return 1
    ## Val
    download_gdrive "1Vpag7VPfdTCCJSao8Pz14rlPfekRMggI" "${work_dir}/how2sign_realigned_val.csv" || return 1
    ## Test
    download_gdrive "1AgwBZW26kFHS4CWNMQTCMPGkBPkH3qCu" "${work_dir}/how2sign_realigned_test.csv" || return 1

    # Move files only if download was successful
    mv "${work_dir}/how2sign_realigned_train.csv" "$dest_train/" || { echo "Failed to move re-aligned train text file!" >&2; return 1; }
    mv "${work_dir}/how2sign_realigned_val.csv" "$dest_val/" || { echo "Failed to move re-aligned val text file!" >&2; return 1; }
    mv "${work_dir}/how2sign_realigned_test.csv" "$dest_test/" || { echo "Failed to move re-aligned test text file!" >&2; return 1; }

    echo "Re-aligned English translation files downloaded and moved."
}

## TODO
# Gloss annotations
# Panoptic Studio data

# --- Main Execution Logic ---
# Iterate through command-line arguments and call corresponding functions
overall_status=0
processed_args=() # Keep track of processed arguments

# Check if any arguments were provided
if (( $# == 0 )); then
    echo "No download arguments provided. See usage info above."
    exit 1
fi


for ARG in "$@"
do
    # Simple check to prevent processing the same argument multiple times if listed twice
    if [[ " ${processed_args[*]} " =~ " ${ARG} " ]]; then
        echo "Skipping already processed argument: ${ARG}"
        continue
    fi

    echo "----------------------------------------"
    echo "Processing argument: ${ARG}"
    echo "----------------------------------------"

    task_failed=0
    case "${ARG}" in
        "rgb_front_videos")             rgb_front_videos;;
        "rgb_side_videos")              rgb_side_videos;;
        "rgb_front_clips")              rgb_front_clips;;
        "rgb_side_clips")               rgb_side_clips;;
        "rgb_front_2D_keypoints")       rgb_front_2D_keypoints;;
        # "rgb_side_2D_keypoints")      rgb_side_2D_keypoints;; # Still commented out
        "english_translation")          english_translation;;
        "english_translation_re-aligned") english_translation_re-aligned;;
        *)
            echo "!!! WARNING: Invalid argument '${ARG}' given. Skipping. !!!" >&2
            # Optionally treat invalid args as failure: task_failed=1; overall_status=1
            ;;
    esac

    # Check the exit status ($?) of the function called by the case statement
    # This relies on the functions returning non-zero on error (e.g., using || return 1)
    if [[ $? -ne 0 ]]; then
        echo "!!! ERROR processing '${ARG}' !!!" >&2
        overall_status=1 # Mark overall failure
        task_failed=1    # Mark this specific task as failed
        # Decide if you want to stop immediately on first error:
        # echo "Exiting due to error." >&2
        # exit 1
    fi

    # Mark argument as processed (even if it failed, to avoid retrying in this run)
    processed_args+=("${ARG}")

    if [[ $task_failed -eq 0 && "${ARG}" != *"Invalid argument"* ]]; then
         echo "--- Successfully processed '${ARG}' ---"
    fi

done

echo "========================================"
if [[ $overall_status -eq 0 ]]; then
    echo "All requested downloads and preparations finished successfully."
    echo "Please check the README file for information about the files you just downloaded."
    echo "Feel free to contact amanda.duarte[at]upc.edu if you have any questions about the data."
else
    echo "One or more tasks failed. Please check the output above for specific errors." >&2
fi
echo "========================================"

exit $overall_status
#
################################################################################
