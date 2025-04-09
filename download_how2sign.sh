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

# Helper function for gdown with retry logic (optional but good for large files)
download_gdrive() {
    local file_id="$1"
    local output_file="$2"
    echo "Downloading GDrive ID: ${file_id} to ${output_file}"
    # Simple download without retry:
    gdown "${file_id}" -O "${output_file}"
    # Uncomment below for basic retry logic:
    # local max_retries=3
    # local retry_count=0
    # until gdown "${file_id}" -O "${output_file}"; do
    #     retry_count=$((retry_count+1))
    #     if (( retry_count >= max_retries )); then
    #         echo "Download failed after ${max_retries} attempts for ${file_id}" >&2
    #         return 1
    #     fi
    #     echo "Download failed for ${file_id}. Retrying (${retry_count}/${max_retries})..."
    #     sleep 5
    # done
    echo "Successfully downloaded ${output_file}"
}


#------------------------- Green Screen RGB videos - Frontal View -------------------------#
rgb_front_videos()
{
    local dest_train="./How2Sign/video_level/train/rgb_front"
    local dest_val="./How2Sign/video_level/val/rgb_front"
    local dest_test="./How2Sign/video_level/test/rgb_front"
    mkdir -p "$dest_train" "$dest_val" "$dest_test"

    echo "***** Downloading Green Screen RGB videos (Frontal View)... This might take a while!*****"

    ## Train
    download_gdrive "1xWlMM2O3Gbp_8LK5FefoH0TVEmae6jIf" "train_raw_videos.z01" || return 1
    download_gdrive "1krtYdpK_LQFgEUCnHxoYAW7EyhLMLWq0" "train_raw_videos.z02" || return 1
    download_gdrive "1fXpWRNFhpuVm3ym7lT9vF_bnDjHkvP_K" "train_raw_videos.z03" || return 1
    download_gdrive "1IFetFt4AzsxNCMZ0VVpX7YRgFAm58X48" "train_raw_videos.z04" || return 1
    download_gdrive "1ZHuuun6Ae-AOLBns3LmuH7w8C9YCB4gH" "train_raw_videos.z05" || return 1
    download_gdrive "1FQQIPblk-oLH_vu7h2tDO0oJaZ3xkp5N" "train_raw_videos.z06" || return 1
    download_gdrive "19XNgERcolGAMPPgX-Gx_GebSTx3W4o0r" "train_raw_videos.z07" || return 1
    download_gdrive "1YN-SA9uzrogEdKeT6UdQUIcuGEyYJILg" "train_raw_videos.z08" || return 1
    download_gdrive "1SZQ2GzPLCkRqvsImAjULAPBiuAKi9DE9" "train_raw_videos.z09" || return 1
    download_gdrive "1Xe1T5okJiopMXUiH3sc0mdCWNDYSBopd" "train_raw_videos.zip" || return 1

    ## Val
    download_gdrive "1fCkyuKSsc7gauljuL9sx_jBomf3N6i0g" "val_raw_videos.zip" || return 1

    ## Test
    download_gdrive "1z0i6BBGHQ12ChY63hZH56QnczvQ0JfTb" "test_raw_videos.zip" || return 1

    # Merge all train zip files
    echo "***** Preparing the downloaded files... this might take some time! *****"
    echo "Combining split train archives..."
    cat train_raw_videos.z* > train_raw_videos_all.zip
    if [[ $? -ne 0 ]]; then echo "Failed to combine train archives!" >&2; return 1; fi
    rm -f train_raw_videos.z?? train_raw_videos.zip # Clean up parts

    echo "Unzipping train set..."
    unzip -q train_raw_videos_all.zip -d "$dest_train" && rm -f train_raw_videos_all.zip
    if [[ $? -ne 0 ]]; then echo "Failed to unzip train set!" >&2; return 1; fi

    echo "Unzipping validation set..."
    unzip -q val_raw_videos.zip -d "$dest_val" && rm -f val_raw_videos.zip
    if [[ $? -ne 0 ]]; then echo "Failed to unzip validation set!" >&2; return 1; fi

    echo "Unzipping test set..."
    unzip -q test_raw_videos.zip -d "$dest_test" && rm -f test_raw_videos.zip
    if [[ $? -ne 0 ]]; then echo "Failed to unzip test set!" >&2; return 1; fi
    echo "Frontal videos downloaded and extracted."
}

#------------------------- Green Screen RGB videos - Side View -------------------------#
rgb_side_videos()
{
    local dest_train="./How2Sign/video_level/train/rgb_side"
    local dest_val="./How2Sign/video_level/val/rgb_side"
    local dest_test="./How2Sign/video_level/test/rgb_side"
    mkdir -p "$dest_train" "$dest_val" "$dest_test"

    echo "***** Downloading Green Screen RGB videos (Side View)... This might take a while! *****"

    ## Train
    download_gdrive "1Rmf6LfNWn6lWkAz6Iuj5pMOI2I5p4j1U" "train_side_raw_videos.z01" || return 1
    download_gdrive "1FytIYIRYrBgAeNWIAhO5vnI2mYOvYC9i" "train_side_raw_videos.z02" || return 1
    download_gdrive "1kC24jgNgjYYiIYhCRE-gGR28H_2xBBbP" "train_side_raw_videos.z03" || return 1
    download_gdrive "1JunkM-ImFYao_MwDW9zeqe-6Th6rOLhR" "train_side_raw_videos.z04" || return 1
    download_gdrive "1-vMckelz9fy4GVNYXRCcy7cJ12X4P3KZ" "train_side_raw_videos.z05" || return 1
    download_gdrive "1uV413eKsihkNzquN2bwtIQG-OZZMz6sh" "train_side_raw_videos.z06" || return 1
    download_gdrive "1sU8xrneFJHBzT_PFz4iRPqI8A7HGilhW" "train_side_raw_videos.z07" || return 1
    download_gdrive "1RPLxeZ54uSZUJSXdPFhXOgeIXziOwTW9" "train_side_raw_videos.z08" || return 1
    download_gdrive "1tClhr98PszBvFpo9ELKuhbTZZgTGGQqh" "train_side_raw_videos.z09" || return 1
    download_gdrive "10xrXWgH7iW3E6sgJZDPRwlIhIaDLfHQm" "train_side_raw_videos.zip" || return 1

    ## Val
    download_gdrive "1Z2H96JT68o7eTChEXPI9z3xyx7zUJPl5" "val_rgb_side_raw_videos.zip" || return 1

    ## Test
    download_gdrive "1tCQ8KIuuiirXHsh29w0XAMNB3HLIGqgA" "test_rgb_side_raw_videos.zip" || return 1

    # Merge all train zip files
    echo "***** Preparing the downloaded files... this might take some time! *****"
    echo "Combining split train archives..."
    # Note: The original script output name here was train_side_raw_videos.zip which conflicts with the last part downloaded. Renaming combined file.
    cat train_side_raw_videos.z* > train_side_raw_videos_all.zip
    if [[ $? -ne 0 ]]; then echo "Failed to combine train archives!" >&2; return 1; fi
    rm -f train_side_raw_videos.z?? train_side_raw_videos.zip # Clean up parts

    echo "Unzipping train set..."
    unzip -q train_side_raw_videos_all.zip -d "$dest_train" && rm -f train_side_raw_videos_all.zip
    if [[ $? -ne 0 ]]; then echo "Failed to unzip train set!" >&2; return 1; fi

    echo "Unzipping validation set..."
    unzip -q val_rgb_side_raw_videos.zip -d "$dest_val" && rm -f val_rgb_side_raw_videos.zip
    if [[ $? -ne 0 ]]; then echo "Failed to unzip validation set!" >&2; return 1; fi

    echo "Unzipping test set..."
    unzip -q test_rgb_side_raw_videos.zip -d "$dest_test" && rm -f test_rgb_side_raw_videos.zip
    if [[ $? -ne 0 ]]; then echo "Failed to unzip test set!" >&2; return 1; fi
    echo "Side videos downloaded and extracted."
}

#------------------------- Green Screen RGB clips -- Frontal view -------------------------#
rgb_front_clips()
{
    local dest_train="./How2Sign/sentence_level/train/rgb_front"
    local dest_val="./How2Sign/sentence_level/val/rgb_front"
    local dest_test="./How2Sign/sentence_level/test/rgb_front"
    mkdir -p "$dest_train" "$dest_val" "$dest_test"

    echo "***** Downloading and preparing the Green Screen RGB clips (Frontal view) *****"

    ## Train
    download_gdrive "1VX7n0jjW0pW3GEdgOks3z8nqE6iI6EnW" "train_rgb_front_clips.zip" || return 1
    ## Val
    download_gdrive "1DhLH8tIBn9HsTzUJUfsEOGcP4l9EvOiO" "val_rgb_front_clips.zip" || return 1
    ## Test
    download_gdrive "1qTIXFsu8M55HrCiaGv7vZ7GkdB3ubjaG" "test_rgb_front_clips.zip" || return 1

    echo "Unzipping train clips..."
    unzip -q train_rgb_front_clips.zip -d "$dest_train" && rm -f train_rgb_front_clips.zip
    if [[ $? -ne 0 ]]; then echo "Failed to unzip train clips!" >&2; return 1; fi

    echo "Unzipping validation clips..."
    unzip -q val_rgb_front_clips.zip   -d "$dest_val" && rm -f val_rgb_front_clips.zip
    if [[ $? -ne 0 ]]; then echo "Failed to unzip validation clips!" >&2; return 1; fi

    echo "Unzipping test clips..."
    unzip -q test_rgb_front_clips.zip  -d "$dest_test" && rm -f test_rgb_front_clips.zip
    if [[ $? -ne 0 ]]; then echo "Failed to unzip test clips!" >&2; return 1; fi
    echo "Frontal clips downloaded and extracted."
}

#------------------------- Green Screen RGB clips -- Side view -------------------------#
rgb_side_clips()
{
    local dest_train="./How2Sign/sentence_level/train/rgb_side"
    local dest_val="./How2Sign/sentence_level/val/rgb_side"
    local dest_test="./How2Sign/sentence_level/test/rgb_side"
    mkdir -p "$dest_train" "$dest_val" "$dest_test"

    echo "***** Downloading and preparing the Green Screen RGB clips (Side view) *****"

    ## Train
    download_gdrive "1oiw861NGp4CKKFO3iuHGSCgTyQ-DXHW7" "train_rgb_side_clips.zip" || return 1
    ## Val
    download_gdrive "1mxL7kJPNUzJ6zoaqJyxF1Krnjo4F-eQG" "val_rgb_side_clips.zip" || return 1
    ## Test
    download_gdrive "1j9v9P7UdMJ0_FVWg8H95cqx4DMSsrdbH" "test_rgb_side_clips.zip" || return 1

    echo "Unzipping train clips..."
    unzip -q train_rgb_side_clips.zip -d "$dest_train" && rm -f train_rgb_side_clips.zip
    if [[ $? -ne 0 ]]; then echo "Failed to unzip train clips!" >&2; return 1; fi

    echo "Unzipping validation clips..."
    unzip -q val_rgb_side_clips.zip   -d "$dest_val" && rm -f val_rgb_side_clips.zip
    if [[ $? -ne 0 ]]; then echo "Failed to unzip validation clips!" >&2; return 1; fi

    echo "Unzipping test clips..."
    unzip -q test_rgb_side_clips.zip  -d "$dest_test" && rm -f test_rgb_side_clips.zip
    if [[ $? -ne 0 ]]; then echo "Failed to unzip test clips!" >&2; return 1; fi
    echo "Side clips downloaded and extracted."
}

#------------------------- B-F-H 2D Keypoints clips -- Frontal view -------------------------#
rgb_front_2D_keypoints()
{
    local dest_train="./How2Sign/sentence_level/train/rgb_front/features"
    local dest_val="./How2Sign/sentence_level/val/rgb_front/features"
    local dest_test="./How2Sign/sentence_level/test/rgb_front/features"
    mkdir -p "$dest_train" "$dest_val" "$dest_test"

    echo "***** Downloading B-F-H 2D Keypoints clips (Frontal view) files... This might take a while! *****"
    ## Train
    download_gdrive "1TBX7hLraMiiLucknM1mhblNVomO9-Y0r" "train_2D_keypoints.tar.gz" || return 1
    ## Val
    download_gdrive "1JmEsU0GYUD5iVdefMOZpeWa_iYnmK_7w" "val_2D_keypoints.tar.gz" || return 1
    ## Test
    download_gdrive "1g8tzzW5BNPzHXlamuMQOvdwlHRa-29Vp" "test_2D_keypoints.tar.gz" || return 1

    echo "***** Preparing the downloaded files... this might take some time! *****"
    echo "Extracting train keypoints..."
    tar -xf train_2D_keypoints.tar.gz -C "$dest_train" && rm -f train_2D_keypoints.tar.gz
    if [[ $? -ne 0 ]]; then echo "Failed to extract train keypoints!" >&2; return 1; fi

    echo "Extracting validation keypoints..."
    tar -xf val_2D_keypoints.tar.gz   -C "$dest_val" && rm -f val_2D_keypoints.tar.gz
    if [[ $? -ne 0 ]]; then echo "Failed to extract validation keypoints!" >&2; return 1; fi

    echo "Extracting test keypoints..."
    tar -xf test_2D_keypoints.tar.gz  -C "$dest_test" && rm -f test_2D_keypoints.tar.gz
    if [[ $? -ne 0 ]]; then echo "Failed to extract test keypoints!" >&2; return 1; fi
    echo "Frontal 2D keypoints downloaded and extracted."
}

# # B-F-H 2D Keypoints clips -- Side view
# rgb_side_2D_keypoints()
# {
#     # This section remains commented out as in the original script.
#     # If enabled, the wget lines would need converting to gdown too.
#     echo "Creating B-F-H 2D Keypoints clips -- Side view folders"
#     mkdir -p "./How2Sign/sentence_level/train/rgb_side/features/openpose_output"
#     mkdir -p "./How2Sign/sentence_level/val/rgb_side/features/openpose_output"
#     mkdir -p "./How2Sign/sentence_level/test/rgb_side/features/openpose_output"
#     # unzip train_rgb_side_2D_keypoints.zip -d ./How2Sign/sentence_level/train/rgb_side/features
#     # unzip val_rgb_side_2D_keypoints.zip   -d ./How2Sign/sentence_level/val/rgb_side/features
#     # unzip test_rgb_side_2D_keypoints.zip  -d ./How2Sign/sentence_level/test/rgb_side/features
# }

#------------------------- English Translation -------------------------#
english_translation()
{
    local dest_train="./How2Sign/sentence_level/train/text/en/raw_text"
    local dest_val="./How2Sign/sentence_level/val/text/en/raw_text"
    local dest_test="./How2Sign/sentence_level/test/text/en/raw_text"
    mkdir -p "$dest_train" "$dest_val" "$dest_test"

    echo "***** Downloading and preparing the English Translation text files *****"
    ## Train
    download_gdrive "1lq7ksWeD3FzaIwowRbe_BvCmSmOG12-f" "how2sign_train.csv" || return 1
    ## Val
    download_gdrive "1aBQUClTlZB504JtDISJ0DJlbuYUZCGu3" "how2sign_val.csv" || return 1
    ## Test
    download_gdrive "1ScxYnEjILZMn22qKjQj8Wyr_F0nha7kG" "how2sign_test.csv" || return 1

    mv how2sign_train.csv "$dest_train/"
    mv how2sign_val.csv "$dest_val/"
    mv how2sign_test.csv "$dest_test/"
    echo "English translation files downloaded."
}

#------------------------- English Translation re-aligned -------------------------#
english_translation_re-aligned()
{
    local dest_train="./How2Sign/sentence_level/train/text/en/raw_text/re_aligned"
    local dest_val="./How2Sign/sentence_level/val/text/en/raw_text/re_aligned"
    local dest_test="./How2Sign/sentence_level/test/text/en/raw_text/re_aligned"
    mkdir -p "$dest_train" "$dest_val" "$dest_test"

    echo "***** Downloading and preparing the re-aligned English Translation text files *****"
    ## Train
    download_gdrive "1dUHSoefk9OxKJnHrHPX--I4tpm9QD0ok" "how2sign_realigned_train.csv" || return 1
    ## Val
    download_gdrive "1Vpag7VPfdTCCJSao8Pz14rlPfekRMggI" "how2sign_realigned_val.csv" || return 1
    ## Test
    download_gdrive "1AgwBZW26kFHS4CWNMQTCMPGkBPkH3qCu" "how2sign_realigned_test.csv" || return 1

    mv how2sign_realigned_train.csv "$dest_train/"
    mv how2sign_realigned_val.csv "$dest_val/"
    mv how2sign_realigned_test.csv "$dest_test/"
    echo "Re-aligned English translation files downloaded."
}

## TODO
# Gloss annotations
# Panoptic Studio data

# --- Main Execution Logic ---
# Iterate through command-line arguments and call corresponding functions
overall_status=0
for ARG in "$@"
do
    echo "----------------------------------------"
    echo "Processing argument: ${ARG}"
    echo "----------------------------------------"
    # We don't use 'shift' here anymore because we want to process all args passed
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
            echo "${ARG}: Invalid argument given"
            overall_status=1 # Mark as failed if an invalid argument is given
            ;;
    esac

    # Check the exit status of the function called by the case statement
    if [[ $? -ne 0 ]]; then
        echo "!!! ERROR processing '${ARG}' !!!" >&2
        overall_status=1 # Mark overall failure
        # Optionally exit immediately on first error:
        # exit 1
    else
        echo "--- Successfully processed '${ARG}' ---"
    fi
done

echo "========================================"
if [[ $overall_status -eq 0 ]]; then
    echo "All requested downloads and preparations finished successfully."
    echo "Please check the README file for information about the files you just downloaded."
    echo "Feel free to contact amanda.duarte[at]upc.edu if you have any questions about the data."
else
    echo "Some tasks failed. Please check the output above for errors." >&2
fi
echo "========================================"

exit $overall_status
#
################################################################################
