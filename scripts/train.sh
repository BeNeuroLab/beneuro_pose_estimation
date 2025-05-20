#!/bin/bash

# Define the main directory
MAIN_DIR="/mnt/rds/il620/projects/beneuro/live/processed/AnnotationParty/training"

# Function to run training for a given camera
train_camera() {
    local CAMERA=$1
    echo "Starting training for $CAMERA"
    sleap-train "$MAIN_DIR/$CAMERA/training_config.json" "$MAIN_DIR/$CAMERA/${CAMERA}_.pkg.slp"
    echo "Finished training for $CAMERA"
}

# List of cameras
CAMERAS=("Camera_Front_Left" "Camera_Front_Right" "Camera_Side_Left" "Camera_Top_Left")

# Run training for each camera
for CAMERA in "${CAMERAS[@]}"; do
    train_camera "$CAMERA"
done

echo "All training scripts have been executed."