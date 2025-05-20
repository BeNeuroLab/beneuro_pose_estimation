#!/bin/bash
# Define source and destination directories
src="/mnt/rds/il620/projects/beneuro/live/raw/M061/M061_2025_03_06_14_00/M061_2025_03_06_14_00_cameras"
dst="/mnt/rds/il620/projects/beneuro/live/processed/AnnotationParty/M061/M061_2025_03_06_14_00/M061_2025_03_06_14_00_cameras"


# List of desired camera numbers
for cam in 0 1 2 4 5 6; do
    file="Camera_${cam}.avi"
    echo "Copying ${src}/${file} to ${dst}/${file}"
    cp "${src}/${file}" "${dst}/${file}"
done