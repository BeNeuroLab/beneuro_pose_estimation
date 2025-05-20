import os

# Directory containing the videos
directory = "/mnt/rds/bb2020/projects/beneuro/live/raw/M041/M041_2024_09_04_10_00/M041_2024_09_04_10_00_cameras"

# Session identifier
session_id = "M041_2024_09_04_10_00"

# Rename the files
def rename_videos(directory, session_id):
    for filename in os.listdir(directory):
        if filename.startswith("Camera_") and filename.endswith(".avi"):
            camera_number = filename.split('_')[1].split('.')[0]  # Extract the camera number
            new_name = f"{session_id}_camera_{camera_number}.avi"
            old_path = os.path.join(directory, filename)
            new_path = os.path.join(directory, new_name)
            
            # Rename the file
            os.rename(old_path, new_path)
            print(f"Renamed: {old_path} -> {new_path}")

# Execute the function
rename_videos(directory, session_id)