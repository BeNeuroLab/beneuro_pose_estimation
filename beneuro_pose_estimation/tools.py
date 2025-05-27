from pathlib import Path
import logging
import shutil
if not logging.getLogger().hasHandlers():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

from beneuro_pose_estimation import params, set_logging
from beneuro_pose_estimation.config import _load_config
import cv2
config = _load_config()

logger = set_logging(__name__)

def copy_model_to_remote(test_folder_name: str):
    """
    Copy a model test folder from local config.custom_models to remote config.training
    
    """
    src_root = config.custom_models

    # Find the folder under one of the camera subdirectories
    src_dir = None
    for cam in params.default_cameras:
        candidate = src_root / cam / test_folder_name
        if candidate.is_dir():
            src_dir = candidate
            camera = cam
            break
    if src_dir is None:
        raise FileNotFoundError(f"'{test_folder_name}' not found under any camera in {src_root}")

    # Build destination path: <remote_root>/<camera>/models/<test_folder_name>
    dest_dir = config.training / camera / "models" / test_folder_name

    if dest_dir.exists():
        resp = input(f"Remote folder '{dest_dir}' already exists. Overwrite? (y/N): ").strip().lower()
        if resp != "y":
            logging.info("Aborted. Existing remote model not overwritten.")
            return
        shutil.rmtree(dest_dir)
        logger.info(f"Deleted existing remote folder: {dest_dir}")

    # Perform recursive copy
    shutil.copytree(src_dir, dest_dir)
    logging.info(f"Copied '{src_dir}' → '{dest_dir}'.")

def cleanup_intermediate_files(session: str):
    """
    Clean up intermediate files for a session with interactive prompts.

    1) Delete any '*triangulation*.h5' files under project_dir (including subfolders).
       Uses shutil.os.remove for files.
    2) Prompt to delete the entire 'tests' directory under project_dir, uses shutil.rmtree.
    """
    animal = session.split("_")[0]
    project_dir = config.predictions3D / animal / session / "pose-estimation"

    if not project_dir.exists():
        logger.error(f"Project directory not found: {project_dir}")
        return

    # 1) Clean up triangulation files (searching recursively)
    triangulation_files = list(project_dir.glob("**/*triangulation*.h5"))
    if triangulation_files:
        resp = input(f"\nFound {len(triangulation_files)} triangulation file(s). Delete them? (y/N): ").strip().lower()
        if resp == "y":
            for fpath in triangulation_files:
                try:
                    # shutil.os.remove is the same as os.remove
                    shutil.os.remove(fpath)
                    logger.info(f"Deleted triangulation file: {fpath}")
                except Exception as e:
                    logger.error(f"Error deleting {fpath}: {e}")
        else:
            logger.info("Skipped deleting triangulation files.")
    else:
        logger.info("No triangulation files found.")

    # 2) Clean up 'tests' directory
    tests_dir = project_dir / "tests"
    if tests_dir.is_dir():
        subdirs = [p for p in tests_dir.iterdir() if p.is_dir()]
        if subdirs:
            print(f"\nFound {len(subdirs)} test folder(s) under '{tests_dir}':")
            for sd in subdirs:
                print(f"  • {sd.name}")
            resp = input("Delete the entire 'tests' directory and its contents? (y/N): ").strip().lower()
            if resp == "y":
                try:
                    shutil.rmtree(tests_dir)
                    logger.info(f"Deleted 'tests' directory: {tests_dir}")
                except Exception as e:
                    logger.error(f"Failed to delete tests directory {tests_dir}: {e}")
            else:
                logger.info("Skipped deleting 'tests' directory.")
        else:
            logger.info(f"'tests' directory exists but has no subfolders: {tests_dir}")
    else:
        logger.info("No 'tests' directory found.")

    logger.info("Cleanup completed.")


def create_test_videos(session, cameras=params.default_cameras, duration_seconds=10, fps=100, 
                      force_new=False, start_frame=None):
    """
    Creates short test videos and corresponding metadata for each camera.
    """
    animal = session.split("_")[0]
    n_frames = duration_seconds * fps 
    
    # Create output directory for test videos
    test_dir = config.LOCAL_PATH /"raw" / animal / session / "pose-estimation"/ "tests" 
    
  
    cameras_dir = test_dir / f"{session}_cameras"
    cameras_dir.mkdir(parents=True, exist_ok=True)
    for camera in cameras:
        try:
            # Output video path
            output_video = cameras_dir / f"{params.camera_name_mapping.get(camera, camera)}.avi"
            
            # Skip if video exists and force_new is False
            if output_video.exists() and not force_new:
                logger.info(f"Test video already exists for {camera}, skipping: {output_video}")
                continue
                
            # Input video path
            input_video = (
                config.recordings
                / animal
                / session
                / f"{session}_cameras"
                / f"{params.camera_name_mapping.get(camera, camera)}.avi"
            )
            
            if not input_video.exists():
                logger.warning(f"Input video not found: {input_video}")
                continue
                
            
            # Create video
            cap = cv2.VideoCapture(str(input_video))
            if not cap.isOpened():
                logger.error(f"Could not open video: {input_video}")
                continue
                
            # Get video properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(str(output_video), fourcc, fps, (width, height))
            
            # Set start frame if specified
            if start_frame is not None:
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            # Read and write frames
            frame_count = 0
            while frame_count < n_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                out.write(frame)
                frame_count += 1
                
            # Release resources
            cap.release()
            out.release()
            
            logger.info(f"Created test video for {camera}: {output_video}")
            
        except Exception as e:
            logger.error(f"Error processing camera {camera}: {e}")
    
    return test_dir