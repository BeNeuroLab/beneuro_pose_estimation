from pathlib import Path
import logging
import shutil
from datetime import datetime
from typing import Optional

if not logging.getLogger().hasHandlers():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

from beneuro_pose_estimation import params, set_logging
from beneuro_pose_estimation.config import _load_config
import cv2
import toml


config = _load_config()

logger = set_logging(__name__)


# ================================= Calibration Registry ====================================

def _load_calibration_registry() -> dict:
    """
    Loads the calibration registry from TOML file.
    Returns an empty dict if registry doesn't exist yet.
    """
    registry_path = config.calibration_local / "calibration_registry.toml"
    if not registry_path.exists():
        logger.info(f"Calibration registry not found at {registry_path}. Starting with empty registry.")
        return {}
    
    try:
        registry = toml.load(registry_path)
        logger.debug(f"Loaded calibration registry with {len(registry)} entries")
        return registry
    except Exception as e:
        logger.error(f"Failed to load calibration registry: {e}")
        return {}


def _save_calibration_registry(registry: dict) -> None:
    """
    Saves the calibration registry to TOML file.
    Creates parent directories if needed.
    """
    registry_path = config.calibration_local / "calibration_registry.toml"
    
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(registry_path, "w") as f:
            toml.dump(registry, f)
        logger.info(f"Saved calibration registry to {registry_path}")
    except Exception as e:
        logger.error(f"Failed to save calibration registry: {e}")
        raise


# ========================= File Sync Utilities ============================

def _copy_file_to_remote(local_path: Path, remote_path: Path) -> bool:
    """
    Attempts to copy a file from local to remote location.
    
    Parameters
    ----------
    local_path : Path
        Path to local file
    remote_path : Path
        Path to remote location
    
    Returns
    -------
    bool
        True if copy succeeded, False otherwise
    """
    if not local_path.exists():
        logger.error(f"Local file not found: {local_path}")
        return False
    
    try:
        remote_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local_path, remote_path)
        logger.info(f"Copied calibration to remote: {remote_path}")
        return True
    except Exception as e:
        logger.warning(f"Could not copy to remote ({remote_path}): {e}")
        return False


def _copy_file_from_remote(remote_path: Path, local_path: Path) -> bool:
    """
    Attempts to copy a file from remote to local.
    
    Parameters
    ----------
    remote_path : Path
        Path to remote file
    local_path : Path
        Path to local location
    
    Returns
    -------
    bool
        True if copy succeeded, False otherwise
    """
    if not remote_path.exists():
        logger.debug(f"Remote file not found: {remote_path}")
        return False
    
    try:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(remote_path, local_path)
        logger.info(f"Downloaded calibration from remote: {remote_path}")
        return True
    except Exception as e:
        logger.warning(f"Could not download from remote ({remote_path}): {e}")
        return False


def _sync_registry_from_remote() -> dict:
    """
    Downloads calibration registry from remote if available.
    Falls back to local registry if remote is not accessible.
    
    Returns
    -------
    dict
        Calibration registry (from remote or local)
    """
    remote_registry_path = config.calibration_remote / "calibration_registry.toml" # Remote path in config
    local_registry_path = config.calibration_local / "calibration_registry.toml"  # Local backup in repo
    
    # Try to download from remote
    if remote_registry_path.exists() and remote_registry_path != local_registry_path:
        try:
            _copy_file_from_remote(remote_registry_path, local_registry_path)
        except Exception as e:
            logger.warning(f"Could not sync registry from remote: {e}")
    else:
        logger.warning("Remote registry not found. Looking for local registry.")
    # Load from local (whether just synced or already there)
    return _load_calibration_registry()


def _sync_registry_to_remote() -> bool:
    """
    Uploads calibration registry to remote.
    
    Returns
    -------
    bool
        True if sync succeeded, False otherwise
    """
    local_registry_path = config.calibration_local / "calibration_registry.toml"
    remote_registry_path = config.calibration_remote / "calibration_registry.toml"  # Same location in remote
    
    if local_registry_path == remote_registry_path:
        logger.debug("Registry already at remote location")
        return True
    
    return _copy_file_to_remote(local_registry_path, remote_registry_path)


def _ensure_calib_file_local(remote_calib_path: Path) -> Path:
    """
    Ensures calibration file is available locally.
    Downloads from remote if not already present.
    
    Parameters
    ----------
    remote_calib_path : Path
        Path to calibration file (remote or already local)
    
    Returns
    -------
    Path
        Path to local copy of calibration file
    """
    # If path is under LOCAL_PATH, it's already local
    try:
        if config.LOCAL_PATH in remote_calib_path.parents or remote_calib_path.parent == config.LOCAL_PATH:
            return remote_calib_path
    except (AttributeError, TypeError):
        pass

    
    local_calib_path = config.calibration_local / remote_calib_path.name
    

    # If already have local copy, return it
    if local_calib_path.exists():
        logger.debug(f"Using cached calibration: {local_calib_path}")
        return local_calib_path
    
    remote_calib_path = config.calibration_remote / remote_calib_path.name
    # Try to download from remote
    if remote_calib_path.exists():
        success = _copy_file_from_remote(remote_calib_path, local_calib_path)
        if success:
            logger.info(f"Downloaded calibration to local cache: {local_calib_path}")
            return local_calib_path
    
    # Fall back to remote if download failed but it exists
    if remote_calib_path.exists():
        logger.warning(f"Using remote calibration path (cache download failed): {remote_calib_path}")
        return remote_calib_path
    
    # Neither local nor remote exists
    logger.error(f"Calibration file not found: {remote_calib_path}")
    return remote_calib_path


# ================================= Calibration Lookup ====================================

def get_calib_for_session(session_name: str) -> Path:
    """
    Looks up the appropriate calibration file for a session based on the registry.
    
    Extracts the session date from session_name and finds a registry entry where:
    - valid_start <= session_date
    - valid_end is empty OR valid_end >= session_date
    
    Falls back to most recent calibration before session date if no registry match.
    
    Parameters
    ----------
    session_name : str
        Session name in format: anything_YYYY_MM_DD_HH_MM (date is parsed from this)
    
    Returns
    -------
    Path
        Path to the calibration .toml file
    """
    from beneuro_pose_estimation.anipose.aniposeTools import get_most_recent_calib
    
    registry = _sync_registry_from_remote()
    
    # Extract session date from session_name
    try:
        session_datetime = datetime.strptime(
            "_".join(session_name.split("_")[1:]), "%Y_%m_%d_%H_%M"
        )
        session_date = session_datetime.date()
    except (ValueError, IndexError) as e:
        logger.error(f"Invalid session format: {session_name}. Cannot extract date: {e}")
        logger.info("Falling back to most recent calibration logic.")
        return get_most_recent_calib(session_name)[1]
    
    # Search registry for matching entry
    matching_entry = None
    for calib_id, entry in registry.items():
        try:
            valid_start = datetime.strptime(entry["valid_start"], "%Y-%m-%d").date()
            valid_end_str = entry.get("valid_end", "")
            valid_end = datetime.strptime(valid_end_str, "%Y-%m-%d").date() if valid_end_str else None
            
            # Check if session_date falls within this calibration's range
            if valid_start <= session_date:
                if valid_end is None or session_date <= valid_end:
                    matching_entry = entry
                    logger.info(f"Found calibration {calib_id} for session {session_name}")
                    break
        except (ValueError, KeyError) as e:
            logger.warning(f"Skipping malformed registry entry {calib_id}: {e}")
            continue
    
    if matching_entry:
        calib_file_path = Path(matching_entry["file_path"])
        # If relative path, make it absolute under calibration directory
        if not calib_file_path.is_absolute():
            calib_file_path = config.calibration_remote / calib_file_path
        return calib_file_path
    
    # No match in registry, fall back to date-based logic
    logger.warning(f"No calibration found in registry for session {session_name}. Using fallback logic.")
    return get_most_recent_calib(session_name)[1]


# ================================= Registration & Generation ====================================

def _register_calibration(calib_folder_name: str, start_date: str, calib_file_path: Path) -> str:
    """
    Registers a new calibration in the registry with remote sync.
    
    Process:
    1. Download registry from remote (if available)
    2. Add new calibration entry
    3. Save locally
    4. Try to copy back to remote (warn if fails)
    
    Parameters
    ----------
    calib_folder_name : str
        Name of the calibration video folder (e.g., "camera_calibration_2026_05_21_18_50")
    start_date : str
        Date when this calibration becomes valid (YYYY-MM-DD format)
    calib_file_path : Path
        Path to the generated .toml calibration file
    
    Returns
    -------
    str
        The ID of the newly registered calibration
    """
    # Download registry from remote (if available)
    registry = _sync_registry_from_remote()
    
    # Generate calibration ID from start_date
    calib_id = f"calibration_{start_date.replace('-', '_')}"
    
    # # Ensure we don't overwrite existing entries
    # counter = 1
    # original_id = calib_id
    # while calib_id in registry:
    #     calib_id = f"{original_id}_{counter}"
    #     counter += 1
    
    # Create registry entry
    registry[calib_id] = {
        "folder_name": calib_folder_name,
        "valid_start": start_date,
        "valid_end": "",  # Empty = no end date (until next calibration)
        "file_path": str(calib_file_path),
        "created": datetime.now().isoformat(),
    }
    
    # Save updated registry locally
    _save_calibration_registry(registry)
    logger.info(f"Registered calibration {calib_id} with valid_start={start_date}")
    
    # Try to sync back to remote
    if not _sync_registry_to_remote():
        logger.warning("[bold yellow]Could not sync registry to remote storage. Registry saved locally only.[/bold yellow]")
    
    return calib_id




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
    project_dir = config.predictions3D / animal / session / f"{session}_pose_estimation"

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
    test_dir = config.LOCAL_PATH /"raw" / animal / session / f"{session}_pose_estimation"/ "tests" 
    
  
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