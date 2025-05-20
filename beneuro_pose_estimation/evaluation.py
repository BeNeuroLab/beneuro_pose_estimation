"""
Module for evaluating 3D pose estimation results
TODO: 
- change input session_dir to session_name
- decide on how to use the evaluation metrics
- combine the 2d and 3d metrics
"""
import matplotlib.cm as cm
import matplotlib
# matplotlib.use("Qt5Agg") 
from matplotlib.animation import FuncAnimation
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from aniposelib.cameras import CameraGroup
import h5py
from beneuro_pose_estimation import params
from beneuro_pose_estimation.config import _load_config
import cv2
import sleap_anipose as slap
from beneuro_pose_estimation.anipose import aniposeTools
import json
import seaborn as sns
import sleap
config = _load_config()
logger = logging.getLogger(__name__)

def load_3d_predictions(session_name,test_dir):
    """Load 3D predictions from h5 file"""
    h5_file = test_dir / f"{session_name}_pose_estimation_combined.h5"
    if not h5_file.exists():
        logger.error(f"3D predictions file not found: {h5_file}")
        return None
    with h5py.File(h5_file, "r") as f:
        points3d = f["points3d"][:]
    return points3d

def load_2d_predictions(session_name, pred_dir, cameras=params.default_cameras):
    """
    Load 2D predictions for all cameras

    TODO: remove min frames truncation 
    
    """
    predictions_list = []
    camera_data = {}
    
    # First load all predictions
    for camera in cameras:
        h5_file = pred_dir / camera / f"{session_name}_{camera}.analysis.h5"
        if not h5_file.exists():
            logger.warning(f"2D predictions file not found for camera {camera}")
            continue
        with h5py.File(h5_file, "r") as f:
            # Shape of 'tracks': (1, 2, n_nodes, n_frames)
            tracks = f["tracks"][:]  # Read the data into memory
            logger.debug(f"Camera {camera} initial tracks shape: {tracks.shape}")
            camera_data[camera] = {
                'predictions': tracks
            }
    
    # Find the minimum number of frames across all cameras (dimension 3)
    min_frames = min(data['predictions'].shape[3] for data in camera_data.values())
    logger.debug(f"\nMinimum number of frames across all cameras: {min_frames}")
    
    # Keep only the first min_frames frames from each camera
    for camera, data in camera_data.items():
        # Take the first min_frames frames
        predictions_truncated = data['predictions'][:, :, :, :min_frames]
        logger.debug(f"Camera {camera} truncated shape: {predictions_truncated.shape}")
        predictions_list.append(predictions_truncated)
    
    # Combine predictions from all cameras along a new first dimension (n_cams)
    stacked = np.stack(predictions_list, axis=0)
    logger.debug(f"Final stacked shape: {stacked.shape}")
    return stacked

def load_2d_predictions_old(session_dir, cameras=params.default_cameras):
    """Load 2D predictions for all cameras"""
    session_name = session_dir.parent.name
    predictions_list = []
    camera_data = {}
    
    # First load all predictions
    for camera in cameras:
        h5_file = session_dir / camera / f"{session_name}_{camera}.analysis.h5"
        if not h5_file.exists():
            logger.warning(f"2D predictions file not found for camera {camera}")
            continue
        with h5py.File(h5_file, "r") as f:
            # Shape of 'tracks': (1, 2, n_nodes, n_frames)
            tracks = f["tracks"][:]  # Read the data into memory
            logger.info(f"Camera {camera} initial tracks shape: {tracks.shape}")
            camera_data[camera] = {
                'predictions': tracks
            }
    
    # Find the minimum number of frames across all cameras (dimension 3)
    min_frames = min(data['predictions'].shape[3] for data in camera_data.values())
    logger.info(f"\nMinimum number of frames across all cameras: {min_frames}")
    
    # Keep only the first min_frames frames from each camera
    for camera, data in camera_data.items():
        # Take the first min_frames frames
        predictions_truncated = data['predictions'][:, :, :, :min_frames]
        logger.info(f"Camera {camera} truncated shape: {predictions_truncated.shape}")
        predictions_list.append(predictions_truncated)
    
    # Combine predictions from all cameras along a new first dimension (n_cams)
    stacked = np.stack(predictions_list, axis=0)
    logger.info(f"Final stacked shape: {stacked.shape}")
    return stacked

def calculate_reprojection_errors(session_name, test_dir, cameras=params.default_cameras):
    """
    Calculate reprojection errors between 3D predictions and 2D data.
    Returns dict with per_camera, per_keypoint, and overall stats.
    """
    logger.info(f"Analyzing session: {session_name}")

    # 1) Load 2D predictions
    preds2d = load_2d_predictions(session_name, test_dir, cameras)
    if preds2d is None:
        return None

    # 2) Load raw 3D predictions
    points3d_raw = load_3d_predictions(session_name, test_dir)
    if points3d_raw is None:
        return None

    # Handle both 3-D and 4-D shapes
    if points3d_raw.ndim == 3:
        # (n_frames, n_kp, 3) → add track dim
        n_frames, n_kp, _ = points3d_raw.shape
        points3d_4d = points3d_raw[:, None, :, :]  # → (n_frames,1,n_kp,3)
    elif points3d_raw.ndim == 4:
        n_frames, n_tracks, n_kp, _ = points3d_raw.shape
        points3d_4d = points3d_raw
    else:
        raise ValueError(f"Unexpected points3d shape {points3d_raw.shape}")
    # logger.info(f"points3d_4d shape: {points3d_4d.shape}")

    # We’ll compare using the first track:
    points3d = points3d_4d[:, 0, :, :]  # (n_frames, n_kp, 3)

    # 3) Load calibration
    if session_name.split("_")[1] == "2025":
        calib_file = config.calibration / "calibration_2025_03_12_11_45.toml"
    else:
        calib_file = aniposeTools.get_most_recent_calib(session_name)
    if not calib_file or not calib_file.exists():
        logger.error(f"Missing calibration for {session_name}")
        return None

    # 4) Reproject all tracks back into each camera
    reproj_file = test_dir / f"{session_name}_reprojections.h5"
    slap.reproject(
        p3d=points3d_4d,                 # 4-D array as required
        calib=str(calib_file),
        frames=(0, n_frames),
        fname=str(reproj_file),
    )

    # 5) Load per-camera reprojections
    reproj = {}
    with h5py.File(reproj_file, "r") as f:
        for cam in cameras:
            arr = f[cam][:]
            # Squeeze any extra singleton dims beyond the last two
            while arr.ndim > 3 and 1 in arr.shape[:-2]:
                arr = arr.squeeze(arr.shape.index(1))
            # If shape is (n_kp, n_frames, 2), transpose
            if arr.shape[0] == n_kp and arr.shape[1] == n_frames:
                arr = arr.transpose(1, 0, 2)
            assert arr.shape == (n_frames, n_kp, 2), f"Bad reproj shape for {cam}: {arr.shape}"
            reproj[cam] = arr
            # logger.info(f"Reproj[{cam}] shape: {arr.shape}")

    # 6) Compute per-camera errors
    errors = {}
    for i, cam in enumerate(cameras):
        # preds2d[i]: (n_tracks,2,n_kp,n_frames)
        pred2d = preds2d[i, 0]                   # (2,n_kp,n_frames)
        pred_xy = np.transpose(pred2d, (2, 1, 0))  # → (n_frames,n_kp,2)
        err = np.linalg.norm(reproj[cam] - pred_xy, axis=-1)  # (n_frames,n_kp)
        errors[cam] = err
        # logger.info(f"errors[{cam}] shape: {err.shape}")

    # 7) Stack and summarize
    all_err     = np.stack(list(errors.values()), axis=0)  # (n_cams,n_frames,n_kp)
    mean_per_kp = np.nanmean(all_err, axis=(0, 1))       # (n_kp,)
    mean_per_cam= np.nanmean(all_err, axis=(1, 2))       # (n_cams,)
    flat        = all_err.flatten()
    mean_all    = float(np.nanmean(flat))
    median_all  = float(np.nanmedian(flat))
    std_all     = float(np.nanstd(flat))

    # 8) Log results
    print("Mean reprojection error per keypoint (px):")
    for idx, kp in enumerate(params.body_parts):
        print(f"  {kp}: {mean_per_kp[idx]:.3f}")

    print("Mean reprojection error per camera (px):")
    for idx, cam in enumerate(cameras):
        print(f"  {cam}: {mean_per_cam[idx]:.3f}")

    print("Overall error stats (px):")
    print(f"  Mean   : {mean_all:.3f}")
    print(f"  Median : {median_all:.3f}")
    print(f"  Std    : {std_all:.3f}")

    return {
        "per_camera": mean_per_cam,
        "per_keypoint": mean_per_kp,
        "overall": {"mean": mean_all, "median": median_all, "std": std_all},
    }

def get_reprojection_errors_array(session_name, test_dir, cameras=params.default_cameras):
    """
    Returns a NumPy array of shape (n_cameras, n_frames, n_keypoints) containing
    reprojection errors for each camera, frame, and keypoint.
    """
    # Load data
    preds2d = load_2d_predictions(session_name, test_dir, cameras)
    points3d = load_3d_predictions(session_name, test_dir)
    
    # Ensure a 4D array for triangulation: (n_frames, n_tracks, n_kp, 3)
    if points3d.ndim == 3:
        points3d_4d = points3d[:, None, :, :]
    else:
        points3d_4d = points3d
    n_frames = points3d_4d.shape[0]
    
    # Load calibration
    if session_name.split("_")[1] == "2025":
        calib_file = config.calibration / "calibration_2025_03_12_11_45.toml"
    else:
        calib_file = aniposeTools.get_most_recent_calib(session_name)
    
    # Triangulate (reproject)
    reproj_file = Path(test_dir) / f"{session_name}_reproj_temp.h5"
    slap.reproject(
        p3d=points3d_4d,
        calib=str(calib_file),
        frames=(0, n_frames),
        fname=str(reproj_file),
    )
    
    # Load reprojections
    reproj = []
    with h5py.File(reproj_file, "r") as f:
        for cam in cameras:
            arr = f[cam][:]
            # Squeeze any extra singleton dims before last two
            while arr.ndim > 3 and 1 in arr.shape[:-2]:
                arr = arr.squeeze(axis=arr.shape.index(1))
            # If dims are (n_kp, n_frames, 2), transpose
            if arr.shape[0] == len(params.body_parts) and arr.shape[1] == n_frames:
                arr = arr.transpose(1, 0, 2)
            reproj.append(arr)
    
    # Compute errors
    errors = []
    for i, cam in enumerate(cameras):
        pred2d = preds2d[i, 0]                      # (2, n_kp, n_frames)
        pred_xy = np.transpose(pred2d, (2, 1, 0))   # (n_frames, n_kp, 2)
        err = np.linalg.norm(reproj[i] - pred_xy, axis=-1)  # (n_frames, n_kp)
        errors.append(err)
    
    return np.stack(errors, axis=0)  # (n_cameras, n_frames, n_kp)

import math

def plot_reprojection_error_histograms(session_name, test_dir, bins=50):
    """
    Plot histograms of 3D reprojection errors:
      - Per camera (shared axes)
      - Per keypoint (shared axes)

    Args:
        session_name (str): Name of the session.
        test_dir (Path): Directory where reprojection and predictions are stored.
        bins (int): Number of bins for histograms.

    Returns:
        tuple: (fig_cam, fig_kp) Matplotlib Figure objects.
    """
    # Load errors
    all_errors = get_reprojection_errors_array(session_name, test_dir)
    cameras   = params.default_cameras
    keypoints = params.body_parts

    # --- Shared binning & axis limits for cameras ---
    cam_data   = [all_errors[i].flatten()[~np.isnan(all_errors[i].flatten())] for i in range(len(cameras))]
    cam_xmax   = np.percentile(np.concatenate(cam_data), 99)
    cam_bins   = np.linspace(0, cam_xmax, bins + 1)
    cam_counts = [np.histogram(d, bins=cam_bins)[0] for d in cam_data]
    cam_ymax   = max(c.max() for c in cam_counts)

    # Per-camera: 3 per row
    n_cam = len(cameras)
    cols_cam = 3
    rows_cam = math.ceil(n_cam / cols_cam)
    fig_cam, axes_cam = plt.subplots(rows_cam, cols_cam,
                                     figsize=(5 * cols_cam, 4 * rows_cam),
                                     constrained_layout=True)
    axes_cam = axes_cam.flatten()
    for i, cam in enumerate(cameras):
        axes_cam[i].hist(cam_data[i], bins=cam_bins, edgecolor='black')
        axes_cam[i].set_title(cam, fontsize=14)
        axes_cam[i].set_xlim(0, cam_xmax)
        axes_cam[i].set_ylim(0, cam_ymax)
        axes_cam[i].set_xlabel('Error (px)')
        axes_cam[i].set_ylabel('Count')
    for ax in axes_cam[n_cam:]:
        ax.axis('off')
    fig_cam.suptitle('Reprojection Error per Camera', fontsize=18, y=1.02)

    # --- Shared binning & axis limits for keypoints ---
    kp_data   = [all_errors[:, :, k].flatten()[~np.isnan(all_errors[:, :, k].flatten())]
                 for k in range(len(keypoints))]
    kp_xmax   = np.percentile(np.concatenate(kp_data), 99)
    kp_bins   = np.linspace(0, kp_xmax, bins + 1)
    kp_counts = [np.histogram(d, bins=kp_bins)[0] for d in kp_data]
    kp_ymax   = max(c.max() for c in kp_counts)

    # Per-keypoint: 3 per row
    n_kp = len(keypoints)
    cols_kp = 3
    rows_kp = math.ceil(n_kp / cols_kp)
    fig_kp, axes_kp = plt.subplots(rows_kp, cols_kp,
                                   figsize=(5 * cols_kp, 4 * rows_kp),
                                   constrained_layout=True)
    axes_kp = axes_kp.flatten()
    for idx, kp in enumerate(keypoints):
        axes_kp[idx].hist(kp_data[idx], bins=kp_bins, edgecolor='black')
        axes_kp[idx].set_title(kp, fontsize=14)
        axes_kp[idx].set_xlim(0, kp_xmax)
        axes_kp[idx].set_ylim(0, kp_ymax)
        axes_kp[idx].set_xlabel('Error (px)')
        axes_kp[idx].set_ylabel('Count')
    for ax in axes_kp[n_kp:]:
        ax.axis('off')
    fig_kp.suptitle('Reprojection Error per Keypoint', fontsize=18, y=1.02)

    plt.show()
    return fig_cam, fig_kp
def plot_reprojection_error_per_camera(session_name, test_dir, bins=50):
    """
    Overlay reprojection‐error histograms for each camera on one plot.
    """
    # Get the errors array: shape (n_cameras, n_frames, n_keypoints)
    errors = get_reprojection_errors_array(session_name, test_dir)
    
    plt.figure(figsize=(8, 6))
    for i, cam in enumerate(params.default_cameras):
        errs = errors[i].flatten()
        errs = errs[~np.isnan(errs)]
        # density=True to normalize, alpha for translucency
        plt.hist(
            errs,
            bins=bins,
            density=True,
            alpha=0.4,
            label=cam,
            edgecolor='none'
        )
    # zoom into the bulk of the distribution
    upper = np.percentile(errors.flatten()[~np.isnan(errors.flatten())], 99)
    plt.xlim(0, upper)
    
    plt.title("Overlayed Reprojection Error Distributions by Camera")
    plt.xlabel("Error (pixels)")
    plt.ylabel("Probability Density")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

def plot_reprojection_error_per_keypoint(session_name, test_dir, bins=50):
    """
    Overlay reprojection‐error histograms for each keypoint on one plot,
    with a distinct color per keypoint.
    """
    # Get the errors array: shape (n_cameras, n_frames, n_keypoints)
    errors = get_reprojection_errors_array(session_name, test_dir)  # (cams, frames, kps)
    n_kp = len(params.body_parts)

    # Choose a qualitative colormap with enough distinct colors
    cmap = plt.get_cmap("tab20")
    colors = [cmap(i % cmap.N) for i in range(n_kp)]

    plt.figure(figsize=(10, 6))
    for kp_idx, kp_name in enumerate(params.body_parts):
        # Flatten and drop NaNs
        errs = errors[:, :, kp_idx].ravel()
        errs = errs[~np.isnan(errs)]

        plt.hist(
            errs,
            bins=bins,
            density=True,
            alpha=0.5,
            label=kp_name,
            color=colors[kp_idx],
            edgecolor="black",
            linewidth=0.5,
        )

    # Zoom into the bulk of the distribution (99th percentile)
    flat = errors.ravel()
    flat = flat[~np.isnan(flat)]
    upper = np.percentile(flat, 99)
    plt.xlim(0, upper)

    plt.title("Overlayed Reprojection Error Distributions by Keypoint", fontsize=16)
    plt.xlabel("Error (pixels)", fontsize=14)
    plt.ylabel("Probability Density", fontsize=14)
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=10)
    plt.tight_layout()
    plt.show()

    return plt.gcf()

def plot_keypoint_errors_by_camera(session_name, test_dir, camera, bins=50):
    """
    Plot histograms of reprojection errors per keypoint for a specified camera.

    Args:
        session_name (str): Session identifier.
        test_dir (Path or str): Directory containing reprojection file.
        camera (str): Name of the camera to plot.
        bins (int): Number of bins for histograms.

    Returns:
        matplotlib.figure.Figure: Figure with subplots per keypoint.
    """
    # Get full error array: (n_cameras, n_frames, n_keypoints)
    all_errors = get_reprojection_errors_array(session_name, test_dir)
    cameras = params.default_cameras

    if camera not in cameras:
        raise ValueError(f"Camera '{camera}' not found; choose from {cameras}")

    cam_idx = cameras.index(camera)
    errors_cam = all_errors[cam_idx]  # shape (n_frames, n_keypoints)

    keypoints = params.body_parts
    n_kp = len(keypoints)
    n_cols = 5
    n_rows = math.ceil(n_kp / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows), constrained_layout=True)
    axes = axes.flatten()

    for idx, kp in enumerate(keypoints):
        errs = errors_cam[:, idx]
        errs = errs[~np.isnan(errs)]
        axes[idx].hist(errs, bins=bins, edgecolor='black')
        axes[idx].set_title(kp)
        # zoom to 99th percentile to focus on bulk
        if errs.size > 0:
            upper = np.percentile(errs, 99)
            axes[idx].set_xlim(0, upper)
        axes[idx].set_xlabel('Error (px)')
        axes[idx].set_ylabel('Count')

    # Turn off unused axes
    for j in range(n_kp, len(axes)):
        axes[j].axis('off')

    fig.suptitle(f"Reprojection Error per Keypoint\nCamera: {camera}", fontsize=16)
    return fig

def get_model_path_from_slp(slp_pred_path):
    """
    Returns the first model path recorded in the provenance of an SLP predictions file.
    """
    slp_pred_path = Path(slp_pred_path)
    with h5py.File(slp_pred_path, "r") as f:
        # The metadata JSON blob is stored as an attribute on the metadata group
        meta_json = f["metadata"].attrs["json"]
    meta = json.loads(meta_json.decode("utf-8"))
    model_paths = meta.get("provenance", {}).get("model_paths", [])
    if not model_paths:
        raise ValueError("No model_paths found in SLP metadata")
    return Path(model_paths[0])


def load_2d_confidence_scores(h5_file):
    """Load confidence scores from a 2D predictions HDF5 file
    
    Args:
        h5_file: Path to the HDF5 file
        
    Returns:
        np.ndarray: Confidence scores with shape (n_tracks, n_keypoints, n_frames)
    """
    with h5py.File(h5_file, "r") as f:
        scores = f["point_scores"][:]
        
        # Ensure we have the correct shape (n_tracks, n_keypoints, n_frames)
        if len(scores.shape) == 1:
            scores = scores.reshape(1, 1, -1)  # Single track, single keypoint
        elif len(scores.shape) == 2:
            scores = scores.reshape(1, scores.shape[0], scores.shape[1])  # Single track, multiple keypoints
        elif len(scores.shape) == 3:
            # Already in correct shape (n_tracks, n_keypoints, n_frames)
            pass
        else:
            logger.error(f"Unexpected scores shape: {scores.shape}")
            return None
            
    return scores

def get_2d_confidence_scores(session_name,test_dir, cameras=params.default_cameras):
    """Extract confidence scores from 2D predictions, keeping only the minimum number of frames
    
    Args:
        session_dir: Directory containing the session data
        cameras: List of cameras to process
        
    Returns:
        np.ndarray: Stacked confidence scores with shape (n_cameras, n_tracks, n_keypoints, n_frames)
    """
    logger.info(f"\nAnalyzing session: {session_name}")
    
    # First load scores for all cameras
    camera_data = {}
    for camera in cameras:
        # Load confidence scores
        h5_file = test_dir / camera / f"{session_name}_{camera}.analysis.h5"
        if not h5_file.exists():
            logger.warning(f"2D predictions file not found for camera {camera}")
            continue
            
        scores = load_2d_confidence_scores(h5_file)
        if scores is not None:
            camera_data[camera] = {
                'scores': scores
            }
    
    # Find the minimum number of frames across all cameras
    min_frames = min(data['scores'].shape[-1] for data in camera_data.values())
    logger.info(f"\nMinimum number of frames across all cameras: {min_frames}")
    
    # Keep only the first min_frames frames from each camera
    aligned_scores = []
    for camera, data in camera_data.items():
        # Take the first min_frames frames
        scores_truncated = data['scores'][..., :min_frames]
        aligned_scores.append(scores_truncated)
    
    # Stack the aligned scores
    stacked_scores = np.stack(aligned_scores, axis=0)
    
    
    return stacked_scores



def visualize_confidence_scores(session_name, test_dir):
    """
    More informative visualizations for 2D confidence scores:
      1. Boxplot per camera
      2. Heatmap of mean confidence per (camera x keypoint)
    """
    # Load stacked scores: shape (n_cameras, n_tracks, n_keypoints, n_frames)
    # Load and prepare data
    confs = get_2d_confidence_scores(session_name, test_dir, cameras=params.default_cameras)
    n_cam, _, n_kp, _ = confs.shape

    # Prepare per-camera arrays, clipped to [0,1]
    camera_data = []
    for i in range(n_cam):
        arr = confs[i].flatten()
        arr = arr[~np.isnan(arr)]
        arr = np.clip(arr, 0, 1)
        camera_data.append(arr)

    
    mean_confidence_per_keypoint = np.nanmean(confs, axis=(0, 1, 3))  # Average over cameras, tracks, and frames
    print("\nMean confidence per keypoint:")
    for i, body_part in enumerate(params.body_parts):
        print(f"  {body_part}: {mean_confidence_per_keypoint[i]:.3f}")
    
    # Calculate and print mean confidence per camera
    mean_confidence_per_camera = np.nanmean(confs, axis=(1, 2, 3))  # Average over tracks, keypoints, and frames
    print("\nMean confidence per camera:")
    for i, camera in enumerate(params.default_cameras):
        print(f"  {camera}: {mean_confidence_per_camera[i]:.3f}")
    n_cam = confs.shape[0]
    n_kp = confs.shape[2]

    # Prepare per-camera arrays
    camera_data = []
    for i in range(n_cam):
        flat = confs[i].flatten()
        camera_data.append(flat[~np.isnan(flat)])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), dpi=100)

    # === 1) Violin plot per camera ===
    parts = ax1.violinplot(
        camera_data,
        showmeans=True,
        showmedians=True
    )
    ax1.set_title("Confidence Score Distribution per Camera")
    ax1.set_ylabel("Confidence Score")
    ax1.set_xticks(np.arange(1, n_cam+1))
    ax1.set_xticklabels(params.default_cameras, rotation=45, ha='right')
    # 2) Heatmap of mean confidence per camera/keypoint
    mean_matrix = np.nanmean(confs, axis=(1,3))  # shape (n_cameras, n_keypoints)
    im = ax2.imshow(
        mean_matrix,
        aspect='auto',
        vmin=0,
        vmax=1,
        cmap='viridis'        # explicit colormap
    )
    ax2.set_title("Mean Confidence per Camera and Keypoint")
    ax2.set_xlabel("Keypoint")
    ax2.set_ylabel("Camera")
    ax2.set_xticks(np.arange(n_kp))
    ax2.set_xticklabels(params.body_parts, rotation=90, ha='center')
    ax2.set_yticks(np.arange(n_cam))
    ax2.set_yticklabels(params.default_cameras)
    cbar = plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    cbar.set_label("Mean Confidence")

    plt.tight_layout()
    plt.show()
def count_cameras_used(predictions_2d, confidence_threshold=0.5):
    """Count how many cameras were used for each keypoint in each frame"""
    # Assuming predictions_2d has shape (n_cameras, n_frames, n_keypoints, 2)
    # and confidence scores have shape (n_cameras, n_frames, n_keypoints)
    confidence_scores = get_2d_confidence_scores(predictions_2d)
    cameras_used = (confidence_scores > confidence_threshold).sum(axis=0)
    return cameras_used

def check_physical_constraints(points3d, fps=30):
    """Check physical constraints on 3D predictions and return validity information
    
    Returns:
        dict: Dictionary containing:
            - joint_angles: Dictionary of joint angles over time
            - max_velocity: Maximum velocity between consecutive frames
            - max_acceleration: Maximum acceleration between consecutive frames
            - valid_timepoints: Boolean array indicating which timepoints are valid
            - invalid_reasons: Dictionary mapping timepoint indices to lists of reasons for invalidity
    """
    results = {}
    
    # Initialize arrays for tracking validity
    n_frames = points3d.shape[0]
    valid_timepoints = np.ones(n_frames, dtype=bool)
    invalid_reasons = {i: [] for i in range(n_frames)}
    
    # Calculate joint angles
    angles = {}
    for angle_name, points in params.angles.items():
        p1, p2, p3 = [params.keypoints_dict[p] for p in points]
        v1 = points3d[:, p1] - points3d[:, p2]
        v2 = points3d[:, p3] - points3d[:, p2]
        angle = np.arccos(np.sum(v1 * v2, axis=1) / (np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1)))
        angles[angle_name] = angle
        
        # Check if angles are within reasonable limits
        invalid_angles = (angle < params.evaluation_thresholds["min_angle"]) | (angle > params.evaluation_thresholds["max_angle"])
        valid_timepoints[invalid_angles] = False
        for frame in np.where(invalid_angles)[0]:
            invalid_reasons[frame].append(f"Invalid {angle_name} angle: {angle[frame]:.2f} radians")
    
    # Check for sudden movements
    velocities = np.diff(points3d, axis=0)
    accelerations = np.diff(velocities, axis=0)
    
    # Calculate velocity and acceleration magnitudes
    velocity_magnitudes = np.linalg.norm(velocities, axis=-1)
    acceleration_magnitudes = np.linalg.norm(accelerations, axis=-1)
    
    # Check velocities
    invalid_velocities = velocity_magnitudes > params.evaluation_thresholds["max_velocity"]
    valid_timepoints[1:][invalid_velocities] = False  # Note: velocities start at frame 1
    for frame in np.where(invalid_velocities)[0]:
        invalid_reasons[frame + 1].append(f"High velocity: {velocity_magnitudes[frame]:.2f} mm/frame")
    
    # Check accelerations
    invalid_accelerations = acceleration_magnitudes > params.evaluation_thresholds["max_acceleration"]
    valid_timepoints[2:][invalid_accelerations] = False  # Note: accelerations start at frame 2
    for frame in np.where(invalid_accelerations)[0]:
        invalid_reasons[frame + 2].append(f"High acceleration: {acceleration_magnitudes[frame]:.2f} mm/frame^2")
    
    # Check for missing or NaN values
    invalid_nans = np.any(np.isnan(points3d), axis=(1, 2))
    valid_timepoints[invalid_nans] = False
    for frame in np.where(invalid_nans)[0]:
        invalid_reasons[frame].append("Missing or NaN values")
    
    results["joint_angles"] = angles
    results["max_velocity"] = np.max(velocity_magnitudes)
    results["max_acceleration"] = np.max(acceleration_magnitudes)
    results["valid_timepoints"] = valid_timepoints
    results["invalid_reasons"] = invalid_reasons
    
    return results

def save_detailed_metrics_to_csv(session_dir, output_dir=None):
    """Save detailed per-frame, per-keypoint metrics to CSV"""
    if output_dir is None:
        output_dir = session_dir / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    h5_file = session_dir / f"{session_dir.name}_pose_estimation_combined.h5"
    points3d = load_3d_predictions(h5_file)
    predictions_2d = load_2d_predictions(session_dir)
    calib_file = aniposeTools.get_most_recent_calib(session_dir.name)
    cgroup = CameraGroup.load(calib_file)
    
    # Calculate metrics
    reprojection_errors = calculate_reprojection_errors(session_dir)
    confidence_scores = get_2d_confidence_scores(session_dir)
    
    # Get number of frames and keypoints
    n_frames = points3d.shape[0]
    n_keypoints = points3d.shape[1]
    n_cameras = len(params.default_cameras)
    
    # Create DataFrame for detailed metrics
    rows = []
    for frame in range(n_frames):
        for kp_idx, kp_name in enumerate(params.keypoints):
            # Get cameras that were used for this keypoint in this frame
            used_cameras = [i for i in range(n_cameras) if confidence_scores[i, frame, kp_idx] > 0.5]
            
            if used_cameras:  # Only include if at least one camera was used
                mean_confidence = np.mean([confidence_scores[i, frame, kp_idx] for i in used_cameras])
                
                # Create row with frame, keypoint, and metrics
                row = {
                    'frame': frame,
                    'keypoint': kp_name,
                    'mean_confidence': mean_confidence,
                    'n_cameras_used': len(used_cameras)
                }
                
                # Add reprojection errors for each camera
                for cam_idx, cam_name in enumerate(params.default_cameras):
                    if cam_idx in used_cameras:
                        row[f'reprojection_error_{cam_name}'] = reprojection_errors[cam_name][frame, kp_idx]
                    else:
                        row[f'reprojection_error_{cam_name}'] = np.nan
                
                rows.append(row)
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "detailed_metrics.csv", index=False)
    
    return df

def rotate_points(points, angle, axis="z"):
    """
    Rotate a set of 3D points around the specified axis by the given angle.
    
    Args:
        points (np.ndarray): Array of points with shape (..., 3).
        angle (float): Rotation angle in radians.
        axis (str): Axis to rotate around ("x", "y", or "z").
        
    Returns:
        np.ndarray: Rotated points.
    """
    c, s = np.cos(angle), np.sin(angle)
    if axis.lower() == "z":
        R = np.array([[c, -s, 0],
                      [s,  c, 0],
                      [0,  0, 1]])
    elif axis.lower() == "x":
        R = np.array([[1, 0, 0],
                      [0, c, -s],
                      [0, s,  c]])
    elif axis.lower() == "y":
        R = np.array([[ c, 0, s],
                      [ 0, 1, 0],
                      [-s, 0, c]])
    else:
        raise ValueError(f"Axis {axis} not supported. Choose from 'x', 'y', or 'z'.")
    
    rotated_points = points @ R.T
    return rotated_points

def create_3d_animation_from_csv(csv_filepath, output_dir, fps=30, start_frame=None, end_frame=None,
                                 body_parts=None, constraints=None,
                                 rotation_angle=0.0, rotation_axis="z"):
    """
    Create a 3D animation from a CSV file with 3D keypoint data and save the video.
    
    Expected CSV format:
        - A column "fnum" (frame number)
        - For each keypoint, columns with suffixes _x, _y, _z (e.g., shoulder_center_x, shoulder_center_y, shoulder_center_z).
    
    If body_parts or constraints are not provided, the function defaults to using values in params.py.
    
    Parameters:
        csv_filepath (str or Path): Path to the CSV file.
        output_dir (str or Path): Directory to save the output video.
        fps (int): Frames per second for the video.
        start_frame (int): Starting frame index. Defaults to 0.
        end_frame (int): Ending frame index (exclusive). Defaults to number of frames in CSV.
        body_parts (list): List of keypoint names. Defaults to params.body_parts.
        constraints (list of [int, int]): Skeleton connections. Defaults to params.constraints.
        rotation_angle (float): Angle (in radians) by which to rotate the coordinate system.
        rotation_axis (str): Axis to rotate about ("x", "y", or "z").
    
    Returns:
        anim: The matplotlib.animation.FuncAnimation object.
    """
    # Use default parameters from params if not provided.
    if body_parts is None:
        body_parts = params.body_parts
    if constraints is None:
        constraints = params.constraints

    # Read CSV; adjust sep if needed (here assumed comma-separated).
    csv_filepath = Path(csv_filepath)
    df = pd.read_csv(csv_filepath, sep=",")  # Change sep here if necessary.
    # Optionally, clean headers (e.g., strip whitespace and lower case)
    df.columns = df.columns.str.strip().str.lower()

    n_frames = len(df)
    n_keypoints = len(body_parts)

    # Build the points3d array (shape: n_frames x n_keypoints x 3)
    points3d = np.empty((n_frames, n_keypoints, 3))
    for i, part in enumerate(body_parts):
        # use lower case to match cleaned column names
        part = part.lower()
        col_x = f"{part}_x"
        col_y = f"{part}_y"
        col_z = f"{part}_z"
        if col_x not in df.columns or col_y not in df.columns or col_z not in df.columns:
            raise ValueError(f"Expected columns for '{part}' not found in CSV. Available: {df.columns.tolist()}")
        points3d[:, i, 0] = df[col_x].values
        points3d[:, i, 1] = df[col_y].values
        points3d[:, i, 2] = df[col_z].values

    # Apply rotation if requested.
    if rotation_angle != 0:
        points3d = rotate_points(points3d, rotation_angle, axis=rotation_axis)

    # Set frame range.
    if start_frame is None:
        start_frame = 0
    if end_frame is None:
        end_frame = n_frames

    # Setup output directory.
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create a figure and 3D axis.
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # Set axis limits based on non-NaN data.
    flat_points = points3d.reshape(-1, 3)
    valid_points = flat_points[~np.isnan(flat_points).any(axis=1)]
    if valid_points.size > 0:
        ax.set_xlim(np.min(valid_points[:,0]), np.max(valid_points[:,0]))
        ax.set_ylim(np.min(valid_points[:,1]), np.max(valid_points[:,1]))
        ax.set_zlim(np.min(valid_points[:,2]), np.max(valid_points[:,2]))
    else:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_zlim(0, 1)

    # Instead of one scatter object, create one for each keypoint with a unique color.
    cmap = cm.get_cmap('tab20')
    scatter_list = []
    for i, part in enumerate(body_parts):
        color = cmap(i % 20)
        sc = ax.scatter([], [], [], c=[color], s=50, label=part)
        scatter_list.append(sc)

    # Create line objects for the skeleton connections.
    line_objs = []
    for conn in constraints:
        line, = ax.plot([], [], [], c="red", linewidth=2)
        line_objs.append(line)
    
    # Add legend (for keypoints).
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1))

    def update(frame):
        frame_points = points3d[frame]  # shape: (n_keypoints, 3)
        # Update each keypoint's scatter.
        for i, sc in enumerate(scatter_list):
            pt = frame_points[i]
            if np.isnan(pt).any():
                sc._offsets3d = ([], [], [])
            else:
                sc._offsets3d = ([pt[0]], [pt[1]], [pt[2]])
        ax.set_title(f"Frame {frame}")
        
        # Update skeleton connections.
        for idx, conn in enumerate(constraints):
            i, j = conn
            pt1 = frame_points[i]
            pt2 = frame_points[j]
            if np.isnan(pt1).any() or np.isnan(pt2).any():
                line_objs[idx].set_data([], [])
                line_objs[idx].set_3d_properties([])
            else:
                xs = [pt1[0], pt2[0]]
                ys = [pt1[1], pt2[1]]
                zs = [pt1[2], pt2[2]]
                line_objs[idx].set_data(xs, ys)
                line_objs[idx].set_3d_properties(zs)
        return scatter_list + line_objs

    # Create the animation.
    anim = FuncAnimation(fig, update, frames=range(start_frame, end_frame),
                         interval=1000/fps, blit=False)

    # Save the animation as an MP4 file.

    video_path = output_dir / "3d_animation.mp4"
    anim.save(str(video_path), writer="ffmpeg", fps=fps)

    return anim
def plot_reprojection_errors(session_name, test_dir, bins=50):
    all_errors = get_reprojection_errors_array(session_name, test_dir)
    flat_errors = all_errors.flatten()
    flat_errors = flat_errors[~np.isnan(flat_errors)]

    # Plot histogram
    plt.figure(figsize=(8, 6))
    plt.hist(flat_errors, bins=bins, edgecolor='black')
    plt.title("Histogram of 3D Reprojection Errors")
    plt.xlabel("Error (pixels)")
    plt.ylabel("Count")
    plt.show()
def create_3d_animation(points3d, session_dir, output_dir=None, fps=30, start_frame=None, end_frame=None):
    """
    Create a simplified 3D animation of pose estimation results.
    
    This function animates 3D keypoints using a scatter plot. Keypoints with any NaN values 
    (i.e. missing data) are ignored in a given frame. The output animation and a static image 
    (from the first frame) will be saved in the output directory.
    
    Args:
        points3d (np.ndarray): 3D points array of shape (n_frames, n_keypoints, 3).
        session_dir (Path): Path to the session directory.
        output_dir (Path, optional): Directory to save animation outputs. If not provided,
                                     a subdirectory "evaluation" is created inside session_dir.
        fps (int, optional): Frames per second for the animation. Default is 30.
        start_frame (int, optional): First frame index to include. Default is 0.
        end_frame (int, optional): Frame index (exclusive) of the last frame to include.
                                   Default is points3d.shape[0].
    
    Returns:
        anim: The Matplotlib FuncAnimation object.
    """
    # Setup output directory
    if output_dir is None:
        output_dir = Path(session_dir) / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set frame range
    if start_frame is None:
        start_frame = 0
    if end_frame is None:
        end_frame = points3d.shape[0]
    
    # Create figure and 3D axis
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # Determine global limits based on all valid (non-NaN) points
    flat_points = points3d.reshape(-1, 3)
    valid_points = flat_points[~np.isnan(flat_points).any(axis=1)]
    if valid_points.size > 0:
        ax.set_xlim(np.min(valid_points[:, 0]), np.max(valid_points[:, 0]))
        ax.set_ylim(np.min(valid_points[:, 1]), np.max(valid_points[:, 1]))
        ax.set_zlim(np.min(valid_points[:, 2]), np.max(valid_points[:, 2]))
    else:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_zlim(0, 1)
    
    # Initialize scatter plot (no data initially)
    scatter = ax.scatter([], [], [], c='blue', s=50)
    
    def update(frame):
        """
        Update function for the animation. For each frame, only valid (non-NaN) keypoints are plotted.
        """
        # Get 3D points for the current frame
        frame_points = points3d[frame]
        # Determine which keypoints have all three coordinates valid (non-NaN)
        valid_mask = ~np.isnan(frame_points).any(axis=1)
        valid_points = frame_points[valid_mask]
        
        if valid_points.size > 0:
            x, y, z = valid_points[:, 0], valid_points[:, 1], valid_points[:, 2]
        else:
            x, y, z = [], [], []
        
        # Update the scatter plot
        scatter._offsets3d = (x, y, z)
        ax.set_title(f"Frame {frame}")
        return scatter,
    
    # Create the animation. Blitting is turned off for 3D plots.
    anim = FuncAnimation(fig, update, frames=range(start_frame, end_frame), interval=1000/fps, blit=False)
    
    # Save the animation as an MP4 video file
    animation_path = output_dir / "3d_animation_simplified.mp4"
    anim.save(str(animation_path), writer="ffmpeg", fps=fps)
    
    # # Also, save a static image of the first frame for quick reference
    # update(start_frame)  # Update plot to the first frame
    # plt.savefig(output_dir / "3d_first_frame_simplified.png")
    
    return anim

def create_3d_animation_(points3d, session_dir, output_dir=None, fps=30, start_frame=None, end_frame=None):
    """Create a 3D animation of the pose estimation results
    
    Args:
        points3d: 3D points array of shape (n_frames, n_keypoints, 3)
        session_dir: Path to session directory
        output_dir: Directory to save animation (default: session_dir/evaluation)
        fps: Frames per second for the animation
        start_frame: First frame to include (default: 0)
        end_frame: Last frame to include (default: last frame)
    """
    from matplotlib.animation import FuncAnimation
    from mpl_toolkits.mplot3d import Axes3D
    
    if output_dir is None:
        output_dir = session_dir / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set frame range
    if start_frame is None:
        start_frame = 0
    if end_frame is None:
        end_frame = points3d.shape[0]
    
    # Load detailed metrics CSV
    detailed_metrics = pd.read_csv(output_dir / "detailed_metrics.csv")
    
    # Create figure and 3D axis
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Set axis labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # Initialize scatter plot for keypoints
    scatter = ax.scatter([], [], [], c='blue', s=50)
    
    # Initialize lines for connections
    lines = []
    for connection in params.connections:
        line, = ax.plot([], [], [], 'r-')
        lines.append(line)
    
    def init():
        # Set initial view
        ax.set_xlim(np.min(points3d[..., 0]), np.max(points3d[..., 0]))
        ax.set_ylim(np.min(points3d[..., 1]), np.max(points3d[..., 1]))
        ax.set_zlim(np.min(points3d[..., 2]), np.max(points3d[..., 2]))
        return [scatter] + lines
    
    def update(frame):
        # Get keypoint colors and validity information
        colors = []
        validity_info = []
        for kp_idx, kp_name in enumerate(params.keypoints):
            # Get data for this keypoint in this frame
            frame_data = detailed_metrics[
                (detailed_metrics['frame'] == frame) & 
                (detailed_metrics['keypoint'] == kp_name)
            ]
            
            if not frame_data.empty:
                # Get reprojection errors and confidence
                reproj_errors = [frame_data[f'reprojection_error_{cam}'].iloc[0] 
                               for cam in params.default_cameras 
                               if f'reprojection_error_{cam}' in frame_data.columns]
                mean_error = np.nanmean(reproj_errors)
                mean_confidence = frame_data['mean_confidence'].iloc[0]
                n_cameras = frame_data['n_cameras_used'].iloc[0]
                
                # Determine validity and color using thresholds from params
                if (mean_error > params.evaluation_thresholds["reprojection_error"] or 
                    mean_confidence < params.evaluation_thresholds["confidence_score"] or 
                    n_cameras < params.evaluation_thresholds["min_cameras"]):
                    colors.append('red')
                    validity_info.append(f"{kp_name}: Invalid (error={mean_error:.1f}, conf={mean_confidence:.2f}, cams={n_cameras})")
                else:
                    colors.append('blue')
                    validity_info.append(f"{kp_name}: Valid (error={mean_error:.1f}, conf={mean_confidence:.2f}, cams={n_cameras})")
            else:
                colors.append('red')
                validity_info.append(f"{kp_name}: No data")
        
        # Update scatter plot with colors
        scatter._offsets3d = (points3d[frame, :, 0], 
                            points3d[frame, :, 1], 
                            points3d[frame, :, 2])
        scatter.set_color(colors)
        
        # Update connections
        for i, (start, end) in enumerate(params.connections):
            start_idx = params.keypoints_dict[start]
            end_idx = params.keypoints_dict[end]
            lines[i].set_data([points3d[frame, start_idx, 0], points3d[frame, end_idx, 0]],
                            [points3d[frame, start_idx, 1], points3d[frame, end_idx, 1]])
            lines[i].set_3d_properties([points3d[frame, start_idx, 2], points3d[frame, end_idx, 2]])
        
        # Update title with frame number and keypoint validity
        title = f'Frame {frame}\n'
        title += '\n'.join(validity_info)
        ax.set_title(title, fontsize=8)
        
        return [scatter] + lines
    
    # Create animation
    anim = FuncAnimation(fig, update, frames=range(start_frame, end_frame),
                        init_func=init, blit=True, interval=1000/fps)
    
    # Save animation
    anim.save(output_dir / '3d_animation.mp4', writer='ffmpeg', fps=fps)
    
    # Also save a static plot of the first frame
    update(start_frame)
    plt.savefig(output_dir / '3d_first_frame.png')
    
    return anim

def generate_evaluation_report(session_dir: Path, output_dir: Path) -> dict:
    """
    Generate a comprehensive evaluation report for a session.
    
    Args:
        session_dir: Path to the session directory
        output_dir: Path to save the evaluation report
        
    Returns:
        Dictionary containing all evaluation metrics
    """
    import json
    import pandas as pd
    import numpy as np
    from pathlib import Path
    from beneuro_pose_estimation import params
    from beneuro_pose_estimation.sleap.sleapTools import get_2d_confidence_scores
    from beneuro_pose_estimation.evaluation import (
        compute_keypoint_missing_frame_stats,
        calculate_reprojection_errors,
    )
    
    session_name = session_dir.name
    csv_file = session_dir / f"{session_name}_3dpts_angles.csv"
    
    # Initialize report dictionary
    report = {}
    
    # 1. Confidence scores
    logger.info("Computing confidence scores...")
    confs = get_2d_confidence_scores(session_dir, cameras=params.default_cameras)
    mean_conf_per_camera = np.nanmean(confs, axis=(1, 2, 3))
    mean_conf_per_keypoint = np.nanmean(confs, axis=(0, 2, 3))
    
    report["confidence_scores"] = {
        "per_camera": mean_conf_per_camera.tolist(),
        "per_keypoint": mean_conf_per_keypoint.tolist()
    }
    
    # 2. Reprojection errors
    logger.info("Computing reprojection errors...")
    reproj_errors = calculate_reprojection_errors(session_dir)
    report["reprojection_errors"] = {
        "per_camera": reproj_errors["per_camera"].tolist(),
        "per_keypoint": reproj_errors["per_keypoint"].tolist(),
        "overall": {
            "mean": float(reproj_errors["overall"]["mean"]),
            "median": float(reproj_errors["overall"]["median"]),
            "std": float(reproj_errors["overall"]["std"])
        }
    }
    
    # 3. Missing frame statistics
    logger.info("Computing missing frame statistics...")
    missing_stats = compute_keypoint_missing_frame_stats(str(csv_file))
    report["missing_frames"] = missing_stats.to_dict(orient="records")
    
    # 4. Angle statistics
    logger.info("Computing angle statistics...")
    df = pd.read_csv(csv_file)
    angle_stats = {}
    for angle_name in params.angles.keys():
        if angle_name in df.columns:
            series = pd.to_numeric(df[angle_name], errors="coerce")
            angle_stats[angle_name] = {
                "min": float(series.min(skipna=True)),
                "max": float(series.max(skipna=True)),
                "mean": float(series.mean(skipna=True)),
                "std": float(series.std(skipna=True)),
                "range": float(series.max(skipna=True) - series.min(skipna=True))
            }
    report["angle_statistics"] = angle_stats
    
    return report

def get_mean_confidence_per_keypoint(session_dir, cameras=params.default_cameras):
    """Calculate mean confidence score per keypoint across all cameras and frames
    
    Args:
        session_dir: Directory containing the session data
        cameras: List of cameras to process
        
    Returns:
        np.ndarray: Array of mean confidence scores per keypoint
    """
    # Get confidence scores for all cameras
    confidence_scores = get_2d_confidence_scores(session_dir, cameras)
    if confidence_scores is None:
        return None
    
    # Calculate mean confidence per keypoint across all cameras and frames
    # Using nanmean to handle any NaN values
    mean_confidence = np.nanmean(confidence_scores, axis=(0, 1, 3))  # Average over cameras, tracks, and frames
    
    logger.info("\nMean confidence per keypoint:")
    for i, body_part in enumerate(params.body_parts):
        logger.info(f"  {body_part}: {mean_confidence[i]:.3f}")
    
    return mean_confidence 


def create_3d_animation_from_csv_good(csv_filepath, output_dir, fps=30, start_frame=None, end_frame=None,
                                 body_parts=None, constraints=None):
    """
    Create a 3D animation from a CSV file with 3D keypoint data and save the video in the output directory.
    
    The CSV is expected to have a column "fnum" and, for each keypoint, columns with suffixes _x, _y, and _z.
    For example: "shoulder_center_x", "shoulder_center_y", "shoulder_center_z", etc.
    
    Parameters:
        csv_filepath (str or Path): Path to the CSV file with the 3D points.
        output_dir (str or Path): Directory to save the animation video.
        fps (int, optional): Frames per second for the animation. Default is 30.
        start_frame (int, optional): First frame index to include. Defaults to 0.
        end_frame (int, optional): Frame index (exclusive) of the last frame to include. Defaults to the total number of frames.
        body_parts (list, optional): List of keypoint names (without coordinate suffixes). 
            If None, defaults to params.body_parts.
        constraints (list of [int, int], optional): List of pairs of indices defining the skeleton connections.
            If None, defaults to params.constraints.
                
    Returns:
        anim: A matplotlib.animation.FuncAnimation object.
    """
    # Use the parameters from params.py if not provided.
    if body_parts is None:
        body_parts = params.body_parts
    if constraints is None:
        constraints = params.constraints

    # Read the CSV file. Adjust the separator if needed (here assumed to be tab-separated).
    csv_filepath = Path(csv_filepath)
    df = pd.read_csv(csv_filepath, sep=",")  # Change sep if your CSV is comma-separated.
    
    n_frames = len(df)
    n_keypoints = len(body_parts)
    
    # Construct a (n_frames, n_keypoints, 3) array from CSV columns.
    points3d = np.empty((n_frames, n_keypoints, 3))
    for i, part in enumerate(body_parts):
        col_x = f"{part}_x"
        col_y = f"{part}_y"
        col_z = f"{part}_z"
        if col_x not in df.columns or col_y not in df.columns or col_z not in df.columns:
            raise ValueError(f"Expected columns for '{part}' not found in CSV.")
        points3d[:, i, 0] = df[col_x].values
        points3d[:, i, 1] = df[col_y].values
        points3d[:, i, 2] = df[col_z].values

    # Set default frame range if not provided.
    if start_frame is None:
        start_frame = 0
    if end_frame is None:
        end_frame = n_frames
    
    # Prepare the output directory.
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a figure with a 3D axis.
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    
    # Set axis limits based on valid points.
    flat_points = points3d.reshape(-1, 3)
    valid_points = flat_points[~np.isnan(flat_points).any(axis=1)]
    if valid_points.size > 0:
        ax.set_xlim(np.min(valid_points[:, 0]), np.max(valid_points[:, 0]))
        ax.set_ylim(np.min(valid_points[:, 1]), np.max(valid_points[:, 1]))
        ax.set_zlim(np.min(valid_points[:, 2]), np.max(valid_points[:, 2]))
    else:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_zlim(0, 1)
    
    # Create a scatter object for the keypoints.
    scatter = ax.scatter([], [], [], c="blue", s=50)
    
    # Create line objects for the skeleton connections.
    line_objs = []
    for conn in constraints:
        line, = ax.plot([], [], [], c="red", linewidth=2)
        line_objs.append(line)
    
    def update(frame):
        # Get the current frame's 3D points.
        frame_points = points3d[frame]
        # Update scatter: only show keypoints that have valid values.
        valid_mask = ~np.isnan(frame_points).any(axis=1)
        valid_points = frame_points[valid_mask]
        if valid_points.size > 0:
            x = valid_points[:, 0]
            y = valid_points[:, 1]
            z = valid_points[:, 2]
        else:
            x, y, z = [], [], []
        scatter._offsets3d = (x, y, z)
        ax.set_title(f"Frame {frame}")
        
        # Update skeleton connections for each constraint.
        for idx, conn in enumerate(constraints):
            i, j = conn
            pt1 = frame_points[i]
            pt2 = frame_points[j]
            # If either point has NaN, hide the line.
            if np.isnan(pt1).any() or np.isnan(pt2).any():
                line_objs[idx].set_data([], [])
                line_objs[idx].set_3d_properties([])
            else:
                xs = [pt1[0], pt2[0]]
                ys = [pt1[1], pt2[1]]
                zs = [pt1[2], pt2[2]]
                line_objs[idx].set_data(xs, ys)
                line_objs[idx].set_3d_properties(zs)
        return [scatter] + line_objs
    
    # Create the animation over the specified frame range.
    anim = FuncAnimation(fig, update, frames=range(start_frame, end_frame),
                         interval=1000/fps, blit=False)
    
    # Save the animation as an MP4 video.
    video_path = output_dir / "3d_animation.mp4"
    anim.save(str(video_path), writer="ffmpeg", fps=fps)
    
    return anim


def compute_keypoint_missing_frame_stats(csv_filepath, body_parts=None, sep=","):
    """
    Reads a CSV of 3D keypoints and computes, for each keypoint,
    how many frames are missing (i.e. where _x, _y, _z are all NaN).

    As it goes, it prints:
        keypoint_name : N frames missing ( P.PP% )

    Returns:
        pd.DataFrame with columns:
            - keypoint
            - n_missing
            - pct_missing
    """
    csv_filepath = Path(csv_filepath)
    df = pd.read_csv(csv_filepath, sep=sep)
    df.columns = df.columns.str.strip()

    if body_parts is None:
        body_parts = params.body_parts

    n_frames = len(df)
    stats = []

    for part in body_parts:
        cols = [f"{part}_x", f"{part}_y", f"{part}_z"]
        missing_cols = [c for c in cols if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Expected columns {cols} for '{part}', but missing {missing_cols}")

        # A frame is "missing" if all three coords are NaN
        missing_mask = df[cols].isna().all(axis=1)
        n_missing = int(missing_mask.sum())
        pct_missing = n_missing / n_frames * 100

        # print as we go
        print(f"{part:15s}: {n_missing:4d} frames missing ({pct_missing:6.2f}%)")

        stats.append({
            "keypoint": part,
            "n_missing": n_missing,
            "pct_missing": pct_missing
        })

    return pd.DataFrame(stats)

def print_angle_ranges_from_csv(csv_filepath, sep="\t"):
    """
    Reads a CSV containing one column per joint angle (named exactly as in params.angles keys)
    and prints min, max, and range (in degrees) for each.

    Expects columns:
        right_knee, left_knee,
        right_ankle, left_ankle,
        right_wrist, left_wrist,
        right_elbow, left_elbow

    Args:
        csv_filepath (str or Path): path to the CSV file.
        sep (str): delimiter (default tab).
    """
    csv_filepath = Path(csv_filepath)
    df = pd.read_csv(csv_filepath, sep=sep)
    df.columns = df.columns.str.strip()

    for angle_name in params.angles.keys():
        if angle_name not in df.columns:
            raise ValueError(f"Column '{angle_name}' not found in CSV. Available columns: {df.columns.tolist()}")
        series = pd.to_numeric(df[angle_name], errors="coerce")
        ang_min = series.min(skipna=True)
        ang_max = series.max(skipna=True)
        ang_range = ang_max - ang_min
        print(
            f"{angle_name:15s}: "
            f"min={ang_min:6.2f}°, "
            f"max={ang_max:6.2f}°, "
            f"range={ang_range:6.2f}°"
        )

def print_angle_ranges_excluding_outliers(
    csv_filepath,
    sep: str = "\t",
    method: str = "iqr",
    thresh: float = 1.5,
    pct_bounds: tuple = (0.01, 0.99)
):
    """
    Reads a CSV with one column per angle (keys of params.angles) and for each:
      • Detects outliers using chosen method:
        - "iqr": 1.5 * IQR
        - "zscore": thresh * std
        - "percentile": drop below pct_bounds[0] or above pct_bounds[1]
        - "mad": thresh * MAD
      • Prints count and % outliers
      • Prints min, max, range over non‑outliers

    Args:
      csv_filepath: path to CSV file
      sep: delimiter
      method: "iqr" | "zscore" | "percentile" | "mad"
      thresh: multiplier (k) for IQR, Z, or MAD
      pct_bounds: (low_pct, high_pct) for percentile method
    """
    df = pd.read_csv(Path(csv_filepath), sep=sep)
    df.columns = df.columns.str.strip()

    for angle_name in params.angles.keys():
        if angle_name not in df.columns:
            raise ValueError(f"Missing column '{angle_name}'")
        series = pd.to_numeric(df[angle_name], errors="coerce").dropna()
        n = len(series)

        # --- outlier masks ---
        if method == "iqr":
            q1, q3 = series.quantile([0.25, 0.75])
            iqr = q3 - q1
            low, high = q1 - thresh * iqr, q3 + thresh * iqr
            is_outlier = (series < low) | (series > high)

        elif method == "zscore":
            z = (series - series.mean()) / series.std(ddof=0)
            is_outlier = z.abs() > thresh

        elif method == "percentile":
            low, high = series.quantile([pct_bounds[0], pct_bounds[1]])
            is_outlier = (series < low) | (series > high)

        elif method == "mad":
            med = series.median()
            mad = np.median(np.abs(series - med))
            # avoid division by zero
            if mad == 0:
                is_outlier = np.zeros(n, dtype=bool)
            else:
                is_outlier = (np.abs(series - med) / mad) > thresh

        else:
            raise ValueError(f"Unknown method '{method}'")

        n_out = is_outlier.sum()
        pct_out = n_out / n * 100 if n > 0 else 0

        # Filtered data
        filtered = series[~is_outlier]
        if filtered.empty:
            print(f"{angle_name:15s} | ALL {n_out}/{n} outliers—no data")
            continue

        amin, amax = filtered.min(), filtered.max()
        arng = amax - amin

        print(
            f"{angle_name:15s} | outliers: {n_out:4d}/{n} ({pct_out:5.2f}%)"
            f" | min={amin:6.2f}°, max={amax:6.2f}°, range={arng:6.2f}°"
        )

from scipy.signal import medfilt
def print_smoothed_angle_ranges_excluding_outliers(
    csv_filepath,
    sep: str = "\t",
    smoothing_window: int = 5,
    method: str = "iqr",
    thresh: float = 1.5,
    pct_bounds: tuple = (0.01, 0.99)
):
    df = pd.read_csv(Path(csv_filepath), sep=sep)
    df.columns = df.columns.str.strip()

    for angle_name in params.angles.keys():
        if angle_name not in df.columns:
            raise ValueError(f"Missing column '{angle_name}' in CSV")

        # ——— BEGIN updated fillna → ffill/bfill ———
        ser = pd.to_numeric(df[angle_name], errors="coerce")
        ser = ser.ffill().bfill()
        raw = ser.values
        # ——— END updated fillna → ffill/bfill ———

        if smoothing_window > 1:
            smoothed = medfilt(raw, kernel_size=smoothing_window)
        else:
            smoothed = raw

        series = pd.Series(smoothed).dropna()
        n = len(series)

        # 2) Outlier detection
        if method == "iqr":
            q1, q3 = series.quantile([0.25, 0.75])
            iqr = q3 - q1
            low, high = q1 - thresh * iqr, q3 + thresh * iqr
            is_outlier = (series < low) | (series > high)

        elif method == "zscore":
            z = (series - series.mean()) / series.std(ddof=0)
            is_outlier = z.abs() > thresh

        elif method == "percentile":
            low, high = series.quantile([pct_bounds[0], pct_bounds[1]])
            is_outlier = (series < low) | (series > high)

        elif method == "mad":
            med = series.median()
            mad = np.median(np.abs(series - med))
            is_outlier = mad > 0 and (np.abs(series - med) / mad) > thresh

        else:
            raise ValueError(f"Unknown method '{method}'")

        n_out = int(is_outlier.sum())
        pct_out = n_out / n * 100 if n else 0.0

        # 3) Compute stats on non‑outliers
        clean = series[~is_outlier]
        if clean.empty:
            print(f"{angle_name:15s} | ALL {n_out}/{n} outliers — no data")
            continue

        amin, amax = clean.min(), clean.max()
        arng = amax - amin

        print(
            f"{angle_name:15s} | outliers: {n_out:4d}/{n} ({pct_out:5.2f}%)"
            f" | min={amin:6.2f}°, max={amax:6.2f}°, range={arng:6.2f}°"
        )

import json
from pathlib import Path

def summarize_model_config(config_path):
    """
    Load a SLEAP training_config.json and print the most important settings:
      • Data/labels
      • Preprocessing
      • Model architecture
      • Training hyperparameters
      • **All** augmentation settings (even those disabled)
      • Outputs
    """
    config_path = Path(config_path)
    with open(config_path, "r") as f:
        cfg = json.load(f)

    # — Data & Labels —
    data = cfg.get("data", {})
    labels = data.get("labels", {})
    print("=== Data & Labels ===")
    print("Training labels:     ", labels.get("training_labels"))
    print("Validation fraction: ", labels.get("validation_fraction"))
    print("Split by indices?    ", labels.get("split_by_inds", False))
    if labels.get("split_by_inds", False):
        print("  • # train idxs:    ", len(labels.get("training_inds", [])))
        print("  • # val   idxs:    ", len(labels.get("validation_inds", [])))

    # — Preprocessing —
    prep = data.get("preprocessing", {})
    print("\n=== Preprocessing ===")
    print("Ensure RGB?         ", prep.get("ensure_rgb"))
    print("Ensure grayscale?   ", prep.get("ensure_grayscale"))
    print(f"Resize & pad?       {prep.get('resize_and_pad_to_target')} "
          f"→ {prep.get('target_width')}×{prep.get('target_height')}")

    # — Model Architecture —
    model = cfg.get("model", {})
    backbone = model.get("backbone", {}).get("unet", {})
    heads    = model.get("heads", {}).get("single_instance", {})
    print("\n=== Model Architecture ===")
    if backbone:
        print("UNet filters:       ", backbone.get("filters"), 
              f"(rate={backbone.get('filters_rate')}, stride={backbone.get('output_stride')})")
    print("Head σ (sigma):     ", heads.get("sigma"), 
          f"(stride={heads.get('output_stride')})")
    print("Keypoint names:     ", ", ".join(heads.get("part_names", [])))

    # — Augmentation —
    aug = cfg.get("optimization", {}).get("augmentation_config", {})
    print("\n=== Augmentation Settings ===")
    # Group them by category for readability
    aug_categories = {
        "Rotation (°)": ["rotate", "rotation_min_angle", "rotation_max_angle"],
        "Translation (px)": ["translate", "translate_min", "translate_max"],
        "Scaling": ["scale", "scale_min", "scale_max"],
        "Uniform Noise": ["uniform_noise", "uniform_noise_min_val", "uniform_noise_max_val"],
        "Gaussian Noise": ["gaussian_noise", "gaussian_noise_mean", "gaussian_noise_stddev"],
        "Contrast (γ)": ["contrast", "contrast_min_gamma", "contrast_max_gamma"],
        "Brightness": ["brightness", "brightness_min_val", "brightness_max_val"],
        "Random Crop": ["random_crop", "random_crop_height", "random_crop_width"],
        "Random Flip": ["random_flip", "flip_horizontal"],
    }
    for title, keys in aug_categories.items():
        values = [aug.get(k) for k in keys]
        print(f"{title:17s}: {values}")

    # — Training Hyperparameters —
    opt = cfg.get("optimization", {})
    print("\n=== Training Hyperparameters ===")
    print("Batch size:         ", opt.get("batch_size"))
    print("Batches per epoch:  ", opt.get("batches_per_epoch"))
    print("Epochs:             ", opt.get("epochs"))
    print("Initial LR:         ", opt.get("initial_learning_rate"))
    lr_sched = opt.get("learning_rate_schedule", {})
    if lr_sched.get("reduce_on_plateau"):
        print("LR sched: reduce on plateau (factor="
              f"{lr_sched.get('reduction_factor')}, patience={lr_sched.get('plateau_patience')})")

    # — Outputs —
    outputs = cfg.get("outputs", {})
    print("\n=== Outputs ===")
    print("Run name:           ", outputs.get("run_name"))
    print("Save visualizations?", outputs.get("save_visualizations"))

def plot_model_metrics_old(model_dir, split="val"):
    """
    Load and display key metrics for a SLEAP model.

    Args:
        model_dir (str or Path): Path to the model directory.
        split (str): Which data split to load metrics for (default "val").
    """

    metrics = sleap.load_metrics(str(model_dir), split=split)

    # Print summary statistics
    print(f"Error distance (50%): {metrics['dist.p50']}")
    print(f"Error distance (90%): {metrics['dist.p90']}")
    print(f"Error distance (95%): {metrics['dist.p95']}")

    print(f"mAP: {metrics['oks_voc.mAP']}")
    print(f"mAR: {metrics['oks_voc.mAR']}")

    # Create a 2x2 figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=100)
    ax1, ax2, ax3, ax4 = axes.flatten()

    # Localization error histogram
    sns.histplot(
        metrics["dist.dists"].flatten(),
        binrange=(0, 20),
        kde=True,
        kde_kws={"clip": (0, 20)},
        stat="probability",
        ax=ax1
    )
    ax1.set_title("Localization Error (px)")
    ax1.set_xlabel("Error distance")
    ax1.set_ylabel("Probability")

    # OKS match score histogram
    sns.histplot(
        metrics["oks_voc.match_scores"].flatten(),
        binrange=(0, 1),
        kde=True,
        kde_kws={"clip": (0, 1)},
        stat="probability",
        ax=ax2
    )
    ax2.set_title("Object Keypoint Similarity")
    ax2.set_xlabel("OKS score")
    ax2.set_ylabel("Probability")

    # Precision-Recall curves
    recall = metrics["oks_voc.recall_thresholds"]
    precisions = metrics["oks_voc.precisions"]
    thresholds = metrics["oks_voc.match_score_thresholds"]
    for idx in range(0, len(thresholds), 2):
        ax3.plot(
            recall,
            precisions[idx],
            label=f"OKS @ {thresholds[idx]:.2f}"
        )
    ax3.set_title("Precision-Recall Curves")
    ax3.set_xlabel("Recall")
    ax3.set_ylabel("Precision")
    ax3.legend(loc="lower left")

    # Turn off the unused subplot
    ax4.axis("off")

    plt.tight_layout()
    plt.show()



def plot_model_metrics(model_dir, split="val", definitions=False):
    """
    Load and display SLEAP model evaluation metrics including visibility metrics.

    Definitions:
      - dist.p90 / p95: 90th/95th percentile pixel error.
      - OKS: Object Keypoint Similarity [0,1].
      - mAP: Mean Average Precision over OKS thresholds.
      - Visibility metrics:
          vis.tp: True Positives (correctly predicted visible)
          vis.fp: False Positives (predicted visible but missing)
          vis.tn: True Negatives (correctly predicted missing)
          vis.fn: False Negatives (missed a visible point)
          vis.precision: vis.tp / (vis.tp + vis.fp)
          vis.recall:    vis.tp / (vis.tp + vis.fn)
    Benchmarks for mice (Pereira et al. 2022):
      - dist.p90 < 3.3 mm
      - dist.p95 < 3.04 mm
      - mAP ≈ 0.927
    """
    defs = """
    Definitions:
    • Localization error
        – Pixel distance between each predicted keypoint and its ground truth
        – Lower values are better (0 px = perfect)

    • Object Keypoint Similarity (OKS)
        – Score in [0,1], higher is better
        – Converts per-keypoint distances into a normalized similarity wrt object scale
        – Object scale comes from the annotation bounding box - not sure whether we did this right.

    • Precision & Recall
        – Precision: fraction of predicted keypoints that are correct 
            (high precision ⇒ few false positives)
        – Recall: fraction of true keypoints that are detected 
            (high recall ⇒ few false negatives)
        – We want precision to stay high even as recall increases

    Metrics (averaged over all OKS thresholds):
    • mAP (mean Average Precision)
        – ∈ [0,1], higher is better
        – Area under the precision–recall curve, then averaged across OKS thresholds

    • mAR (mean Average Recall)
        – ∈ [0,1], higher is better
        – Average recall value across all OKS thresholds
    """
    if definitions:
        print(defs)

    model_dir = Path(model_dir)
    metrics = sleap.load_metrics(str(model_dir), split=split)

    # Print error and OKS metrics with benchmarks
    print("Distance metrics: (SLEAP benchmark for mice (Pereira et al. 2022): < 3.3 mm)")
    print(f"  dist.p50 (50th pct error): {metrics['dist.p50']:.3f} px  (ChatGPT recommends < 5 px)")
    print(f"  dist.p90 (90th pct error): {metrics['dist.p90']:.3f} px  (ChatGPT recommends < 10 px)")
    print(f"  dist.p95 (95th pct error): {metrics['dist.p95']:.3f} px  (ChatGPT recommends < 15 px)\n")
 
    # Print visibility metrics
    print("Visibility metrics:")
    print(f"  vis.tp        (True Positives)          : {metrics['vis.tp']}")
    print(f"  vis.fp        (False Positives)         : {metrics['vis.fp']}")
    print(f"  vis.tn        (True Negatives)          : {metrics['vis.tn']}")
    print(f"  vis.fn        (False Negatives)         : {metrics['vis.fn']}")
    print(f"  vis.precision (Precision = TP/(TP+FP)) : {metrics['vis.precision']:.3f}")
    print(f"  vis.recall    (Recall   = TP/(TP+FN)) : {metrics['vis.recall']:.3f}\n")
    print(f"mAP: {metrics['oks_voc.mAP']:.3f}  (SLEAP benchmark : ~0.927)")
    print(f"mAR: {metrics['oks_voc.mAR']:.3f} ")


    # Plot histograms and PR curves as before
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=100)
    ax1, ax2, ax3, ax4 = axes.flatten()

    sns.histplot(
        metrics["dist.dists"].flatten(),
        binrange=(0, 20),
        kde=True,
        kde_kws={"clip": (0, 20)},
        stat="probability",
        ax=ax1
    )
    ax1.set_title("Localization Error (px)")
    ax1.set_xlabel("Pixel Error")
    ax1.set_ylabel("Probability")

    sns.histplot(
        metrics["oks_voc.match_scores"].flatten(),
        binrange=(0, 1),
        kde=True,
        kde_kws={"clip": (0, 1)},
        stat="probability",
        ax=ax2
    )
    ax2.set_title("Object Keypoint Similarity (OKS)")
    ax2.set_xlabel("OKS Score")
    ax2.set_ylabel("Probability")

    recall = metrics["oks_voc.recall_thresholds"]
    precisions = metrics["oks_voc.precisions"]
    thresholds = metrics["oks_voc.match_score_thresholds"]
    for idx in range(0, len(thresholds), max(1, len(thresholds)//5)):
        ax3.plot(recall, precisions[idx], label=f"OKS ≥ {thresholds[idx]:.2f}")
    ax3.set_title("Precision-Recall Curves")
    ax3.set_xlabel("Recall")
    ax3.set_ylabel("Precision")
    ax3.legend(loc="lower left")

    ax4.axis("off")

    plt.tight_layout()
    plt.show()


def _load_angle_df(csv_path, frame_start=0, frame_end=None):
    """Internal: load CSV and slice by frame index."""
    df = pd.read_csv(csv_path)
    if frame_end is None:
        frame_end = len(df)
    return df.iloc[frame_start:frame_end].reset_index(drop=True)


def plot_angle_histograms(csv_path, frame_start=0, frame_end=None, bins=50):
    """
    Plot a separate histogram for each `<joint>_angle` column in the given frame range.
    """
    df = _load_angle_df(csv_path, frame_start, frame_end)
    angle_cols = [c for c in df.columns if c.endswith("_angle")]
    for col in angle_cols:
        vals = df[col].dropna().values
        plt.figure(figsize=(6,4))
        plt.hist(vals, bins=bins, edgecolor="black")
        plt.title(f"Histogram of {col}")
        plt.xlabel("Angle (°)")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.show()

def plot_angles_timeseries(csv_path, frame_start=0, frame_end=None):
    """
    Overlay all `<joint>_angle` time series over fnum (frame number).
    """
    df = _load_angle_df(csv_path, frame_start, frame_end)
    angle_cols = [c for c in df.columns if c.endswith("_angle")]
    if "fnum" not in df.columns:
        raise KeyError("CSV must contain 'fnum' column for frame numbers.")
    plt.figure(figsize=(10,6))
    for col in angle_cols:
        plt.plot(df["fnum"], df[col], label=col)
    plt.title("Joint Angles over Time")
    plt.xlabel("Frame number")
    plt.ylabel("Angle (°)")
    plt.legend(bbox_to_anchor=(1.02,1), loc="upper left")
    plt.tight_layout()
    plt.show()


def plot_bodypart_autocorr_spectrum(
    csv_path,
    body_parts = params.body_parts,
    frame_start=0,
    frame_end=None,
    max_lag=30,
    dt=0.001
):
    """
    For each body part in `body_parts` (e.g., 'shoulder_center'):
      • Plot autocorrelation of its X, Y, Z trajectories up to max_lag (samples).
      • Plot power spectrum of the autocorrelation, annotating the peak frequency.

    Args:
        csv_path (str or Path): Path to the CSV with columns '<bodypart>_x', '_y', '_z'.
        body_parts (list of str): List of body part base names.
        frame_start (int): First frame index to include.
        frame_end (int or None): One-past-last frame index (None => end of CSV).
        max_lag (int): Maximum lag (in samples) for autocorrelation.
        dt (float): Time step per sample (in seconds).

    Returns:
        None; shows figures for each body part.
    """
    # Load and slice
    df = pd.read_csv(csv_path)
    if frame_end is None:
        frame_end = len(df)
    df = df.iloc[frame_start:frame_end].reset_index(drop=True)

    # Sampling params
    fs = 1.0 / dt
    lags = np.arange(max_lag + 1)
    freqs = np.fft.rfftfreq(max_lag + 1, d=dt)

    # Loop over body parts
    for bp in body_parts:
        # Prepare plot
        fig, (ax_ac, ax_psd) = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)

        # Colors for X, Y, Z
        dims = ['x', 'y', 'z']
        colors = ['r', 'g', 'b']
        for dim, color, coord in zip(dims, colors, ['X', 'Y', 'Z']):
            col = f"{bp}_{dim}"
            # Extract series, handle non-numeric
            series = pd.to_numeric(df[col], errors='coerce').values
            # Remove NaNs
            series = series[~np.isnan(series)]
            T = len(series)
            if T == 0:
                continue

            # Zero-mean and normalize
            x0 = series - np.mean(series)
            var = np.var(x0)
            # Compute autocorrelation
            if var == 0:
                ac = np.zeros_like(lags, dtype=float)
                ac[0] = 1.0
            else:
                ac = np.array([
                    np.sum(x0[:T - l] * x0[l:]) / (T - l) / var
                    for l in lags
                ])
            # Plot autocorr
            ax_ac.plot(lags, ac, color=color, label=coord)
            # FFT of autocorr to get spectrum
            S = np.abs(np.fft.rfft(ac))
            # Peak frequency (exclude DC)
            peak_idx = np.argmax(S[1:]) + 1
            peak_f = freqs[peak_idx]
            # Plot spectrum
            ax_psd.plot(freqs, S, color=color, label=f"{coord}, fₚ={peak_f:.2f}Hz")

        # Finalize autocorr plot
        ax_ac.set_title(f"{bp} autocorrelation", fontsize=14)
        ax_ac.set_xlabel("Lag (samples)")
        ax_ac.set_ylabel("Autocorr")
        ax_ac.set_xlim(0, max_lag)
        ax_ac.legend()

        # Finalize PSD plot
        ax_psd.set_title(f"{bp} power spectrum", fontsize=14)
        ax_psd.set_xlabel("Frequency (Hz)")
        ax_psd.set_ylabel("Amplitude")
        ax_psd.set_xlim(0, fs / 2)
        ax_psd.legend()

        plt.show()

def plot_angle_velocity_histograms(csv_path, frame_start=0, frame_end=None, bins=50):
    """
    Plot a histogram of instantaneous angular velocity (Δangle/frame) for each <joint>_angle column.
    """
    df = pd.read_csv(csv_path)
    if frame_end is None:
        frame_end = len(df)
    df = df.iloc[frame_start:frame_end].reset_index(drop=True)
    
    angle_cols = [c for c in df.columns if c.endswith("_angle")]
    for col in angle_cols:
        vals = pd.to_numeric(df[col], errors="coerce").dropna().values
        vel = np.diff(vals)
        plt.figure(figsize=(6,4))
        plt.hist(vel, bins=bins, edgecolor="black")
        plt.title(f"Histogram of {col} Velocity")
        plt.xlabel("Velocity (°/frame)")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.show()

def plot_angle_acceleration_histograms(csv_path, frame_start=0, frame_end=None, bins=50):
    """
    Plot a histogram of instantaneous angular acceleration (Δ²angle/frame²) for each <joint>_angle column.
    """
    df = pd.read_csv(csv_path)
    if frame_end is None:
        frame_end = len(df)
    df = df.iloc[frame_start:frame_end].reset_index(drop=True)
    
    angle_cols = [c for c in df.columns if c.endswith("_angle")]
    for col in angle_cols:
        vals = pd.to_numeric(df[col], errors="coerce").dropna().values
        vel = np.diff(vals)
        acc = np.diff(vel)
        plt.figure(figsize=(6,4))
        plt.hist(acc, bins=bins, edgecolor="black")
        plt.title(f"Histogram of {col} Acceleration")
        plt.xlabel("Acceleration (°/frame²)")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.show()

def print_angle_stats(csv_path, frame_start=0, frame_end=None):
    """
    Compute and print detailed statistics for each joint angle column:
      • Full range (min, max)
      • Interquartile range [Q1, Q3] and IQR
      • For velocities (Δangle/frame) and accelerations (Δ²angle/frame²):
        same metrics

    Args:
        csv_path (str or Path): Path to CSV containing '<joint>_angle' columns.
        frame_start (int): First row index to include.
        frame_end (int or None): One-past-last row index; defaults to end of CSV.
    """
    df = pd.read_csv(csv_path)
    if frame_end is None:
        frame_end = len(df)
    df = df.iloc[frame_start:frame_end].reset_index(drop=True)

    angle_cols = [c for c in df.columns if c.endswith("_angle")]


    for col in angle_cols:
        vals = pd.to_numeric(df[col], errors="coerce").dropna().values
        if vals.size == 0:
            print(f"\n{col}: no data")
            continue

        # Full range
        mn, mx = vals.min(), vals.max()
        # Interquartile range
        q1, q3 = np.percentile(vals, [25, 75])
        iqr = q3 - q1

        print(f"\n{col}:")
        print(f"  Full range: [{mn:.2f}, {mx:.2f}]°")
        print(f"  IQR: [{q1:.2f}, {q3:.2f}]° ")

        # Velocity
        vel = np.diff(vals)
        if vel.size > 0:
            mn_v, mx_v = vel.min(), vel.max()
            q1_v, q3_v = np.percentile(vel, [25, 75])
            iqr_v = q3_v - q1_v
            print("  Velocity (Δ°/frame):")
            print(f"    Full range: [{mn_v:.2f}, {mx_v:.2f}]")
            print(f"    IQR: [{q1_v:.2f}, {q3_v:.2f}] ")
        else:
            print("  Velocity: not enough data")

        # Acceleration
        acc = np.diff(vel) if vel.size > 0 else np.array([])
        if acc.size > 0:
            mn_a, mx_a = acc.min(), acc.max()
            q1_a, q3_a = np.percentile(acc, [25, 75])
            iqr_a = q3_a - q1_a
            print("  Acceleration (Δ²°/frame²):")
            print(f"    Full range: [{mn_a:.2f}, {mx_a:.2f}]")
            print(f"    IQR: [{q1_a:.2f}, {q3_a:.2f}] ")
        else:
            print("  Acceleration: not enough data")

def print_angle_stats_90(csv_path, frame_start=0, frame_end=None):
    """
    Compute and print detailed statistics for each joint angle column:
      • Full range (min, max)
      • Interquartile range [Q1, Q3] and IQR
      • For velocities (Δangle/frame) and accelerations (Δ²angle/frame²):
        same metrics

    Args:
        csv_path (str or Path): Path to CSV containing '<joint>_angle' columns.
        frame_start (int): First row index to include.
        frame_end (int or None): One-past-last row index; defaults to end of CSV.
    """
    df = pd.read_csv(csv_path)
    if frame_end is None:
        frame_end = len(df)
    df = df.iloc[frame_start:frame_end].reset_index(drop=True)

    angle_cols = [c for c in df.columns if c.endswith("_angle")]


    for col in angle_cols:
        vals = pd.to_numeric(df[col], errors="coerce").dropna().values
        if vals.size == 0:
            print(f"\n{col}: no data")
            continue

        # Full range
        mn, mx = vals.min(), vals.max()
        # Interquartile range
        q1, q3 = np.percentile(vals, [5, 95])
        iqr = q3 - q1

        print(f"\n{col}:")
        print(f"  Full range: [{mn:.2f}, {mx:.2f}]°")
        print(f"  90%: [{q1:.2f}, {q3:.2f}]° (90% = {iqr:.2f}°)")

        # Velocity
        vel = np.diff(vals)
        if vel.size > 0:
            mn_v, mx_v = vel.min(), vel.max()
            q1_v, q3_v = np.percentile(vel, [5, 95])
            iqr_v = q3_v - q1_v
            print("  Velocity (Δ°/frame):")
            print(f"    Full range: [{mn_v:.2f}, {mx_v:.2f}]")
            print(f"    90%: [{q1_v:.2f}, {q3_v:.2f}] (90% = {iqr_v:.2f})")
        else:
            print("  Velocity: not enough data")

        # Acceleration
        acc = np.diff(vel) if vel.size > 0 else np.array([])
        if acc.size > 0:
            mn_a, mx_a = acc.min(), acc.max()
            q1_a, q3_a = np.percentile(acc, [5, 95])
            iqr_a = q3_a - q1_a
            print("  Acceleration (Δ²°/frame²):")
            print(f"    Full range: [{mn_a:.2f}, {mx_a:.2f}]")
            print(f"    90%: [{q1_a:.2f}, {q3_a:.2f}] (90% = {iqr_a:.2f})")
        else:
            print("  Acceleration: not enough data")
def build_angle_stats_table(csv_path, frame_start=0, frame_end=None):
    """
    Build a DataFrame summarizing full range and IQR for each joint angle,
    its velocity, and its acceleration.
    """
    df = pd.read_csv(csv_path)
    if frame_end is None:
        frame_end = len(df)
    df = df.iloc[frame_start:frame_end].reset_index(drop=True)

    angle_cols = [c for c in df.columns if c.endswith("_angle")]
    records = []

    for col in angle_cols:
        vals = pd.to_numeric(df[col], errors="coerce").dropna().values
        if vals.size == 0:
            continue

        # Angle stats
        mn, mx = vals.min(), vals.max()
        q1, q3 = np.percentile(vals, [25, 75])
        records.append({
            "joint": col,
            "type": "angle",
            "full_min": f"{mn:.2f}°",
            "full_max": f"{mx:.2f}°",
            "IQR_low": f"{q1:.2f}°",
            "IQR_high": f"{q3:.2f}°"
        })

        # Velocity stats
        vel = np.diff(vals)
        if vel.size > 0:
            mn_v, mx_v = vel.min(), vel.max()
            q1_v, q3_v = np.percentile(vel, [25, 75])
            records.append({
                "joint": col,
                "type": "velocity",
                "full_min": f"{mn_v:.2f}°/frame",
                "full_max": f"{mx_v:.2f}°/frame",
                "IQR_low": f"{q1_v:.2f}°/frame",
                "IQR_high": f"{q3_v:.2f}°/frame"
            })

        # Acceleration stats
        acc = np.diff(vel) if vel.size > 0 else np.array([])
        if acc.size > 0:
            mn_a, mx_a = acc.min(), acc.max()
            q1_a, q3_a = np.percentile(acc, [25, 75])
            records.append({
                "joint": col,
                "type": "acceleration",
                "full_min": f"{mn_a:.2f}°/frame²",
                "full_max": f"{mx_a:.2f}°/frame²",
                "IQR_low": f"{q1_a:.2f}°/frame²",
                "IQR_high": f"{q3_a:.2f}°/frame²"
            })

    stats_df = pd.DataFrame.from_records(records,
        columns=["joint", "type", "full_min", "full_max", "IQR_low", "IQR_high"])
    return stats_df

def print_angle_stats_table(csv_path, frame_start=0, frame_end=None):
    """
    Compute and print a summary table of angle, velocity, and acceleration stats.
    """
    stats_df = build_angle_stats_table(csv_path, frame_start, frame_end)
    # Print as Markdown for clearer console display
    print(stats_df.to_markdown(index=False))