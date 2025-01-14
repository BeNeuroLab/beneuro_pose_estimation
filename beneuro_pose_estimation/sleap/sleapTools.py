"""
Module for SLEAP processing
TBD:
- test evaluation functions



-> define paths in repo_path/beneuro_pose_estimation/params.py
 
-> to run as package (in dev stage):
conda activate bnp
cd repo_path

python -m beneuro_pose_estimation.cli annotate --sessions session_name --cameras camera_name --pred (to launch annotation GUI to annotate; 
(if --pred, predictions are run on the annotation videos using the current model so anotations can be made by correcting predictions)
python -m beneuro_pose_estimation.cli predict-2D --sessions session_name --cameras camera_name (to get 2D predictions)
python -m beneuro_pose_estimation.cli visualize-2D --sessions session_name --cameras camera_name (to launch annotation GUI to visualize predictions)
python -m beneuro_pose_estimation.cli create-annotations --sessions session_name --cameras camera_name (to create annotation projects using frame selection pipeline without launching the GUI)
python -m beneuro_pose_estimation.cli train --sessions session_name --cameras camera_name (to train models)
# cameras argument optional - default = params.default_cameras

-> after package installation:
conda activate bnp

bnp annotate --sessions session_name --cameras camera_name (to launch annotation GUI to annotate)
bnp predict-2D --sessions session_name --cameras camera_name (to get 2D predictions)
bnp visualize-2D --sessions session_name --cameras camera_name (to launch annotation GUI to visualize predictions)
bnp create-annotation-projects --sessions session_name --cameras camera_name (to create annotation projects using frame selection pipeline without launching the GUI)
bnp train --sessions session_name --cameras camera_name (to train models)
# cameras argument optional - default = params.default_cameras

"""
import sys
import os
import subprocess
from pathlib import Path
from beneuro_pose_estimation import params, set_logging
import logging
if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
import attr
import numpy as np
import random
import json
import matplotlib.pyplot as plt
import cv2
import json
from typing import Dict, List, Optional, Union

from sleap.io.video import Video
from sleap import Labels, Video, LabeledFrame, Instance, Skeleton
from sleap.info.feature_suggestions import (
    FeatureSuggestionPipeline,
    ParallelFeaturePipeline,
)
import sleap
import seaborn as sns
import pandas as pd
# import argparse


def compare_models(models_folder,test_gt_path = None):
    """
    TBD - test

    """
    metrics_list = []
    
    for folder in os.listdir(models_folder):
        model_folder = os.path.join(models_folder, folder)
        if os.path.isdir(model_folder):
            try:
                # Load and evaluate model
                if test_gt_path is not None:
                    predictor = sleap.load_model(model_folder)
                    labels_gt = sleap.load_file(test_gt_path)
                    labels_pr = predictor.predict(labels_gt)
                    metrics = sleap.nn.evals.evaluate(labels_gt, labels_pr)
                else:
                    metrics = sleap.load_metrics(model_folder, split="val")
                
                # Flatten metrics into a single row for DataFrame
                metrics_flat = {
                    "Model": folder,
                }
                for key, value in metrics.items():
                    if isinstance(value, (float, int)):
                        metrics_flat[key] = value
                    elif isinstance(value, (list, np.ndarray)):
                        metrics_flat[key] = np.mean(value)  # Use mean for lists/arrays
                
                metrics_list.append(metrics_flat)
            
            except Exception as e:
                print(f"Error evaluating model in folder {folder}: {e}")
    output_csv = f"{models_folder}/metrics.csv"
    # Create DataFrame from collected metrics
    metrics_df = pd.DataFrame(metrics_list)
    metrics_df.to_csv(output_csv, index=False)
    print(f"Metrics comparison saved to {output_csv}")
    
    return metrics_df 

def find_best_models(metrics_df,metric = None):
    """
    Print the best model for each metric.
    TBD - test
    
    Args:
        metrics_df (pd.DataFrame): DataFrame containing metrics for all models.
    """
    if metric:
        # Check if the specified metric exists in the DataFrame
        if metric not in metrics_df.columns:
            print(f"Metric '{metric}' not found in the DataFrame.")
            return
        
        if pd.api.types.is_numeric_dtype(metrics_df[metric]):
            best_index = metrics_df[metric].idxmax()
            best_model = metrics_df.iloc[best_index]["Model"]
            print(f"Best model for {metric}: {best_model}")
        else:
            print(f"Metric '{metric}' is not numeric and cannot be evaluated.")
    else:
        # Evaluate all metrics
        best_models = {}
        for column in metrics_df.columns:
            if column != "Model" and pd.api.types.is_numeric_dtype(metrics_df[column]):
                best_index = metrics_df[column].idxmax()
                best_models[column] = metrics_df.iloc[best_index]["Model"]
        
        print("Best models for each metric:")
        for metric_name, model in best_models.items():
            print(f"{metric_name}: {model}")

def evaluate_model(model_path,test_gt_path = None):
    """
    TBD - test

    """
    if test_gt_path is not None:
        predictor = sleap.load_model(model_path)
        labels_gt = sleap.load_file(test_gt_path)
        labels_pr = predictor.predict(labels_gt)
        metrics = sleap.nn.evals.evaluate(labels_gt, labels_pr)
    else:
        metrics = sleap.load_metrics(model_path, split="val")
    plt.figure(figsize=(6, 3), dpi=150, facecolor="w")
    sns.histplot(metrics["dist.dists"].flatten(), binrange=(0, 20), kde=True, kde_kws={"clip": (0, 20)}, stat="probability")
    plt.xlabel("Localization error (px)")
    plt.figure(figsize=(6, 3), dpi=150, facecolor="w")
    sns.histplot(metrics["oks_voc.match_scores"].flatten(), binrange=(0, 1), kde=True, kde_kws={"clip": (0, 1)}, stat="probability")
    plt.xlabel("Object Keypoint Similarity")
    plt.figure(figsize=(4, 4), dpi=150, facecolor="w")
    for precision, thresh in zip(metrics["oks_voc.precisions"][::2], metrics["oks_voc.match_score_thresholds"][::2]):
        plt.plot(metrics["oks_voc.recall_thresholds"], precision, "-", label=f"OKS @ {thresh:.2f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc="lower left")
    node_names = metrics.get("node_names", [f"Node {i}" for i in range(metrics["dist.dists"].shape[1])])
    dists_df = pd.DataFrame(metrics["dist.dists"], columns=node_names)
    dists_melted = dists_df.melt(var_name="Node", value_name="Error")

    # Create the boxplot
    plt.figure(figsize=(8, 6), dpi=150, facecolor="w")
    sns.boxplot(data=dists_melted, x="Error", y="Node", fliersize=0)
    sns.stripplot(data=dists_melted, x="Error", y="Node", alpha=0.5, jitter=True, color="red")
    plt.title("Localization Error by Node")
    plt.xlabel("Error (px)")
    plt.ylabel("Node")
    plt.grid(True)
    plt.show()
    return 

def select_frames_to_annotate(session,camera,pipeline = params.frame_selection_pipeline,new_video_path = None):
    """
    - Selects frames to annotate using the feature suggestion pipeline, 
    - Saves them as .png,
    - Creates a new .mp4 video for faster processing 

    """
    
    # Define input video path
    animal = session.split("_")[0]
    # video_path = f"{params.recordings_dir}/{animal}/{session}/{session}_cameras/{session}_{params.camera_name_mapping.get(camera, camera)}.avi"
    video_path = f"{params.recordings_dir}/{animal}/{session}/{session}_cameras/{session}_{params.camera_name_mapping.get(camera, camera)}.avi"
    try: 
        video = Video.from_filename(video_path)
        
        # Run frames selection pipeline
        pipeline.run_disk_stage([video])
        frame_data = pipeline.run_processing_state()

        # Define selected frames path 
        frames_dir = f"{params.slp_annotations_dir}/{session}_annotations/{session}_{camera}_annotations"
        os.makedirs(frames_dir, exist_ok=True)
        # Save selected frames as images in the frame directory
        for item in frame_data.items:
            frame_idx = item.frame_idx
            frame = video.get_frame(frame_idx)
            plt.imsave(os.path.join(frames_dir, f'{session}_{camera}_frame_{frame_idx}.png'), frame)
        
        logging.info(f"Selected frames saved for {session}, {camera}")
    except Exception as e:
        logging.error(f"An error occurred while selecting frames to annotate for session {session}, camera {camera}: {e}")

    # create new video from the selected frames
    if new_video_path is None:
       new_video_path = f"{frames_dir}/{session}_{camera}_annotations.mp4"
    
    try:
        create_video_from_frames(frames_dir,new_video_path) 
    except Exception as e:
        logging.error(f"An error occurred while creating the annotation video for session {session}, camera {camera}: {e}")

    
    

    return 

def create_annotation_projects(sessions = params.sessions_to_annotate,cameras = params.default_cameras):
    '''
    create annotation projects for a list of sessions and cameras without launching GUI for annotation
    '''
    cameras = cameras or params.default_cameras
    if isinstance(sessions, str):
        sessions = [sessions]
    if isinstance(cameras, str):
        cameras = [cameras]
    for session in sessions:
        for camera in cameras:
            create_annotation_project(session,camera)
    

def create_video_from_frames(frames_dir, video_path, output_width=1280, output_height=720, fps=5):
    # Get a list of PNG image filenames
    images = [img for img in os.listdir(frames_dir) if img.endswith(".png")]

    # Sort the image filenames to ensure correct order
    images = sorted(images)

    # Set the video codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(video_path, fourcc, fps, (output_width, output_height))

    # Iterate through the PNG images and write them to the video
    for image in images:
        frame = cv2.imread(os.path.join(frames_dir, image))

        if frame is not None:
            # Resize the frame to the desired output size
            resized_frame = cv2.resize(frame, (output_width, output_height))

            # Write the resized frame to the video
            video.write(resized_frame)
        else:
            logging.info(f"Skipping image {image} due to reading error.")

    for image in images:
        image_path = os.path.join(frames_dir, image)
        os.remove(image_path)

    # Release the VideoWriter object
    video.release()

    return



def create_annotation_project(session, camera):
    '''
    Create slp project for annotation to launch annotation GUI on
    * should we initialize instances for all the frames in the annotation video instead of just the first one?
    '''
    # Paths
    # video_path = f"{params.recordings_dir}/{animal}/{session}/{session}_cameras/{session}_{params.camera_name_mapping.get(camera, camera)}.avi"
    select_frames_to_annotate(session,camera,params.frame_selection_pipeline)
    labels_output_path = f"{params.slp_annotations_dir}/{session}_annotations/{session}_{camera}_annotations/{session}_{camera}.slp"
    annotations_dir_path = f"{params.slp_annotations_dir}/{session}_annotations/{session}_{camera}_annotations/"
    videos = [vid for vid in os.listdir(annotations_dir_path) if vid.endswith(".mp4")]
    

    # Load skeleton
    with open(params.skeleton_path, 'r') as f:
        skeleton_data = json.load(f)
    skeleton = Skeleton.from_dict(skeleton_data)
    
    # Initialize a list of labeled frames
    labeled_frames = []
    for vid in videos:
        video = Video.from_filename(annotations_dir_path + vid)
        instances = [Instance(skeleton=skeleton)]
        labeled_frame = LabeledFrame(video=video, frame_idx=0, instances=instances)
        labeled_frames.append(labeled_frame)
    labels = Labels(labeled_frames)
    os.makedirs(os.path.dirname(labels_output_path), exist_ok=True)
    labels.save(labels_output_path)
    logging.info(f"Sleap project created for session {session},camera {camera}.")
    

    return 


def create_annotation_project_inefficient(session, camera):
    '''
    create annotation video using the full video (without  creating a new video from the selected frames)
    '''
    logging.info(f"Creating SLEAP project for session {session} and camera {camera}...")
            
    animal = session.split("_")[0]
    # Paths
    video_path = f"{params.recordings_dir}/{animal}/{session}/{session}_cameras/{session}_{params.camera_name_mapping.get(camera, camera)}.avi"
    labels_output_path = f"{params.slp_annotations_dir}/{session}_annotations/{session}_{camera}_annotations/{session}_{camera}.slp"

    # Load video and skeleton
    video = Video.from_filename(video_path)
    with open(params.skeleton_path, 'r') as f:
        skeleton_data = json.load(f)
    skeleton = Skeleton.from_dict(skeleton_data)

    # Select frames using the feature suggestion pipeline
    params.frame_selection_pipeline.run_disk_stage([video])
    frame_data = params.frame_selection_pipeline.run_processing_state()
    # Initialize labeled frames with selected frames
    labeled_frames = []
    for item in frame_data.items:
        frame_idx = item.frame_idx
        instances = [Instance(skeleton=skeleton)]  # Empty instance
        labeled_frame = LabeledFrame(video=video, frame_idx=frame_idx, instances=instances)
        labeled_frames.append(labeled_frame)
        logging.info(f"Labeled frame created for {session}, {camera}, frame {frame_idx}")

    # Save the labeled frames to a .slp project file
    labels = Labels(labeled_frames)
    os.makedirs(os.path.dirname(labels_output_path), exist_ok=True)
    labels.save(labels_output_path)
    logging.info(f"Sleap project created for session {session},camera {camera}.")
    
    return 



def annotate_videos(sessions, cameras = params.default_cameras, pred = False):
    """
    creates slp project using selected frames from raw video and launches annotation GUI
    if pred, runs predictions on the selected frames with existing model,
    so the annotation can be made by correcting the predictions

    ------
    """
    sessions = sessions or params.sessions_to_annotate
    cameras = cameras or params.default_cameras
    if isinstance(sessions, str):
        sessions = [sessions]
    if isinstance(cameras, str):
        cameras = [cameras]
    # for each session, check if a sleap projects exists already or not
    for session in sessions:
        try:
            session_dir = f"{params.slp_annotations_dir}/{session}_annotations"
            os.makedirs(session_dir, exist_ok=True)
            for camera in cameras:
                try:
                    project_dir = f"{session_dir}/{session}_{camera}_annotations"
                    project_path = f"{project_dir}/{session}_{camera}.slp"
                    os.makedirs(project_dir, exist_ok=True)
                    if not os.path.exists(project_path):
                        create_annotation_project(session,camera)
                    if pred:
                        model_dir = f"{params.slp_models_dir}/{camera}"
                        if not os.path.exists(model_dir):
                            logging.info(f"Model directory for {camera} does not exist, skipping.")
                            continue
                        model_path = f"{model_dir}/training_config.json"
                        command = [
                        "sleap-track",
                        project_path,
                        "--video.index", "0",
                        "-m", model_path,
                        "-o", project_path
                        ]
                        logging.info(f"Running sleap-track on annotation video")
                        # Run the sleap-track command using subprocess
                        subprocess.run(command, check=True)
                        logging.info(f"Tracking completed\n")

                    logging.info(f"Launching annotation GUI...")
                    subprocess.run(["sleap-label", project_path]) # first test if the project is created
                except Exception as e:
                    logging.error(f"Failed to process camera {camera} for session {session}: {e}")
        except Exception as e:
            logging.error(f"Failed to process session {session}: {e}")

def create_training_file(camera, sessions):
    """
    .slp project for a specific camera - merging all projects for that camera
    """
    # Path to save the combined training project
    combined_project_path = f"{params.slp_training_dir}/{camera}.slp"
    
    all_labeled_frames = []

    for session in sessions:
        try:
            # Define path to the session-specific .slp file
            session_slp_path = f"{params.slp_annotations_dir}/{session}_annotations/{session}_{camera}_annotations/{session}_{camera}.slp"
            # # Check if the .slp file exists for the session
            if os.path.exists(f"{session_slp_path}"):
                session_labels = sleap.load_file(session_slp_path)
                session_labeled_frames = session_labels.labeled_frames
                all_labeled_frames.extend(session_labeled_frames)
                logging.info(f"Added {len(session_labeled_frames)} frames from session {session} for camera {camera}.")
            else:
                logging.info(f"SLP annotation project for {session}, {camera} does not exist. Skipping.")
                continue
        except Exception as e:
            logging.error(f"An error occurred while processing session {session} for camera {camera}: {e}")

    # Create a new Labels object with the combined labeled frames
    combined_labels = Labels(labeled_frames=all_labeled_frames)

    # Ensure the directory for the combined project exists
    os.makedirs(os.path.dirname(combined_project_path), exist_ok=True)
    try:
        # Save the combined Labels object to a new .slp file
        combined_labels.save(combined_project_path)
        logging.info(f"Combined training project saved at {combined_project_path}")
    except Exception as e:
        logging.error(f"Failed to save combined training project: {e}")

    return

def create_training_projects(sessions, cameras = params.default_cameras):
    """
    creates .slp training projects
    """
    cameras = cameras or params.default_cameras
    if isinstance(cameras, str):
        cameras = [cameras]
    for camera in cameras:
        training_dir = f"{params.slp_training_dir}/{camera}"
        os.makedirs(training_dir, exist_ok=True)
        labels_file = f"{params.slp_training_dir}/{camera}/{camera}.slp"
        config_file = f"{params.slp_training_dir}/{camera}/training_config.json"

        # Check if the .slp file exists; if not, run create_training_file
        if not os.path.exists(labels_file):
            logging.info(f"{labels_file} does not exist. Creating training file...")
            create_training_file(camera, sessions)

        # Ensure configuration file exists
        if not os.path.exists(config_file):
            logging.info(f"Configuration file for {camera} does not exist, using default one")
            source_path = params.training_config_path

            # Read JSON data from the source file
            with open(source_path, 'r') as src_file:
                data = json.load(src_file)

            # Write JSON data to the destination file
            with open(config_file, 'w') as dest_file:
                json.dump(data, dest_file, indent=4)

            logging.info(f"File copied from {source_path} to {config_file}")

    return


def create_training_config_file(config_file):
    return

def train_models(cameras = params.default_cameras, sessions = params.training_sessions): 
    '''
    TBD 
    - create config file with training parameters; check if config file exists, if not create it using the parameters in params
    - test creation of training project
    - can be done from GUI
    '''
    cameras = cameras or params.default_cameras

    if isinstance(sessions, str):
        sessions = [sessions]

    if isinstance(cameras, str):
        cameras = [cameras]


    # Run sleap-train for each session and camera combination
    for camera in cameras:
        # Define paths for model and labels
        # model_dir = os.path.join(params.slp_models_dir, camera)
        training_dir = f"{params.slp_training_dir}/{camera}"
        os.makedirs(training_dir, exist_ok=True)

        labels_file = f"{params.slp_training_dir}/{camera}.slp"
        config_file = f"{params.slp_training_dir}/models/{camera}/training_config.json"

        # Check if the .slp file exists; if not, run create_training_file
        if not os.path.exists(labels_file):
            logging.info(f"{labels_file} does not exist. Creating training file...")
            create_training_file(camera, sessions)

        # Ensure configuration file exists
        if not os.path.exists(config_file):
            create_training_config_file(config_file)
            logging.info(f"Configuration file for {camera} created.")
        
        # Run sleap-train command
        logging.info(f"Training model for {camera}...")
        command = ["sleap-train", config_file, labels_file]
        result = subprocess.run(command, cwd=training_dir)
        
        if result.returncode == 0:
            logging.info(f"Finished training for {camera}.")
        else:
            logging.info(f"Training failed for {camera}.")

    logging.info("All training has been executed.")



def select_frames_to_predict():
    return 


def get_2Dpredictions(sessions = params.sessions_to_predict, cameras = params.default_cameras,frames = params.frames_to_predict,input_file = None, output_file = None, model_path = None):
    """
    Runs sleap-track on a list of sessions and cameras.
    -------

    """
    logging.info("Running get_2Dpredictions...")
    ## check if the output folder exists

    # if the arguments are passes as None from cli:
    sessions = sessions or params.sessions_to_predict
    cameras = cameras or params.default_cameras
    frames = frames or params.frames_to_predict

    if isinstance(sessions, str):
        sessions = [sessions]
    if isinstance(cameras, str):
        cameras = [cameras]
    try:
        os.makedirs(os.path.dirname(params.predictions_dir), exist_ok=True)
    except Exception as e:
        logging.error(f"Failed to create predictions directory: {e}")
        return

        
    tracking_options = params.tracking_options

    ## If input, model, output files are specified, run directly - could remove this
    if input_file is not None:
        if os.path.isfile(input_file) and output_file is not None and model_path is not None:
            command = [
                        "sleap-track",
                        input_file,
                        "--video.index", "0",
                        "-m", model_path,
                        "-o", output_file
                    ]
                    
            # Add frames to predict on if specified - otherwise all frames
            if frames:
                command.extend(["--frames", frames])
            
            # Add tracking options if specified
            if tracking_options:
                command.extend(tracking_options.split())
            logging.info(f"Running sleap-track")
            logging.info(f"Input file: {input_file}")
            logging.info(f"Output file: {output_file}")
            # Run the sleap-track command using subprocess
            subprocess.run(command, check=True)
            logging.info(f"Tracking completed\n")
        
        else:
            input_dir = input_file
            for camera in cameras:
                model_dir = f"{params.slp_models_dir}/{camera}"
                if not os.path.exists(model_dir):
                    logging.info(f"Model directory for {camera} does not exist, skipping.")
                    continue
                model_path = f"{model_dir}/training_config.json"
                input_file = f"{input_dir}/{params.camera_name_mapping.get(camera, camera)}.avi"
                output_file = f"{params.predictions_dir}/{sessions[0]}_{camera}.slp.predictions.slp"


                logging.info(f"Running sleap-track for camera {camera}")
                logging.info(f"Input file: {input_file}")
                logging.info(f"Output file: {output_file}")

                # construct sleap-track command 

                command = [
                    "sleap-track",
                    input_file,
                    "--video.index", "0",
                    "-m", model_path,
                    "-o", output_file
                ]
                
                # Add frames to predict on if specified - otherwise all frames
                if frames:
                    command.extend(["--frames", frames])
                
                # Add tracking options if specified
                if tracking_options:
                    command.extend(tracking_options.split())

                # Run the sleap-track command using subprocess
                subprocess.run(command, check=True)
                logging.info(f"Tracking completed for session {sessions[0]}, camera {camera}\n")

    
    ## Otherwise go through the list of sessions and cameras
    else:
        for session in sessions:
            animal = session.split("_")[0]
            for camera in cameras:
                try:
                    model_dir = f"{params.slp_models_dir}/{camera}"
                    if not os.path.exists(model_dir):
                        logging.info(f"Model directory for {camera} does not exist, skipping.")
                        continue
                    model_path = f"{model_dir}/training_config.json"

                    # Different cases for different input directories because different saving formats are used 
                    if "raw" in params.input_2Dpred: 
                        input_file = f"{params.input_2Dpred}/{animal}/{session}/{session}_cameras/{session}_{params.camera_name_mapping.get(camera, camera)}.avi"
                    elif "annotations" in params.input_2Dpred:
                        input_file = f"{params.slp_annotations_dir}/{session}_annotations/{session}_{camera}_annotations/{session}_{camera}.slp"
                    else:
                        input_file = f"{params.input_2Dpred}/{session}/{camera}/{session}_{camera}.slp"
                    output_file = f"{params.predictions_dir}/{session}_{camera}.slp.predictions.slp"


                    logging.info(f"Running sleap-track for session {session} and camera {camera}")
                    logging.info(f"Input file: {input_file}")
                    logging.info(f"Output file: {output_file}")

                    # construct sleap-track command 

                    command = [
                        "sleap-track",
                        input_file,
                        "--video.index", "0",
                        "-m", model_path,
                        "-o", output_file
                    ]
                    
                    # Add frames to predict on if specified - otherwise all frames
                    if frames:
                        command.extend(["--frames", frames])
                    
                    # Add tracking options if specified
                    if tracking_options:
                        command.extend(tracking_options.split())

                    # Run the sleap-track command using subprocess
                    subprocess.run(command, check=True)
                    logging.info(f"Tracking completed for session {session}, camera {camera}\n")

                except Exception as e:
                    logging.error(f"Failed to process session {session}, camera {camera}: {e}")


    return


def visualize_predictions(sessions, cameras = params.default_cameras):
    """
    Launches SLEAP GUI for the predictions slp project for a list of sessions and cameras
    """
    cameras = cameras or params.default_cameras
    if isinstance(sessions, str):
        sessions = [sessions]
    if isinstance(cameras, str):
        cameras = [cameras]
    for session in sessions:
        for camera in cameras:
            predictions_path = f"{params.predictions_dir}/{session}_{camera}.slp.predictions.slp"
            subprocess.run(["sleap-label", predictions_path])
    return

