"""
Module for Sleap processing
"""
import sys
import os
import subprocess
from pathlib import Path
project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_path)
import params

import attr
import numpy as np
import random
import json
import matplotlib.pyplot as plt
import cv2

from typing import Dict, List, Optional, Union

from sleap.io.video import Video
from sleap import Labels, Video, LabeledFrame, Instance, Skeleton
from sleap.info.feature_suggestions import (
    FeatureSuggestionPipeline,
    ParallelFeaturePipeline,
)
import sleap
import argparse




def select_frames_to_annotate(session,camera,pipeline = params.frame_selection_pipeline,new_video_path = None):
    """
    - Selects frames to annotate using the feature suggestion pipeline, 
    - Saves them as .png,
    - Creates a new .mp4 video for faster processing 

    """

    # Define input video path
    animal = session.split("_")[0]
    video_path = f"{params.recordings_dir}/{animal}/{session}/{session}_cameras/{session}_{params.camera_name_mapping.get(camera, camera)}.avi"
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
    
    print(f"Selected frames saved for {session}, {camera}")
    
    # create new video from the selected frames
    if new_video_path is None:
       new_video_path = f"{frames_dir}/{session}_{camera}_annotations.mp4"
    create_video_from_frames(frames_dir,new_video_path) 

    
    

    return 

def create_annotation_projects(sessions = params.sessions_to_annotate,cameras = params.default_cameras):
    '''
    create annotation projects for a list of sessions and cameras without launching GUI for annotation
    '''
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
            print(f"Skipping image {image} due to reading error.")

    for image in images:
        image_path = os.path.join(frames_dir, image)
        os.remove(image_path)

    # Release the VideoWriter object
    video.release()

    return



def create_annotation_project(session, camera):
    '''
    Create slp project for annotation to lunch annotation GUI on
    * should we initialize instances for all the frames in the annotation video instead of just the first one?
    '''
    animal = session.split("_")[0]
    # Paths
    video_path = f"{params.recordings_dir}/{animal}/{session}/{session}_cameras/{session}_{params.camera_name_mapping.get(camera, camera)}.avi"
    select_frames_to_annotate(session,camera,params.frame_selection_pipeline)
    labels_output_path = f"{params.slp_annotations_dir}/{session}_annotations/{session}_{camera}_annotations/{session}_{camera}.slp"
    annotations_dir_path = f"{params.slp_annotations_dir}/{session}_annotations/{session}_{camera}_annotations/"
    videos = [vid for vid in os.listdir(annotations_dir_path) if vid.endswith(".mp4")]
    

    # Load skeleton
    with open(params.skeleton_path, 'r') as f:
        skeleton_data = json.load(f)
    skeleton = Skeleton.from_dict(skeleton_data)
    
    # Initialize a list of labeled frames
    labeled_frames = [] = []
    for vid in videos:
        video = Video.from_filename(annotations_dir_path + vid)
        instances = [Instance(skeleton=skeleton)]
        labeled_frame = LabeledFrame(video=video, frame_idx=0, instances=instances)
        labeled_frames.append(labeled_frame)
    labels = Labels(labeled_frames)
    os.makedirs(os.path.dirname(labels_output_path), exist_ok=True)
    labels.save(labels_output_path)
    print(f"Sleap project created for session {session},camera {camera}.")
    

    return 


def create_annotation_project_inefficient(session, camera):
    '''
    create annotation video using the full video (without  creating a new video from the selected frames)
    '''
    print(f"Creating SLEAP project for session {session} and camera {camera}...")
            
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
        print(f"Labeled frame created for {session}, {camera}, frame {frame_idx}")

    # Save the labeled frames to a .slp project file
    labels = Labels(labeled_frames)
    os.makedirs(os.path.dirname(labels_output_path), exist_ok=True)
    labels.save(labels_output_path)
    print(f"Sleap project created for session {session},camera {camera}.")
    
    return 

def create_prediction_project(session, camera):
    '''
    select frames to predict from from a raw video and save them in a slp project to perform tracking
    TBD if needed
    '''

    project_path = f"{params.slp_projects_path}/{session}/{camera}/{session}_{camera}.slp"
    
    
    return 

def annotate_video(sessions = params.sessions_to_annotate, cameras = params.default_cameras ):
    """
    creates slp project from raw video and launches annotation GUI
    should it be for one session one camera at a time?
    ------
    """
    
    if isinstance(sessions, str):
        sessions = [sessions]
    if isinstance(cameras, str):
        cameras = [cameras]
    # for each session, check if a sleap projects exists already or not
    for session in sessions:
        session_dir = f"{params.slp_annotations_dir}/{session}_annotations"
        os.makedirs(session_dir, exist_ok=True)
        for camera in cameras:
            project_dir = f"{session_dir}/{session}_{camera}_annotations"
            project_path = f"{project_dir}/{session}_{camera}.slp"
            os.makedirs(project_dir, exist_ok=True)
            if not os.path.exists(project_path):
                create_annotation_project(session,camera)
    
            print(f"Launching annotation GUI...")
            subprocess.run(["sleap-label", project_path]) # first test if the project is created


def create_training_project(camera, sessions):
    """
    .slp project for a specific camera - merging projects 
    """
    # Path to save the combined training project
    combined_project_path = f"{params.slp_annotations_dir}/{camera}.slp"
    all_labeled_frames = []

    for session in sessions:
        # Define path to the session-specific .slp file
        session_slp_path = f"{params.slp_annotations_dir}/{session}_annotations/{session}_{camera}_annotations/{session}_{camera}.slp"
        
        # Check if the .slp file exists for the session
        if not os.path.exists(session_slp_path):
            print(f"SLP project file {session_slp_path} does not exist. Skipping session {session}.")
            continue

        # Load the .slp project and extract labeled frames
        session_labels = sleap.load_file(session_slp_path)
        session_labeled_frames = session_labels.labeled_frames
        all_labeled_frames.extend(session_labeled_frames)
        print(f"Added {len(session_labeled_frames)} frames from session {session} for camera {camera}.")

    # Create a new Labels object with the combined labeled frames
    combined_labels = Labels(labeled_frames=all_labeled_frames)

    # Ensure the directory for the combined project exists
    os.makedirs(os.path.dirname(combined_project_path), exist_ok=True)

    # Save the combined Labels object to a new .slp file
    combined_labels.save(combined_project_path)
    print(f"Combined training project saved at {combined_project_path}")

    return


def train_models(sessions = params.training_sessions, cameras = params.default_cameras):

    if isinstance(sessions, str):
        sessions = [sessions]

    if isinstance(cameras, str):
        cameras = [cameras]


    # Run sleap-train for each session and camera combination
    for camera in cameras:
        # Define paths for model and labels
        model_dir = os.path.join(params.slp_models_dir, camera)
        labels_file = os.path.join(params.slp_annotations_dir, f"{camera}.slp")
        config_file = os.path.join(model_dir, "single_instance.json")

        # Check if the .slp file exists; if not, run create_training_file
        if not os.path.exists(labels_file):
            print(f"{labels_file} does not exist. Creating training file...")
            create_training_project(camera, sessions)

        # Ensure model directory exists
        if not os.path.exists(model_dir):
            print(f"Model directory for {camera} does not exist, skipping.")
            continue
        
        # Run sleap-train command
        print(f"Training model for {camera}...")
        command = ["sleap-train", config_file, labels_file]
        result = subprocess.run(command, cwd=model_dir)
        
        if result.returncode == 0:
            print(f"Finished training for {camera}.")
        else:
            print(f"Training failed for {camera}.")

    print("All training has been executed.")




def evaluate_model(camera):
    """
    TBD
    """
    # Load evaluation metrics
    metrics_path = f"{params.slp_models_dir}/{camera}/metrics.npz"
    metrics = np.load(metrics_path)
    print("Localization Error (50th percentile):", metrics["dist.p50"])
    print("Mean Average Precision (OKS):", metrics["oks_voc.mAP"])

    labels_path = os.path.join(params.slp_annotations_dir, f"{camera}.slp")
    # Load ground truth and predicted labels for comparison
    labels_gt = sleap.load_file(labels_path)
    labels_pr = sleap.load_file(f"{params.slp_models_dir}/{camera}/validation_predictions.slp")

    # Metric calculations
    mean_error, median_error = calculate_localization_error(labels_gt, labels_pr)
    pck = calculate_pck(labels_gt, labels_pr, threshold=5)
    oks = calculate_oks(labels_gt, labels_pr, sigmas=[0.1] * len(labels_gt[0].instances[0].points), image_size=(640, 480))

    print(f"Localization error: {mean_error}, {median_error}")
    print(f"PCK (5-pixel threshold): {pck}")
    print(f"Average OKS: {oks}")


    return

def select_frames_to_predict():
    return 


def get_2Dpredictions(sessions = params.sessions_to_predict, cameras = params.default_cameras,frames = params.frames_to_predict,input_file = None, output_file = None, model_path = None):
    """
    Create? or load camera-wise sleap and compute the 2 predictions of each one
    -------

    """
    print("Running get_2Dpredictions...")
    ## check if the output folder exists
    if isinstance(sessions, str):
        sessions = [sessions]
    if isinstance(cameras, str):
        cameras = [cameras]
    os.makedirs(os.path.dirname(params.predictions_dir), exist_ok=True)
    
        
    tracking_options = params.tracking_options

    ## If input, model, output files are specified, run directly - could remove this
    if input_file is not None and output_file is not None and model_path is not None:
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
        print(f"Tracking completed\n")
    
    ## Otherwise go through the list of sessions and cameras
    else:
        for session in sessions:
            animal = session.split("_")[0]
            for camera in cameras:
                model_dir = f"{params.slp_models_dir}/{camera}"
                if not os.path.exists(model_dir):
                    print(f"Model directory for {camera} does not exist, skipping.")
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


                print(f"Running sleap-track for session {session} and camera {camera}")
                print(f"Input file: {input_file}")
                print(f"Output file: {output_file}")

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
                print(f"Tracking completed for session {session}, camera {camera}\n")

    return


def visualize_predictions(sessions, cameras = params.default_cameras):
    """
    Launches SLEAP GUI for the predictions slp project for a list of 
    """
    if isinstance(sessions, str):
        sessions = [sessions]
    if isinstance(cameras, str):
        cameras = [cameras]
    for session in sessions:
        for camera in cameras:
            predictions_path = f"{params.predictions_dir}/{session}_{camera}.slp.predictions.slp"
            subprocess.run(["sleap-label", predictions_path])
    return


def visualize_prediction_plot(session, camera, frames_to_visualize = None):
    """
    Visualize prediction for a certain session and camera (can change to multiple), and specifc frames
    Plots predictions over each (specified) predicted frame
    """
    # load predictions 
    predictions_path = f"{params.predictions_dir}/{session}_{camera}.slp.predictions.slp"
    labels = sleap.load_file(predictions_path)

    # load video
    animal = session.split("_")[0]
    if "raw" in params.input_2Dpred:
        video_path = f"{params.input_2Dpred}/{animal}/{session}/{session}_cameras/{session}_{params.camera_name_mapping.get(camera, camera)}.avi"
    
    # handle other inputs - TBD
    video = sleap.io.video.Video.from_filename(video_path)
    # iterate over frames with predictions
    for labeled_frame in labels:
        frame_idx = labeled_frame.frame_idx

        # skip frames not in the list
        if  frames_to_visualize is not None and frame_idx not in  frames_to_visualize:
            continue

        # Fetch the specific frame from the video
        frame = video.get_frame(frame_idx)
        
        # Plot grayscale frame
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        plt.imshow(frame_gray, cmap="gray")  # Use grayscale colormap
        
        # Overlay keypoints and skeleton
        for instance in labeled_frame.instances:
            # Draw keypoints
            for node in instance.skeleton.nodes:
                point = instance[node]
                if point is not None:  # Check if the point is valid
                    plt.scatter(point.x, point.y, s=20, c="red", marker="o")  # Draw keypoint in red
            
            # Draw edges between connected nodes based on skeleton structure
            for edge in instance.skeleton.edges:
                pt1 = instance[edge[0]]
                pt2 = instance[edge[1]]
                if pt1 is not None and pt2 is not None:
                    plt.plot([pt1.x, pt2.x], [pt1.y, pt2.y], "r-", linewidth=1)  # Draw skeleton edges

        plt.title(f"Predictions for frame {frame_idx}, session: {session}, camera: {camera}")
        plt.axis("off")  # Hide axis for cleaner visualization
        plt.show()  
        # save image in params.predicition_eval_path
    return


def parse_arguments():
    parser = argparse.ArgumentParser(description="SLEAP Processing Commands")
    subparsers = parser.add_subparsers(dest="command", required=True, help="Select a command to run")

    # Subparser for get_2Dpredictions
    parser_predict = subparsers.add_parser("get_2Dpredictions", help="Run 2D predictions")
    parser_predict.add_argument("--sessions", nargs="+", default=params.sessions_to_predict, help="List of sessions to process")
    parser_predict.add_argument("--cameras", nargs="*", default=params.default_cameras, help="List of cameras to process")
    parser_predict.add_argument("--frames", nargs="*", default=params.frames_to_predict, help="Specific frames to predict on")
    parser_predict.add_argument("--input_file", type=str, help="Optional input file path for predictions")
    parser_predict.add_argument("--output_file", type=str, help="Optional output file path for predictions")
    parser_predict.add_argument("--model_path", type=str, help="Optional path to the SLEAP model for predictions")

    # Subparser for visualize_predictions
    parser_visualize = subparsers.add_parser("visualize_predictions", help="Visualize 2D predictions")
    parser_visualize.add_argument("--sessions", nargs="+", required=True, help="List of sessions to visualize predictions for")
    parser_visualize.add_argument("--cameras", nargs="*", default=params.default_cameras, help="List of cameras to visualize predictions for")

    # Subparser for annotate_video
    parser_annotate = subparsers.add_parser("annotate_video", help="Annotate video frames")
    parser_annotate.add_argument("--sessions", nargs="+", default=params.sessions_to_annotate, help="List of sessions to annotate")
    parser_annotate.add_argument("--cameras", nargs="*", default=params.default_cameras, help="List of cameras to annotate")

    # Subparser for train_models
    parser_train = subparsers.add_parser("train_models", help="Train models for sessions and cameras")
    parser_train.add_argument("--sessions", nargs="+", default=params.training_sessions, help="List of sessions for training")
    parser_train.add_argument("--cameras", nargs="*", default=params.default_cameras, help="List of cameras for training")

    # Subparser for evaluate_model
    parser_evaluate = subparsers.add_parser("evaluate_model", help="Evaluate trained model for a camera")
    parser_evaluate.add_argument("--camera", nargs="*", default=params.default_cameras[0], help="Camera to evaluate")

     # Subparser for create_annotation_projects
    parser_create_annotations = subparsers.add_parser("create_annotation_projects", help="Create annotation projects for sessions and cameras")
    parser_create_annotations.add_argument("--sessions", nargs="+", default=params.sessions_to_annotate, help="List of sessions for creating annotation projects")
    parser_create_annotations.add_argument("--cameras", nargs="*", default=params.default_cameras, help="List of cameras for creating annotation projects")

    return parser.parse_args()

def main():
    args = parse_arguments()

    # Execute the appropriate function based on the command
    if args.command == "get_2Dpredictions":
        get_2Dpredictions(
            sessions=args.sessions,
            cameras=args.cameras,
            frames=args.frames,
            input_file=args.input_file,
            output_file=args.output_file,
            model_path=args.model_path
        )
    elif args.command == "visualize_predictions":
        visualize_predictions(sessions=args.sessions, cameras=args.cameras)
    elif args.command == "annotate_video":
        annotate_video(sessions=args.sessions, cameras=args.cameras)
    elif args.command == "train_models":
        train_models(sessions=args.sessions, cameras=args.cameras)
    elif args.command == "create_annotation_projects":
        create_annotation_projects(sessions=args.sessions, cameras=args.cameras)
    elif args.command == "evaluate_model":
        evaluate_model(camera=args.camera)
    

if __name__ == "__main__":
    main()