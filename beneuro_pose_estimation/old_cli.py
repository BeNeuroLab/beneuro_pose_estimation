# command line interface
import argparse

from beneuro_pose_estimation import set_logging
from beneuro_pose_estimation.anipose.aniposeTools import run_pose_estimation
from beneuro_pose_estimation.sleap.sleapTools import (
    annotate_videos,
    create_annotation_projects,
    get_2Dpredictions,
    train_models,
    visualize_predictions,
)
from beneuro_pose_estimation import Config



def main():
    # Initialize logging
    set_logging()

    # Set up main parser
    parser = argparse.ArgumentParser(
        description="Beneuro Pose Estimation Toolkit",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(
        dest="command", required=True, help="Available commands"
    )

    # Subcommand: Annotate video
    annotate_parser = subparsers.add_parser(
        "annotate", help="Annotate video frames for SLEAP"
    )
    annotate_parser.add_argument(
        "--sessions", nargs="+", required=True, help="List of sessions to annotate"
    )
    annotate_parser.add_argument(
        "--cameras", nargs="*", help="List of cameras to annotate"
    )
    annotate_parser.add_argument(
        "--pred",
        action="store_true",
        help="Run predictions before annotating using an existing model",
    )
    # Subcommand: Create annotation projects
    annotation_parser = subparsers.add_parser(
        "create-annotations", help="Create annotation projects for SLEAP"
    )
    annotation_parser.add_argument(
        "--sessions",
        nargs="+",
        required=True,
        help="List of sessions for annotation projects",
    )
    annotation_parser.add_argument(
        "--cameras", nargs="*", help="List of cameras for annotation projects"
    )

    # Subcommand: Run pose estimation
    pose_parser = subparsers.add_parser(
        "pose-estimation", help="Run full 3D pose estimation"
    )
    pose_parser.add_argument(
        "--sessions", nargs="+", required=True, help="List of sessions to process"
    )

    # Subcommand: Run 2D predictions
    predict_parser = subparsers.add_parser(
        "predict-2D", help="Run 2D predictions with SLEAP"
    )
    predict_parser.add_argument(
        "--sessions", nargs="+", required=True, help="List of sessions to process"
    )
    predict_parser.add_argument(
        "--cameras", nargs="*", help="List of cameras to process"
    )
    predict_parser.add_argument(
        "--frames", nargs="*", help="Specific frames to predict on"
    )
    predict_parser.add_argument(
        "--input_file", type=str, help="Input file to video or directory"
    )
    predict_parser.add_argument(
        "--output_file", type=str, help="Output file for predictions"
    )
    predict_parser.add_argument(
        "--model_path", type=str, help="Model configuration file path"
    )

    # Subcommand: Train SLEAP models
    train_parser = subparsers.add_parser("train", help="Train SLEAP models")
    train_parser.add_argument(
        "--sessions", nargs="+", required=True, help="List of sessions for training"
    )
    train_parser.add_argument(
        "--cameras", nargs="*", help="List of cameras for training"
    )

    # Subcommand: Visualize predictions
    visualize_parser = subparsers.add_parser(
        "visualize-2D", help="Launch SLEAP label to visualize predictions"
    )
    visualize_parser.add_argument(
        "--sessions", nargs="+", required=True, help="List of sessions to process"
    )
    visualize_parser.add_argument(
        "--cameras", nargs="*", help="List of cameras to process"
    )

    # Parse arguments
    args = parser.parse_args()

    # Dispatch to the appropriate function
    if args.command == "annotate":
        annotate_videos(sessions=args.sessions, cameras=args.cameras, pred=args.pred)
    elif args.command == "create-annotations":
        create_annotation_projects(sessions=args.sessions, cameras=args.cameras)
    elif args.command == "pose-estimation":
        run_pose_estimation(sessions=args.sessions)
    elif args.command == "predict-2D":
        get_2Dpredictions(
            sessions=args.sessions,
            cameras=args.cameras,
            frames=args.frames,
            input_file=args.input_file,
            output_file=args.output_file,
            model_path=args.model_path,
        )
    elif args.command == "train":
        train_models(sessions=args.sessions, cameras=args.cameras)
    elif args.command == "visualize-2D":
        visualize_predictions(sessions=args.sessions, cameras=args.cameras)
    else:
        parser.print_help()


if __name__ == "__main__":
    # main()
    config = Config()
    print(type(config.REPO_PATH))
