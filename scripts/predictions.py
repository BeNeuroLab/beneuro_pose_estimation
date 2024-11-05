# get 2D predictions

import argparse
from beneuro_pose_estimation import set_logging
from beneuro_pose_estimation.sleap.sleapTools import get_2Dpredictions

def main():
    # Set up logging
    set_logging("get_2Dpredictions.log")

    # Parse arguments
    parser = argparse.ArgumentParser(description="Run 2D predictions with SLEAP")
    parser.add_argument("--sessions", nargs="+", required=True, help="List of sessions to process")
    parser.add_argument("--cameras", nargs="*", help="List of cameras to process")
    parser.add_argument("--frames", nargs="*", help="Specific frames to predict on")
    args = parser.parse_args()

    # Run 2D predictions
    get_2Dpredictions(sessions=args.sessions, cameras=args.cameras, frames=args.frames)

if __name__ == "__main__":
    main()