# launch sleap label to visualize predictions

import argparse
from beneuro_pose_estimation import set_logging
from beneuro_pose_estimation.sleap.sleapTools import visualize_predictions

def main():
    # Set up logging
    set_logging("get_2Dpredictions.log")

    # Parse arguments
    parser = argparse.ArgumentParser(description="Run 2D predictions with SLEAP")
    parser.add_argument("--sessions", nargs="+", required=True, help="List of sessions to process")
    parser.add_argument("--cameras", nargs="*", help="List of cameras to process")
    args = parser.parse_args()

    # Run 2D predictions
    visualize_predictions(sessions=args.sessions, cameras=args.cameras)

if __name__ == "__main__":
    main()