# train models
import argparse
from beneuro_pose_estimation import set_logging
from beneuro_pose_estimation.sleap.sleapTools import train_models

def main():
    set_logging("train_models.log")

    parser = argparse.ArgumentParser(description="Train SLEAP models")
    parser.add_argument("--sessions", nargs="+", required=True, help="List of sessions for training")
    parser.add_argument("--cameras", nargs="*", help="List of cameras for training")
    args = parser.parse_args()

    # Run model training
    train_models(sessions=args.sessions, cameras=args.cameras)

if __name__ == "__main__":
    main()