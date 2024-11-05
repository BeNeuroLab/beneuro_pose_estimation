# 3D pose estimation
import argparse
from beneuro_pose_estimation import set_logging
from beneuro_pose_estimation.anipose.aniposeTools import run_pose_estimation

def main():
    set_logging("pose_estimation.log")

    parser = argparse.ArgumentParser(description="Run full pose estimation")
    parser.add_argument("--sessions", nargs="+", required=True, help="List of sessions to run pose estimation on")
    args = parser.parse_args()

    # Run pose estimation
    run_pose_estimation(sessions=args.sessions)

if __name__ == "__main__":
    main()