# annotate

import argparse
from beneuro_pose_estimation import set_logging
from beneuro_pose_estimation.sleap.sleapTools import annotate_video

def main():
    set_logging("annotate_videos.log")

    parser = argparse.ArgumentParser(description="Annotate video frames for SLEAP")
    parser.add_argument("--sessions", nargs="+", required=True, help="List of sessions to annotate")
    parser.add_argument("--cameras", nargs="*", help="List of cameras to annotate")
    args = parser.parse_args()

    # Run annotation
    annotate_videos(sessions=args.sessions, cameras=args.cameras)

if __name__ == "__main__":
    main()