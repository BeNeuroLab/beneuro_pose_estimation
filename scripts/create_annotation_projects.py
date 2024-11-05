# create annotation projects for a list of sessions and cameras 
import argparse
from beneuro_pose_estimation import set_logging
from beneuro_pose_estimation.sleap.sleapTools import create_annotation_projects


def main():
    set_logging("create_annotation_projects.log")

    parser = argparse.ArgumentParser(description="Create annotation projects for SLEAP")
    parser.add_argument("--sessions", nargs="+", required=True, help="List of sessions to create annotation projects for")
    parser.add_argument("--cameras", nargs="*", help="List of cameras for creating annotation projects")
    args = parser.parse_args()

    # Create annotation projects
    create_annotation_projects(sessions=args.sessions, cameras=args.cameras)

if __name__ == "__main__":
    main()