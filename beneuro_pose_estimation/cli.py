from pathlib import Path

import typer
from typing import List, Optional
from rich import print

from beneuro_pose_estimation import params, set_logging
from beneuro_pose_estimation.config import _check_config, _get_package_path, \
    _check_is_git_track, _check_root, _get_env_path
# from beneuro_pose_estimation.sleap.sleapTools import annotate_videos, get_2Dpredictions
from beneuro_pose_estimation.update_bnp import check_for_updates, update_bnp
from pathlib import Path
# Create a Typer app
app = typer.Typer(
    add_completion=False,  # Disable the auto-completion options
)

logger = set_logging(__name__)

# ================================== Functionality =========================================


@app.command()
def annotate(
    session: str = typer.Argument(..., help="Session name to annotate"),
    camera: str = typer.Argument(..., help=f"Camera name to annotate. Must be part of {params.default_cameras}"),
    pred: bool = typer.Option(True, "--pred/--no-pred", help="Run annotation on prediction or not." )
):
    """
    Create annotation project for the session if it doesn't exist and launch annotation GUI.
    """
    from beneuro_pose_estimation.sleap.sleapTools import annotate_videos
    annotate_videos(
        sessions=session,
        cameras=camera,
        pred=pred)

    return

@app.command()
def create_annotation_projects(
    sessions: List[str] = typer.Argument(
        ..., help="Session name(s) to annotate. Provide as a single session name or a list of session names."
    ),
    cameras: List[str] = typer.Option(
        None, "--cameras", "-c", help=f"Camera name(s) to annotate. Provide as a single camera name or a list of camera names. Defaults to {params.default_cameras} if not specified."
    ),
    pred: bool = typer.Option(True, "--pred/--no-pred", help="Run annotation on prediction or not." )
    ):
    """
    Create annotation projects for a list of sessions and cameras without launching the GUI.
    """
    from beneuro_pose_estimation.sleap.sleapTools import create_annotation_projects
    create_annotation_projects(sessions, cameras,pred)
    return

@app.command()
def pose(
    sessions: List[str] = typer.Argument(
        ..., help="Session name(s) to run pose estimation on. Provide as a single session name or a list of session names."
    ),
    custom_model_name: Optional[str] = typer.Option(
        None,
        "--model-name", "-m",
        help="Optional custom model name to override the default."
    ),

):
    from beneuro_pose_estimation.anipose.aniposeTools import run_pose_estimation

    run_pose_estimation(sessions, custom_model_name)

    return

@app.command()
def track_2d(
    sessions: List[str] = typer.Argument(
        ..., help="Session name(s) to track. Provide as a single session name or a list of session names."
    ),
    cameras: List[str] = typer.Option(
        None, "--cameras", "-c", help=f"Camera name(s) to track. Provide as a single camera name or a list of camera names. Defaults to {params.default_cameras} if not specified."
    ),
):
    """
    Get 2D predictions for a list of sessions and cameras
    """
    if cameras is None:
        logger.info(f"No cameras specified. Predictions will be run on all default cameras: {params.default_cameras}")
    from beneuro_pose_estimation.sleap.sleapTools import get_2Dpredictions
    get_2Dpredictions(
        sessions=sessions,
        cameras = cameras

    )
    
    return

@app.command()
def create_training_project(
    camera: str = typer.Argument(..., help=f"Camera name to annotate. Must be part of {params.default_cameras}"),
    sessions: List[str] = typer.Argument(
        ..., help="Session name(s) to annotate. Provide as a single session name or a list of session names."
    )
    ):
    """
    Create annotation projects for a list of sessions and cameras without launching the GUI.
    """
    from beneuro_pose_estimation.sleap.sleapTools import create_training_file
    create_training_file(camera,sessions)
    return


@app.command()
def cleanup(
    session: str = typer.Argument(..., help="Session name to clean up intermediate files for."),
  
):
    """
    Clean up intermediate files for a session.
    By default, only asks about cleaning up triangulation files.
    Use --slp flag to also clean up 2D prediction .slp files.
    """
    from beneuro_pose_estimation.tools import cleanup_intermediate_files
    
    cleanup_intermediate_files(session)
    
    return



@app.command()
def model_up(
    test_folder_name: str = typer.Argument(
        ..., 
        help="Name of the test folder under config.models/<camera>/ to copy to remote_models"
    )
):
    """
    Recursively copy a model test folder from your local models path to the remote_models path,
    prompting if it already exists.
    """
    from beneuro_pose_estimation.tools import copy_model_to_remote
    copy_model_to_remote(test_folder_name)

@app.command()
def eval_report(
    session_name: str = typer.Argument(..., help="Session name to evaluate"),
    test_name: Optional[str] = typer.Option(None, "--test-name", "-t", help="Test name to evaluate"),
):
    """
    TBD
    Generate a comprehensive evaluation report for a session, including:
    - Mean confidence scores per camera and per body point
    - Reprojection errors per camera and per body point
    - Detection percentages per camera and per body point
    - Joint angle statistics
    - Missing frame statistics
    """
    from beneuro_pose_estimation.evaluation import generate_evaluation_report
    from pathlib import Path
    import json

    report = generate_evaluation_report(session_name, test_name)

    return

@app.command()
def pose_test(
    session: str = typer.Argument(..., help="Session name to run test pose estimation on."),
    test_name: Optional[str] = typer.Option(
        None,
        "--test-name", "-n",
        help="An optional name for this pose test run."
    ),
    cameras: List[str] = typer.Option(
        None, 
        "--cameras", "-c",
        help="Cameras to process. If not provided, uses default cameras from params."
    ),
    force_new: bool = typer.Option(
        False,
        "--force-new", "-f",
        help="Force creation of new test videos even if they exist."
    ),
    start_frame: Optional[int] = typer.Option(
        None,
        "--start-frame", "-s",
        help="Frame number to start from. If not specified, uses frame 0."
    ),
    duration: Optional[int] = typer.Option(
        10,
        "--duration", "-d",
        help="Duration in seconds. If not specified, uses 100 frames."
    )
):
    """
    Run pose estimation pipeline on short test videos.
    
    This command creates short test videos from each camera, runs the full pose pipeline on them,
    and generates an evaluation report. The test videos can be either newly created or reused
    if they already exist.
    """
    from beneuro_pose_estimation.anipose.aniposeTools import run_pose_test

    run_pose_test(
        session=session,
        test_name=test_name,
        cameras=cameras or params.default_cameras,
        force_new_videos=force_new,
        start_frame=start_frame,
        duration_seconds=duration,
    )

    return

@app.command()
def train(
    cameras: Optional[List[str]] = typer.Option(
        None,
        "--cameras", "-c",
        help=(
            "List of camera names to train models for. "
            "If not provided, uses default cameras from params.default_cameras."
        ),
    ),
    custom_labels: bool = typer.Option(
        False,
        "--custom-labels", "-cl",
        help="If set, use custom labels when training the models.",
    ),
):
    """
    Train SLEAP models for specified cameras (or all defaults).
    """
    from beneuro_pose_estimation.sleap.sleapTools import train_models

    cams = cameras or params.default_cameras
    train_models(cameras=cams, custom_labels=custom_labels)
# =================================== Updating ==========================================


@app.command()
def check_updates():
    """
    Check if there are any new commits on the repo's main branch.
    """
    logger.info('test_message')

    check_for_updates()


@app.command()
def self_update():
    """
    Update the bnd tool by pulling the latest commits from the repo's main branch.
    """
    update_bnp()

# ================================= Initialization =========================================

@app.command()
def init():
    """
    Create a .env file to store the paths to the local and remote data storage.
    """

    # check if the file exists
    env_path = _get_env_path()

    if env_path.exists():
        print("\n[yellow]Config file already exists.\n")

        _check_config()

    else:
        print("\nConfig file doesn't exist. Let's create one.")
        repo_path = _get_package_path()
        _check_is_git_track(repo_path)

        local_path = Path(
            typer.prompt(
                "Enter the absolute path to the root of the local data storage"
            )
        )
        _check_root(local_path)

        remote_path = Path(
            typer.prompt("Enter the absolute path to the root of remote data storage")
        )
        _check_root(remote_path)

        with open(env_path, "w") as f:
            f.write(f"REPO_PATH = {repo_path}\n")
            f.write(f"LOCAL_PATH = {local_path}\n")
            f.write(f"REMOTE_PATH = {remote_path}\n")

        # make sure that it works
        _check_config()

        print("\n[green]Config file created successfully.\n")


# Main Entry Point
if __name__ == "__main__":
    app()
