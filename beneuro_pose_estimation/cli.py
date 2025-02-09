from pathlib import Path

import typer

from rich import print

from beneuro_pose_estimation import params, set_logging
from beneuro_pose_estimation.config import _check_config, _get_package_path, \
    _check_is_git_track, _check_root, _get_env_path
from beneuro_pose_estimation.sleap.sleapTools import annotate_videos
from beneuro_pose_estimation.update_bnp import check_for_updates, update_bnp

# Create a Typer app
app = typer.Typer(
    add_completion=False,  # Disable the auto-completion options
)

logger = set_logging(__name__)

# ================================== Functionality =========================================


@app.command()
def annotate(
    session_name: str = typer.Argument(..., help="Session name to annotate"),
    camera: str = typer.Argument(..., help=f"Camera name to annotate. Must be part of {params.default_cameras}"),
    pred: bool = typer.Option(True, "--pred/--no-pred", help="Run annotation on prediction or not.", ),
):
    """
    Annotate sleap project
    """
    annotate_videos(
        sessions=session_name,
        cameras=camera,
        pred=pred)

    return

def create_annotation_project():
    return

def run_pose_estimation():
    return

def get_2d_predictions():
    return

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
