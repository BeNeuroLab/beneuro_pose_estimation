from pathlib import Path

import typer

from beneuro_pose_estimation.config import _check_config, _get_package_path, \
    _check_is_git_track, _check_root, _get_env_path
from beneuro_pose_estimation.sleap.sleapTools import annotate_videos

# Create a Typer app
app = typer.Typer()

# ================================== Functionality =========================================

def annotate(
    session_name: str = typer.Argument(..., help="The first number.")
    camera: str,
    pred: bool
):
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
    check_for_updates()


@app.command()
def self_update():
    """
    Update the bnd tool by pulling the latest commits from the repo's main branch.
    """
    update_bnd()

# ================================= Initialiation ==========================================

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

        print("[green]Config file created successfully.")

# Main Entry Point
if __name__ == "__main__":
    app()