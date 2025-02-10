"""
Initialize macro variables and functions
"""

from pathlib import Path
from rich import print


def _get_package_path() -> Path:
    """
    Returns the path to the package directory.
    """
    return Path(__file__).absolute().parent.parent


def _get_env_path() -> Path:
    """
    Returns the path to the .env file containing the configuration settings.
    """
    package_path = _get_package_path()
    return package_path / ".env"


def _check_is_git_track(repo_path):
    folder = Path(repo_path)  # Convert to Path object
    assert (folder / ".git").is_dir()


def _check_root(root_path: Path):
    assert root_path.exists(), f"{root_path} does not exist."
    assert root_path.is_dir(), f"{root_path} is not a directory."

    files_in_root = [f.stem for f in root_path.iterdir()]

    assert "raw" in files_in_root, f"No raw folder in {root_path}"


def _check_config():
    """
    Check that the local and remote root folders have the expected raw and processed folders.
    """
    config = _load_config()

    print(
        "Checking that local and remote root folders have the expected raw and processed folders..."
    )

    _check_root(config.LOCAL_PATH)
    _check_root(config.REMOTE_PATH)

    print("[green]Config looks good.")


class Config:
    """
    Class to load local configuration
    """

    def __init__(self, env_path=_get_env_path()):
        self.load_env(env_path)
        self.assign_paths()

    def load_env(self, env_path: Path):
        with open(env_path, "r") as file:
            for line in file:
                # Ignore comments and empty lines
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                # Parse key-value pairs
                key, value = map(str.strip, line.split("=", 1))

                # Set as environment variable
                setattr(self, key, Path(value))

    def assign_paths(self):
        self.recordings_remote = self.REMOTE_PATH / "raw"
        self.annotation_party = self.REMOTE_PATH  / "processed" / "AnnotationParty"
        self.annotations = self.annotation_party / "annotations"
        self.models = self.REMOTE_PATH /"raw"/ "pose-estimation" / "models" / "h1_new_setup" 
        self.skeleton_path = self.REPO_PATH / "beneuro_pose_estimation"/"sleap" / "skeleton.json"
        self.recordings = self.annotation_party # can change to self.recordings_local
        self.predictions2D = self.LOCAL_PATH / "predictions2D"
        self.training = self.REMOTE_PATH / "pose-estimation" / "models" / "uren_setup" 
        self.predictions3D = self.LOCAL_PATH / "predictions3D"
        self.calibration_videos = self.REMOTE_PATH / "raw" / "calibration_videos"
        self.calibration = self.LOCAL_PATH / "calibration_config" 
        return


def _load_config() -> Config:
    """
    Loads the configuration settings from the .env file and returns it as a Config object.
    """
    if not _get_env_path().exists():
        raise FileNotFoundError("Config file not found. Run `bnp init` to create one.")

    return Config()
