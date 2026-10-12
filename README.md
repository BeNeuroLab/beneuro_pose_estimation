# Beneuro Pose Estimation
Tools to centralize and run 3d pose estimation from data recorded in the lab. Eventually 
this tool will be integrated into `bnd`, our main data organization tool found [here](https://github.com/BeNeuroLab/beneuro_experimental_data_organization).

The pipeline uses SLEAP for 2D tracking on raw videos from each camera and Anipose for 3D triangulation, to then save a .csv file with 3D body-keypoint coordinates and joint angles. This is then added to the nwb file of the session and further to the pyaldata .mat file.

## Installation
Installing the dependencies of this project is not that straighforward. This is 
the main reason why we are opting to make it a standalone project hosted in its own 
environment and separate from `bnd`. 

### Build the environment

1. Install `python` 3.7
2. Install `Miniforge` or `Miniconda` if you don't have it already:
    ```shell
    # Miniforge (recommended)
    $ Invoke-WebRequest -Uri "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Windows-x86_64.exe" -OutFile "$env:UserProfile/Downloads/Miniforge3-Windows-x86_64.exe"; Start-Process -FilePath "$env:UserProfile/Downloads/Miniforge3-Windows-x86_64.exe" -ArgumentList "/InstallationType=JustMe /RegisterPython=1 /S" -Wait; Remove-Item -Path "$env:UserProfile/Downloads/Miniforge3-Windows-x86_64.exe"
    ```
   ```shell
   # Miniconda
   $ curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe -o miniconda.exe; Start-Process -FilePath ".\miniconda.exe" -ArgumentList "/S" -Wait; del miniconda.exe
   ```
3. Clone repo and navigate to folder:
   ```shell
   git clone git@github.com:BeNeuroLab/beneuro_pose_estimation.git
   cd ./beneuro_pose_estimation
   ```
4. Creating the conda environment
 
```shell
conda create -y -n bnp -c conda-forge -c nvidia -c sleap/label/dev -c sleap -c anaconda sleap=1.4.1

# Activate the Conda environment
conda activate bnp

# Remove opencv pypi version to avoid conflicts
pip uninstall -y opencv-python-headless

# Install the required version of OpenCV
pip install "opencv-contrib-python<4.7.0"

# Install sleap_anipose and the required version of anipose
pip install sleap_anipose
pip install "anipose<1.1"

# Upgrade apptools to the latest version
pip install --upgrade apptools

# Install package in editable form
pip install -e .\  # Windows

# Install CLI and evaluation dependencies
conda install typer
pip install plotly
```

The key package versions are:
```text
# Name                    Version                   Build  Channel
anipose                   1.0.1                    pypi_0    pypi
aniposelib                0.5.1                    pypi_0    pypi
sleap-anipose             0.1.8                    pypi_0    pypi
opencv-contrib-python     4.6.0.66                 pypi_0    pypi
opencv-python             4.10.0.84                pypi_0    pypi
```

4. Configure paths
```shell
bnd init
```
You will be asked for two paths, which are saved to a local .env file:

`LOCAL_PATH` path to the folder on the local machine (brain or personal computer) that contains a folder called "raw"

`REMOTE_PATH` path to rds: “…/live” (parent of “raw” folder on RDS)

## Quick use

```shell
conda activate bnp
bnp pose <session_name>
```

### End to end
```shell
# 1. Download the session videos and metadata locally
conda activate bnd
bnd dl M0XX_YYYY_MM_DD_HH_MM .avi -v
bnd dl M0XX_YYYY_MM_DD_HH_MM .csv 

# 2. (Recommended) Test on a short clip first and check the result
conda activate bnp
bnp pose-test M0XX_YYYY_MM_DD_HH_MM

# 3. Run pose estimation on the full session (one or several sessions)
bnp pose M0XX_YYYY_MM_DD_HH_MM

# 4. Add the pose data to the pyaldata .mat file and upload to RDS
conda activate bnd
bnd dl M0XX_YYYY_MM_DD_HH_MM      # g0 files are needed to rebuild the NWB
bnd to-pyal M0XX_YYYY_MM_DD_HH_MM  # delete any existing .nwb first; answer "yes" to overwrite .mat
bnd up M0XX_YYYY_MM_DD_HH_MM # delete any existing .nwb and .mat from rds first
```

For details on evaluation, annotations, model training, etc., see SOP.


