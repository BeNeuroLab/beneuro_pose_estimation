# Beneuro Pose Estimation
Tools to centralize and run 3d pose estimation from data recorded in the lab. Eventually 
this tool will be integrated into `bnd`, our main data organization tool found [here](https://github.com/BeNeuroLab/beneuro_experimental_data_organization).

## Installation
Installing the dependencies of this project is not a straighforward task at all. This is 
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
3. Creating the conda environment
  
    Open miniconda or miniforge prompt:
      ```shell
      # Install bash in base environment
      conda install -c conda-forge m2-base
   
      # Run the environment creation script
      bash setup_env.sh
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