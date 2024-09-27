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

3. Creating the conda environment
  
    This seems to be working:
      ```shell
      # Create the environment called bnp and install sleap
      $ conda create -y -n bnp -c conda-forge -c nvidia -c sleap -c anaconda sleap
      $ conda activate bnp
   
      # Remove opencv pypi version because it conflicts with sleap-anipose and anipose
      $ pip uninstall -y opencv-python-headless 
   
      # Install required version
      $ pip install "opencv-contrib-python<4.7.0" 
   
      # Install sleap anipose and anipose version 1.0 because we cannot use 1.1
      $ pip install sleap_anipose
      $ pip install "anipose<1.1" 
      $ conda install -y mayavi ffmpeg  # Do we need this??
      $ pip install --upgrade apptools
      ```