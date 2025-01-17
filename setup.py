from setuptools import find_packages, setup

setup(
    name="beneuro_pose_estimation",  # The name of your package
    version="0.1.0",  # Version of your package
    packages=find_packages(),  # Automatically find all packages in your project
    install_requires=[
        # Add any additional Python dependencies here
        # These should match what you've included in your environment file
        "opencv-contrib-python<4.7.0",
        "sleap_anipose",
        "anipose<1.1",
        "apptools",
    ],
    entry_points={
        "console_scripts": [
            "bnp=bnp.cli:app",  # Register the CLI command
        ],
    },
    python_requires=">=3.7",
)


