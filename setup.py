from setuptools import find_packages, setup

setup(
    name="bnp",  # The name of your package
    version="0.1.0",  # Version of your package
    packages=find_packages(),  # Automatically find all packages in your project
    install_requires=[],
    entry_points={
        "console_scripts": [
            "bnp=beneuro_pose_estimation.cli:app",  # Register the CLI command
        ],
    },
    python_requires=">=3.7",
)


