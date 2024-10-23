from setuptools import setup, find_packages

setup(
    name="disaster_response_pipeline",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "pandas",       # Add other dependencies your project needs
        "numpy",
        "sqlalchemy",
        "scikit-learn",
    ],
    entry_points={
        'console_scripts': [
            # Add entry points for any scripts you want to run as commands
        ],
    },
)