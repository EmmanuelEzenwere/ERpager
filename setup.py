from setuptools import setup, find_packages
import os

# Read requirements from requirements.txt
def read_requirements(filename="requirements.txt"):
    with open(filename) as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]

# Ensure packages can be found
packages = find_packages(include=['app', 'app.*', 'models', 'models.*'])

setup(
    name="disaster_response_pipeline",
    version="0.1",
    packages=packages,
    include_package_data=True,  # Include non-Python files like templates
    install_requires=read_requirements(),
    entry_points={
        'console_scripts': [
            'disaster-response=app.run:main' # execute the main() function in run.py
        ],
    },
    package_data={
        'app': [
            'templates/*.html',  # Include HTML templates
            'static/*',         # Include static files if any
            'models/*.pkl',     # Include model files
            'data/*.db'         # Include database files
        ]
    },
    python_requires='>=3.6',
)