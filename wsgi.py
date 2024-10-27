# from app.run import app

# if __name__ == "__main__":
#     app.run()


import sys
import os

# Add the project root to the Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)

from app.run import app

if __name__ == "__main__":
    app.run()