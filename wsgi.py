# from app.run import app

# if __name__ == "__main__":
#     app.run()


# import sys
# import os

# # Add the project root to the Python path
# PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# sys.path.append(PROJECT_ROOT)

# from app.run import app

# if __name__ == "__main__":
#     app.run()
    
import os
import sys
# Add the project root directory to the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from app.run import app