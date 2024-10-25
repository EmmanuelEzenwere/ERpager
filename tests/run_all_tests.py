import unittest
import sys
import os

def run_all_tests():
    """Discover and run all tests in the tests directory"""
    # Get the directory containing this file
    test_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Discover and run tests
    loader = unittest.TestLoader()
    suite = loader.discover(test_dir, pattern='test_*.py')
    
    # Run tests with verbosity
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Return 0 if all tests passed, 1 if any failed
    return 0 if result.wasSuccessful() else 1

if __name__ == '__main__':
    sys.exit(run_all_tests())