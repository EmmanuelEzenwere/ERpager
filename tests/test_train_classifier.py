# tests/test_train_classifier.py
import unittest
import sys
import os
import numpy as np
import pandas as pd
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
import pickle

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.train_classifier import (
    load_data, tokenize, StartingVerbExtractor, calculate_multiclass_accuracy,
    per_class_accuracy, build_model, save_model
)

class TestTrainClassifier(unittest.TestCase):
    """Test cases for disaster response model training functions"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures including sample database"""
        cls.test_data_dir = os.path.join(os.path.dirname(__file__), 'test_data')
        cls.database_path = os.path.join(cls.test_data_dir, 'test_db.db')
        os.makedirs(cls.test_data_dir, exist_ok=True)
        
        # Create a small test database
        from sqlalchemy import create_engine
        engine = create_engine(f'sqlite:///{cls.database_path}')
        
        # Create sample cleaned data
        test_data = pd.DataFrame({
            'id': [1, 2, 3],
            'message': [
                'Water needed urgently!',
                'Medical supplies required immediately',
                'Food and shelter needed'
            ],
            'original': [
                'Water needed urgently!',
                'Medical supplies required immediately',
                'Food and shelter needed'
            ],
            'genre': ['direct', 'direct', 'news'],
            'related': [1, 1, 1],
            'request': [1, 1, 1],
            'offer': [0, 0, 0],
            'medical_help': [0, 1, 0],
            'water': [1, 0, 0],
            'food': [0, 0, 1]
        })
        
        test_data.to_sql('cleandata', engine, index=False, if_exists='replace')

    def test_load_data(self):
        """Test data loading functionality"""
        X, Y, category_names = load_data(
            database_path=self.database_path,
            x_column_name="message",
            y_start=4
        )
        
        # Test data shapes
        self.assertEqual(len(X), 3)  # 3 messages
        self.assertEqual(Y.shape, (3, 6))  # 3 samples, 6 categories
        self.assertEqual(len(category_names), 6)  # 6 category names
        
        # Test data content
        self.assertTrue("Water needed urgently!" in X)
        self.assertTrue("related" in category_names)
        self.assertTrue("water" in category_names)

    def test_tokenize(self):
        """Test text tokenization"""
        test_cases = [
            {
                'input': 'Water needed urgently! http://example.com',
                'expected': ['water', 'need', 'urgent']
            },
            {
                'input': 'Medical supplies required IMMEDIATELY!!!',
                'expected': ['medical', 'supply', 'require', 'immediate']
            },
            {
                'input': '',  # Empty text
                'expected': []
            }
        ]
        
        for case in test_cases:
            tokens = tokenize(case['input'])
            self.assertEqual(tokens, case['expected'])

    def test_starting_verb_extractor(self):
        """Test StartingVerbExtractor functionality"""
        extractor = StartingVerbExtractor()
        
        test_cases = [
            ('Need medical help!', 1),  # Starts with verb
            ('The building collapsed', 0),  # Starts with article
            ('RT @user: Emergency!', 1),  # Starts with RT
            ('', 0),  # Empty text
            ('!!!', 0)  # Only punctuation
        ]
        
        for text, expected in test_cases:
            result = extractor.starting_verb(text)
            self.assertEqual(
                result, 
                expected, 
                f"Failed for text: '{text}'. Expected {expected}, got {result}"
            )

    def test_calculate_multiclass_accuracy(self):
        """Test accuracy calculation for multi-class classification"""
        y_true = np.array([
            [1, 0, 1],
            [0, 1, 0],
            [1, 1, 1]
        ])
        y_pred = np.array([
            [1, 0, 1],
            [0, 1, 0],
            [1, 0, 1]
        ])
        
        accuracy = calculate_multiclass_accuracy(y_true, y_pred)
        self.assertIsInstance(accuracy, float)
        self.assertTrue(0 <= accuracy <= 1)

    def test_per_class_accuracy(self):
        """Test per-class accuracy calculation"""
        y_true = np.array([
            [1, 0, 1],
            [0, 1, 0],
            [1, 1, 1]
        ])
        y_pred = np.array([
            [1, 0, 1],
            [0, 1, 0],
            [1, 0, 1]
        ])
        class_names = ['class1', 'class2', 'class3']
        
        accuracies_df = per_class_accuracy(y_true, y_pred, class_names)
        
        self.assertIsInstance(accuracies_df, pd.DataFrame)
        self.assertEqual(len(accuracies_df), 3)
        self.assertTrue(all(0 <= acc <= 1 for acc in accuracies_df['accuracy']))

    def test_build_model(self):
        """Test model building functionality"""
        model = build_model()
        
        # Check if model has the correct structure
        self.assertTrue(hasattr(model, 'estimator'))
        self.assertTrue(hasattr(model, 'param_grid'))
        
        # Check if the pipeline has the correct steps
        pipeline_steps = model.estimator.named_steps
        self.assertIn('features', pipeline_steps)
        self.assertIn('clf', pipeline_steps)

    def test_save_model(self):
        """Test model saving functionality"""
        # Create a simple model to save
        model = MultiOutputClassifier(RandomForestClassifier(n_estimators=10))
        model_path = os.path.join(self.test_data_dir, 'test_model.pkl')
        
        # Save the model
        save_model(model, model_path)
        
        # Check if file exists
        self.assertTrue(os.path.exists(model_path))
        
        # Try to load the model to verify it was saved correctly
        with open(model_path, 'rb') as f:
            loaded_model = pickle.load(f)
        
        self.assertIsInstance(loaded_model, MultiOutputClassifier)

    def tearDown(self):
        """Clean up after each test"""
        pass

    @classmethod
    def tearDownClass(cls):
        """Clean up all test fixtures"""
        # Remove test database and model files
        test_files = [
            cls.database_path,
            os.path.join(cls.test_data_dir, 'test_model.pkl')
        ]
        
        for file_path in test_files:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
            except (OSError, PermissionError) as e:
                print(f"Error removing file {file_path}: {e}")
        
        # Try to remove test directory
        try:
            os.rmdir(cls.test_data_dir)
        except (OSError, PermissionError) as e:
            print(f"Error removing directory {cls.test_data_dir}: {e}")

if __name__ == '__main__':
    unittest.main()