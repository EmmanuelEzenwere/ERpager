# tests/test_process_data.py
import unittest
import sys
import os
import pandas as pd
from sqlalchemy import create_engine

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data.process_data import load_data, clean_data, save_data
from models.train_classifier import tokenize

class TestProcessData(unittest.TestCase):
    """Test cases for disaster response data processing functions"""
    
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures including sample CSV files"""
        # Define paths for test files
        cls.test_data_dir = os.path.join(os.path.dirname(__file__), 'test_data')
        cls.messages_filepath = os.path.join(cls.test_data_dir, 'test_messages.csv')
        cls.categories_filepath = os.path.join(cls.test_data_dir, 'test_categories.csv')
        cls.database_filepath = os.path.join(cls.test_data_dir, 'test_db.db')

        # Create test_data directory if it doesn't exist
        os.makedirs(cls.test_data_dir, exist_ok=True)

    def setUp(self):
        """Create sample CSV files before each test"""
        # Create a small sample messages.csv file
        messages_data = """id,message,original,genre
1,"Help! We need water","Help! We need water",direct
2,"Need food and shelter","Need food and shelter",direct
3,"No electricity","No electricity",news
4,"Medical assistance needed","Medical assistance needed",news"""
        
        # Create a small sample categories.csv file
        categories_data = """id,categories
1,"related-1;request-1;aid_related-1;medical_help-0;water-1"
2,"related-1;request-1;aid_related-1;medical_help-0;water-0"
3,"related-1;request-0;aid_related-1;medical_help-0;water-1"
4,"related-1;request-1;aid_related-1;medical_help-1;water-1" """

        # Write test CSV files
        with open(self.messages_filepath, 'w') as f:
            f.write(messages_data)
        
        with open(self.categories_filepath, 'w') as f:
            f.write(categories_data)

    def test_load_data(self):
        """Test if load_data correctly loads and merges the datasets"""
        df = load_data(self.messages_filepath, self.categories_filepath)

        # Test the loaded data structure
        self.assertEqual(len(df), 4)  # Should have 4 rows
        self.assertTrue(all(col in df.columns 
                          for col in ['id', 'message', 'original', 'genre', 'categories']))
        
        # Test data content
        self.assertEqual(df.iloc[0]['message'], 'Help! We need water')
        self.assertEqual(df.iloc[0]['genre'], 'direct')

    def test_clean_data(self):
        """Test if clean_data correctly processes the DataFrame"""
        # First load the data
        df = load_data(self.messages_filepath, self.categories_filepath)
       
        # Clean the data
        cleaned_df = clean_data(df)

        # Test binary values in category columns
        category_columns = [col for col in cleaned_df.columns 
                          if col not in ['id', 'message', 'original', 'genre']]

        for col in category_columns:
            unique_vals = cleaned_df[col].unique()
            self.assertTrue(all(val in [0.0, 1.0] for val in unique_vals),
                          f"Column {col} contains non-binary values: {unique_vals}")
        
        # Test no duplicates
        self.assertEqual(len(cleaned_df), len(cleaned_df.drop_duplicates()))
        
        # Test expected transformations
        self.assertTrue('water' in cleaned_df.columns)
        self.assertEqual(cleaned_df.iloc[0]['water'], 1.0)

    def test_save_data(self):
        """Test if save_data correctly saves the DataFrame to SQLite database"""
        # Load and clean the data
        df = load_data(self.messages_filepath, self.categories_filepath)
        cleaned_df = clean_data(df)
        
        # Save to database
        save_data(cleaned_df, self.database_filepath)
        
        # Verify data was saved correctly
        engine = create_engine(f'sqlite:///{self.database_filepath}')
        saved_df = pd.read_sql_table('cleandata', engine)
        
        self.assertEqual(len(saved_df), len(cleaned_df))
        self.assertTrue(all(col in saved_df.columns for col in cleaned_df.columns))

    def tearDown(self):
        """Clean up test files after each test"""
        # Remove test files
        for filepath in [self.messages_filepath, self.categories_filepath, self.database_filepath]:
            try:
                if os.path.exists(filepath):
                    os.remove(filepath)
            except PermissionError:
                pass  # Handle Windows file lock issues

    @classmethod
    def tearDownClass(cls):
        """Clean up test directory after all tests"""
        try:
            os.rmdir(cls.test_data_dir)
        except (OSError, PermissionError):
            pass  # Directory might not be empty or might be locked

class TestTextProcessing(unittest.TestCase):
    """Test cases for text processing functions"""
    
    def setUp(self):
        """Load sample messages from test CSV"""
        messages_filepath = os.path.join(
            os.path.dirname(__file__), 
            'test_data/test_messages.csv'
        )
        if os.path.exists(messages_filepath):
            self.test_df = pd.read_csv(messages_filepath)
        else:
            self.test_df = pd.DataFrame({
                'message': [
                    'Help! Need water.',
                    'We need medical supplies and food immediately!!',
                    'Need 100 blankets at shelter 5'
                ]
            })

    def test_tokenize(self):
        """Test if tokenize correctly processes text"""
        
        # Test first message
        tokens = tokenize(self.test_df['message'].iloc[0])
        self.assertTrue(all(isinstance(token, str) for token in tokens))
        self.assertTrue(len(tokens) > 0)

    def test_tokenize_empty(self):
        """Test tokenize with empty input"""
        self.assertEqual(tokenize(''), [])

    def tearDown(self):
        """Clean up test resources"""
        # Clear the test DataFrame
        self.test_df = None

if __name__ == '__main__':
    unittest.main(verbosity=2)