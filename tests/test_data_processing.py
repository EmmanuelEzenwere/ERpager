import unittest
import pandas as pd
from io import StringIO
from your_script_name import load_data, clean_data, save_data  # Adjust the import based on your script name

class TestDataProcessing(unittest.TestCase):
    
    def setUp(self):
        # Sample CSV data for testing
        self.messages_data = """id,message
        1,Hello World
        2,Machine Learning is fun
        3,Unit testing is important
        """
        self.categories_data = """id,categories
        1,related;request;1;0;0
        2,related;offer;0;1;0
        3,request;related;0;0;1
        """
        # Create DataFrames
        self.messages_filepath = 'messages_test.csv'
        self.categories_filepath = 'categories_test.csv'
        self.df_messages = pd.read_csv(StringIO(self.messages_data))
        self.df_categories = pd.read_csv(StringIO(self.categories_data))

        # Save to CSV for testing
        self.df_messages.to_csv(self.messages_filepath, index=False)
        self.df_categories.to_csv(self.categories_filepath, index=False)

    def test_load_data(self):
        """Test loading data from CSV files."""
        df = load_data(self.messages_filepath, self.categories_filepath)
        self.assertEqual(df.shape[0], 3)  # Check if 3 rows are loaded
        self.assertIn('message', df.columns)  # Check if 'message' column exists
        self.assertIn('categories', df.columns)  # Check if 'categories' column exists

    def test_clean_data(self):
        """Test cleaning of data."""
        df = load_data(self.messages_filepath, self.categories_filepath)
        cleaned_df = clean_data(df)
        self.assertIn('related', cleaned_df.columns)  # Check if the category 'related' is present
        self.assertIn('request', cleaned_df.columns)  # Check if the category 'request' is present
        self.assertTrue((cleaned_df['related'].isin([0, 1])).all())  # Check binary values in 'related' column

    def test_save_data(self):
        """Test saving data to SQLite database."""
        df = load_data(self.messages_filepath, self.categories_filepath)
        cleaned_df = clean_data(df)
        database_path = 'test_database.db'
        
        # Use a context manager to avoid leaving the database open
        with self.assertRaises(Exception):
            save_data(cleaned_df, database_path)

    def tearDown(self):
        import os
        os.remove(self.messages_filepath)  # Remove test CSV files
        os.remove(self.categories_filepath)

if __name__ == '__main__':
    unittest.main()
    
    
print(tokenize('There.'))
print(WordNetLemmatizer().lemmatize('there'))
'there' in set(stopwords.words("english"))