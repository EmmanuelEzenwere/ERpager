import sys
import unittest
import pandas as pd
import os
from sqlalchemy import create_engine
from data import load_data
from test_train_classifier import chicken_duties
from data.process_data import load_data
from data.process_data import save_data


class TestProcessData(unittest.TestCase):

    def setUp(self):
        """Set up sample data for testing."""
        # Sample messages and categories data
        self.messages_data = {
            'id': [1, 2, 3],
            'message': ['Hello', 'Help', 'Goodbye'],
            'original': ['Hola', 'Ayuda', 'Adiós'],
            'genre': ['social', 'news', 'direct']
        }
        self.categories_data = {
            'id': [1, 2, 3],
            'categories': ['related-1;request-0;offer-0', 
                           'related-1;request-1;offer-0', 
                           'related-0;request-0;offer-1']
        }

        # Create DataFrames from sample data
        self.messages_df = pd.DataFrame(self.messages_data)
        self.categories_df = pd.DataFrame(self.categories_data)

    def test_load_data(self):
        """Test loading and merging data."""
        # Create temporary CSV files for messages and categories
        self.messages_df.to_csv('test_messages.csv', index=False)
        self.categories_df.to_csv('test_categories.csv', index=False)
        
        # Load and merge data
        df = load_data('test_messages.csv', 'test_categories.csv')
        
        # Test the shape and contents of the combined dataframe
        self.assertEqual(df.shape[0], 3)  # Check the number of rows
        self.assertIn('message', df.columns)  # Check for specific column
        self.assertIn('categories', df.columns)  # Check for specific column
        
        # Clean up temporary files
        os.remove('test_messages.csv')
        os.remove('test_categories.csv')

    def test_clean_data(self):
        """Test cleaning of data."""
        # Merge the messages and categories first
        df = pd.merge(self.messages_df, self.categories_df, on="id")
        
        # Clean the merged dataframe
        cleaned_df = clean_data(df)
        
        # Test the new columns are binary and correctly converted
        self.assertEqual(cleaned_df.shape[0], 3)  # Check the number of rows
        self.assertIn('related', cleaned_df.columns)  # Check for expanded categories
        self.assertTrue(all(cleaned_df['related'].isin([0, 1])))  # Ensure binary conversion
        
        # Test that there are no duplicates
        self.assertFalse(cleaned_df.duplicated().any())

    def test_save_data(self):
        """Test saving data to a SQLite database."""
        # Sample cleaned dataframe
        cleaned_df = clean_data(pd.merge(self.messages_df, self.categories_df, on="id"))
        
        # Save to an SQLite database
        save_data(cleaned_df, 'test_database.db')
        
        # Test that the database and table exist
        engine = create_engine("sqlite:///test_database.db")
        table_names = engine.table_names()
        self.assertIn('cleandata', table_names)  # Check that table was created
        
        # Clean up the test database file
        os.remove('test_database.db')

if __name__ == '__main__':
    unittest.main()

    
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


def display_dataset(X_train, y_train, X_test=None, y_test=None):
    """
    """
    print("unique Y values: ", np.unique(Y))
    print("training set, X: ", X_train.shape)
    if X_test is not None:
        print("test set, X: ",X_test.shape)
    print("training set, Y: ",y_train.shape)
    if y_test is not None:
        print("test set, Y: ",y_test.shape)
        
def data_type_check(X1, X2):
    """
    """
    # check data types of 
    print("X1 shape: ", X1.shape)
    print("X2 shape: ", X2.shape)
    print("X1 Type: ", type(X1))
    print("X2 Type: ",type(X2)) 
    
    
text = 'What can I do?'
tokens = tokenize(text)
print(tokens)
for token in word_tokenize(text.lower()):
    print(WordNetLemmatizer().lemmatize(token))
    print(f'{token}, {token in set(stopwords.words("english"))}')
    
    

# print(accuracy(y_test, y_pred))

# # Now you can generate the classification report
# for col_index in range(0,y_test.shape[1]):
#     report = classification_report(y_test[:,col_index], y_pred[:, col_index], zero_division=0)
#     print(classes[col_index])
#     print(report)







#Testing Tokenize Function - Use this to test the output of the tokenize function.

text1 = "Barclaysjbki CEO stresses the importance of regulatory and cultural reform in financial services at Brussels conference  https://www.google.com"
print(f'input text: "{text1}"\n')
print(f"text tokens: {tokenize(text1)} \n")
text2 = "The No. 8 Northeast Gale or storm signal was issued at 5.55pm yesterday (September 14) and was replaced by Southeast gale and storm signal at 12.35am today (September 15)."
print(f'input text: "{text2}" \n')
print(f"text tokens: {tokenize(text2)} \n")
sentence_list = sent_tokenize(text2)
print(f"sentences: {sentence_list} \n")
print("testing sentence tokenization...")
for text in sentence_list:
    print(f'\ntext: "{text}"')
    print(f"\ntext tokens: {tokenize(text)}")
    
    
    