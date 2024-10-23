import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """ 
    load_data extracts message and categories datasets from csv files and 
    loads them in a pandas data frame.
    
    Args:
        messages_filepath (str): file paths for messages csv file
        categories_filepath (str): file paths for categories csv file
        
    Returns:
        combined_df(pandas.DataFrame): concatenated df containing both mssg_df and 
        cat_df
    """
    mssg_df = pd.read_csv(messages_filepath)
    cat_df = pd.read_csv(categories_filepath)
      
    # merge datasets using id as a common key.
    combined_df = pd.merge(mssg_df, cat_df, on="id")
    return combined_df


def clean_data(df):
    """
    clean_data takes an input dataframe, df performs data type conversions 
    where neccessary, expands the columns of the data frame, and covert the 
    column values to binary and drop duplicate rows. 

    Args:
        df(pandas.DataFrame): _description_
        
    Return:
       cleaned_df(pandas.DataFrame): 
    """
    print("\nBegining Data Cleaning")
    print("-"*100)
    print("\nInitial Combined Dataset (First Five Rows): \n", df.head(5))
    print(f"\nInitial Combined Dataset Data Frame dimensions: rows : "
          f"{df.shape[0]}, col : {df.shape[1]}")

    # Get the df excluding the categories column.
    non_categories_columns = df.drop(columns="categories")
    
    # Create new column labels from the inputs of categories.
    categories_labels = df.categories.values[0].split(";")
    categories_labels = [col_name[:-2] for col_name in categories_labels]
    categories_columns = df.categories.str.split(";", expand=True)
    categories_columns.columns = categories_labels
    print("\nFeature Extraction from Message Column and Data type conversion,"
          "convert category values to binary indicators .................")

    # Convert category values to binary indicators (1 or 0)
    for col_name in categories_labels:
        # replace categories column values with the binary value at the end of 
        # the text value.
        categories_columns[col_name] = categories_columns[col_name].str[-1]
        # Data type conversion: convert column values from string to 
        # numeric type, float
        categories_columns[col_name] = categories_columns[col_name].astype(float)

    # Concatenate the new categories columns, with the rest of the 
    # non-categories data frame.
    transformed_df = pd.concat([non_categories_columns, categories_columns], axis=1)
    print("\nTransformed Dataset (First Five Rows): \n", transformed_df.head(5))
    print(f"\nTransformed Dataset Data Frame dimensions: row :"
          f" {transformed_df.shape[0]}, col : {transformed_df.shape[1]}")

    print(f"\nFinding duplicate rows, number of duplicate rows : "
          f"{transformed_df.duplicated().sum()}")
    
    # Remove Duplicates
    transformed_df.drop_duplicates(inplace=True)
    
    print(f"\nDropped duplicate rows, Data Frame dimensions: row :"
          f"{transformed_df.shape[0]}, col : {transformed_df.shape[1]}")
    print("\nData cleaning completed, Transformed Dataset (First Five Rows) :"
          "\n", transformed_df.head(5))
    print(f"Initial Combined Dataset Data Frame dimensions: rows : "
          f"{df.shape[0]}, col : {df.shape[1]}\n")
    print(f"\nCleaned Dataset Data Frame dimensions: rows : "
          f"{transformed_df.shape[0]}, col : {transformed_df.shape[1]}")
    
    return transformed_df


def save_data(data_frame, database_path):
    """
    save pandas dataframe table into an sqlite database.

    Args:
        df(pandas.DataFrame): Pandas Data Frame containing cleaned data
                                to be loaded into the database.
        database_path(str): relative path to where the sqllite database
                                should be saved. eg DisasterTweets.db
    """
    # Create an Sqlite database with a table for the Cleaned Data.
    engine = create_engine("sqlite:///"+database_path)
    data_frame.to_sql("cleandata", engine, index=False)


def main():
    if len(sys.argv) == 4:
        # fetch file paths for messages, categories and database from the 
        # command line arguments supplied when running process_data.py
        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)
        
        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')

# mssg_path = "disaster_messages.csv"
# categories_path = "disaster_categories.csv"
# db_path = "DisasterResponse.db"

if __name__ == '__main__':
    main()
    
# python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db