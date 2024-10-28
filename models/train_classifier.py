# import libraries
import sys
import sklearn
import nltk
import re
import pickle 
import dill
import os

import numpy as np
import pandas as pd
from pprint import pprint
from sqlalchemy import create_engine
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.base import BaseEstimator
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import FeatureUnion,Pipeline

# Add the project root directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.tokenizer import tokenize  

def load_data(
    database_path,
    x_column_name = "message", 
    y_start=4, 
    y_stop=None, 
    table_name='cleandata'
    ):
    """
    Fetch cleaned data from an sqlite database and return as 
    pandas data frame.

    Args:
       x_column_name (str): Name of the column in the dataset table
                             for input features (default is 'message').
        y_start (int): The starting index for the target columns (default is 4).
        y_stop (int or None): The stopping index for the target columns.
                              If None, all columns from y_start onward will be 
                              selected (default is None).
        table_name (str): Name of the table in the database containing the dataset
                          (default is 'cleandata').
        database_path (str): Relative path to the database file containing the 
                             cleaned dataset (default is 'DisasterTweets.db').
    
    Returns:
        X (numpy.ndarray): Input feature data.
        y (numpy.ndarray): Target output data.
        df (pandas.DataFrame): DataFrame containing both input features and target outputs.
    """
    conn_engine = create_engine("sqlite:///"+database_path)
    # Load data from a specific table into a DataFrame
    df = pd.read_sql_table(table_name, con=conn_engine)
    # Extract the input feature column
    X = df[x_column_name]
    # Extract the target columns
    if y_stop == None:
        y = df.iloc[:,y_start:]
    else:
        y = df.iloc[:, y_start:y_stop] 
    classes = list(y.columns)
    return X.values, y.values, classes


class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        """
        Generates a new binary input feature, (1 if the first word in the input text is a verb, 0 otherwise),
        This feature will be combined with the input text data processed through a TF-IDF transformer pipeline.
        """
        self.run_count = 1
        print("Start Verb Extractor running...")

    def starting_verb(self, text):
        """
        Tags the first word of the tokenized input text.
        
        Args:
            text(string): a row input text message.
        returns:
             tag(integer): 1 if the first word is a verb, 0 otherwise.
        """
        self.run_count += 1
        try:
            # Check for RT (retweet) first
            if text.strip().upper().startswith('RT'):
                return 1
        
            # print("\n\nText:", text)
            first_sentence = nltk.sent_tokenize(text)[0]
            
            # Tokenize the first_sentence
            text_tokens = tokenize(first_sentence) 
            
            if not text_tokens:
                # No text token in first sentence or first sentence in text is empty
        
                # print(f'Empty Text Tokens, {text_tokens} in first "{first_sentence}"')
                # print(f"Non-Text first word({self.run_count}): ")
                tag = 0
                return tag
            
            # Get the POS (parts of speech) of the words in the text.
            first_word, first_tag = nltk.pos_tag(text_tokens)[0]
            
            # Check if the first word is a Verb. 
            if first_tag in ['VB', 'VBP', 'UH']:
                # print("\nVerb, Tag: ",first_tag, ", First Word: ",first_word)
                tag = 1
                return tag
            else:
                # print(f"\nNon-verb, Tag: {first_tag}, First Word: ,{first_word}")
                tag = 0
                return tag
                
        except Exception as e:
            # print(f"Unexpected error: {e}")
            # print(f"Run No.({self.run_count})")
            # print(f"Text causing issue: {text}")
            tag = 0
            return tag
    
    def fit(self, x, y=None):
        return self

    def transform(self, X):
        """
        Transform method.
        
        Args:
            X(numpy.ndarray):input column vector of text messages. Each row is a 
                             sample text message.
        returns:
            df_array(numpy.ndarray):a new column binary vector. Each row is a 
                             1 or 0 (1 if the first word in the corresponding 
                             row in the input column vector is a verb, and 0,
                             otherwise).
        """
        X_tagged = pd.Series(X).apply(self.starting_verb)
        df_array = pd.DataFrame(X_tagged).values # converts to a 2D numpy array.
        # df = X_tagged.values # removing this because the hstack fails.
        
        # Log information about the transformation
        print("\n\nFeature Extraction and Text Transformation Complete:")
        print("Extracted/New feature shape:", df_array.shape)
        if type(X) == list:
            print("Input feature shape: ", len(X))
        else:
            print("Input feature shape: ", X.shape)
        
        return df_array
 
 
def calculate_multiclass_accuracy(y_true, y_pred):
    """
    Calculate accuracy for multi-class classification
    
    Args:
        y_test(numpy.ndarray): actual output data.
        y_pred(numpy.ndarray): predicted output data as returned by our model,
                              classifier.
    
    Returns:
        summary_accuracy_score(float): summary accuracy using decoded labels.
    """
    # Convert to labels if one-hot encoded
    if (len(y_true.shape) > 1) & (len(y_pred.shape) > 1):
        # Initialize an empty list to store per-column accuracy
        accuracies = []

        # Iterate over each column (class output)
        for col in range(y_true.shape[1]):
            # Compute accuracy for the current column
            col_accuracy = accuracy_score(y_true[:, col], y_pred[:, col])
            accuracies.append(col_accuracy)

        # Average accuracy across all columns
        summary_accuracy_score = np.mean(accuracies)
       
    else:
        print("Empty values for y test or y pred")
        summary_accuracy_score = np.nan
    
    return summary_accuracy_score


def per_class_accuracy(y_true, y_pred, class_names):
    """
    Calculate accuracy for each class separately

    Args
        y_test(numpy.ndarray): actual output data.
        y_pred(numpy.ndarray): predicted output data as returned by our model,
                              classifier.
        class_names (list): list of all class names in the output dataset.
    
    return
        accuracies_df (pd.DataFrame): a dataframe of class names 
        and their correspoinding accuracy score for binary classification
    
    """
    accuracies = []
    for i in range(len(class_names)):
        class_data_true = y_true[:, i]
        class_data_pred = y_pred[:, i]
        if (class_data_true.shape[0] > 0) & (class_data_pred.shape[0] > 0):
            class_acc = accuracy_score(class_data_true, class_data_pred)
            accuracies.append(class_acc)
            
        else:
            print("Empty values for y test or y pred")
            class_acc = np.nan
            accuracies.append(class_acc)
            
    
    accuracies_df = pd.DataFrame({"category":class_names, "accuracy":accuracies})
    return accuracies_df
    
    
def evaluate_multilabel_model(y_true, y_pred, class_names):
    """
    Comprehensive evaluation of a multi-label classification model.
    
    Args:
        y_test (numpy.ndarray): actual output data.
        y_pred (numpy.ndarray): predicted output data as returned by our model,
                              classifier.
        class_names (list): list of all class names in the output dataset.
        
    return
        metrics_df(pd.DataFrame): a dataframe of class names 
                              and their correspoinding precision, recall and 
                              F1-score values, for binary classification
    """
    average_accuracy = calculate_multiclass_accuracy(y_true, y_pred)
    print(f"Overall Sample-wise Accuracy: {average_accuracy:.3f}")
    print("-"*100)

    # Calculate per-class accuracy
    accuracies_df = per_class_accuracy(y_true, y_pred, class_names)

    # Per-class classification metrics
    classification_metrics = []
    reports = {}
    for i in range(y_true.shape[1]):
        if np.sum(y_true[:, i]) > 0:  # Check if there are true instances
            precision = precision_score(y_true[:, i], y_pred[:, i], average='macro', zero_division=0)
            recall = recall_score(y_true[:, i], y_pred[:, i],average='macro', zero_division=0)
            f1 = f1_score(y_true[:, i], y_pred[:, i],average='macro', zero_division=0)
        
            classification_metrics.append({
                'Class': class_names[i],
                'Precision': precision,
                'Recall': recall,
                'F1-Score': f1
            })
        
            
            report_dict = classification_report(y_true[:, i], y_pred[:, i], zero_division=0, output_dict=True)
            report_df = pd.DataFrame(report_dict).transpose()
            report_df = report_df.round(3)
            report_log = None
            reports[class_names[i]]=[report_df, report_log]
            
        else:
            classification_metrics.append({
                'Class': class_names[i],
                'Precision': np.nan,
                'Recall': np.nan,
                'F1-Score': np.nan
            })
            report_df = None
            reports_log = f"\nFound No true instances for {class_names[i]}. Ommitted from report\n"
            reports[class_names[i]]=[report_df, reports_log]
            
    # Create a DataFrame with a summary classification metrics per class.
    classification_metrics_df = pd.DataFrame(classification_metrics)
    print("\nSummary of all metrics:")
    print("-"*100)
    metrics_df = pd.concat([accuracies_df, classification_metrics_df.iloc[:,1:]], axis=1).round(3)
    
    if 'ipykernel' in sys.modules:
        # Running in a Jupyter Notebook.
        display(metrics_df.round(3))

    else:
        # Not running in a Jupyter Notebook.
        print(metrics_df.round(3))
        
    
    print('Detailed classification report per class')
    print("-"*100)
    for class_name in reports.keys():
        print(f"\nDetailed metrics for {class_name}:")
        report_df = reports[class_name][0]
        if report_df is not None:
            if 'ipykernel' in sys.modules:
                # Running in a Jupyter Notebook.
                display(report_df)

            else:
                # Not running in a Jupyter Notebook.
                print(report_df)
        else:
            report_log = reports[class_name][1]
            print(report_log)
        
    # Visualize metrics
    plt.figure(figsize=(10, 6))
    metrics_melted = pd.melt(classification_metrics_df, id_vars=['Class'], 
                           value_vars=['Precision', 'Recall', 'F1-Score'])
    sns.barplot(x='Class', y='value', hue='variable', data=metrics_melted)
    plt.xticks(rotation=80)
    plt.title('Model Performance Metrics by Class')
    plt.tight_layout()
    plt.show()
    
    return metrics_df
             
              
def model_evaluator(model, x_test, y_test, class_names):
    """
    Evaluate the model and print all relevant metrics
    
    Args:
        x_test(numpy.ndarray): actual input data
        y_test(numpy.ndarray): actual output data.
        class_names (list): list of all class names in the output dataset.
    
    Returns:
        y_pred(numpy.ndarray): predicted output data as returned by our model,
                              classifier.
        metrics_df(pd.DataFrame): a dataframe of class names 
                              and their correspoinding precision, recall and 
                              F1-score values, for binary classification
        
    """
    print("-"*100)
    print("MODEL PERFORMANCE EVALUATION")
    print("-"*100)
    
    # Get predictions
    y_pred = model.predict(x_test)
    
    # Get best parameters if using GridSearchCV
    if hasattr(model, 'best_params_'):
        print('ML-Pipeline Model')
        print("-"*100)
        print("\nBest cross-validation score:", model.best_score_)
        print("\nBest parameters found:")
        pprint(model.best_params_)
        print("-"*100)
    else:
        print('Non-ML Pipeline Model')
        print("-"*100)
    
    # Evaluate the model
    metrics_df = evaluate_multilabel_model(y_test, y_pred, class_names)
  
    return y_pred, metrics_df


# def build_model():
#     """
#     Constructs an ML pipeline using FeatureUnion to add a new binary feature.
#     The feature checks if the first word in each text is a verb (1 if a verb, 0 otherwise),
#     and combines this feature with the text data processed through a TF-IDF transformer pipeline.
#     The combined feature matrix is used to train the model to enhance its performance.

#     Returns:
#         model (Pipeline): A scikit-learn pipeline object for training and evaluation, 
#         which includes feature extraction, transformation, and a classifier.
#     """
    
#     text_pipeline = Pipeline([
#         ("vect", CountVectorizer(tokenizer=tokenize)),
#         ("tfidf", TfidfTransformer())
#     ])
    
#     features = FeatureUnion([
#         ("text_pipeline",text_pipeline),
#         ("starting_verb", StartingVerbExtractor())
#     ])
    
#     model = Pipeline([
#         ("features", features),
#         ("clf", MultiOutputClassifier(RandomForestClassifier()))
#     ])
    
#     # specify parameters for grid search
#     parameters = {
#         'features__text_pipeline__vect__ngram_range': [(1, 2)],
#         'clf__estimator__n_estimators': [200],
#         'clf__estimator__min_samples_split': [2]
#     }
    
#     # create grid search object
#     cv = GridSearchCV(estimator=model, param_grid=parameters, verbose=2, cv=3, error_score='raise')
    
#     return cv


def build_model():
    """ 
    """
    model = Pipeline([
        ("vectorize", CountVectorizer(tokenizer=tokenize)),
        ("tfidf", TfidfTransformer()),
        ("clf", RandomForestClassifier())
    ])
    return model


def save_model(model, filepath, overwrite=True):
    """Save ML model using dill"""
    try:
        if not overwrite and os.path.exists(filepath):
            raise FileExistsError(f"File {filepath} already exists and overwrite=False")
            
        with open(filepath, 'wb') as f:
            dill.dump(model, f)
        print(f"Model saved successfully as {filepath}")
        
    except Exception as e:
        print(f"Error saving model: {e}")
        
        
def load_model(filepath):
    """Load ML model using dill"""
    try:
        # Configure dill for compatibility
        dill._dill._reverse_typemap['ObjectType'] = object
        
        with open(filepath, 'rb') as f:
            model = dill.load(f)
        print(f"Model loaded successfully from {filepath}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
    
    
def main():
    """
    ML model run script.
    
    """
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        model_evaluator(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
    
# python train_classifier.py ../data/DisasterResponse.db classifier.pkl