# import libraries
import sys
import re
import sklearn
import nltk
import pickle 

import numpy as np
import pandas as pd

from pprint import pprint
from sqlalchemy import create_engine

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer

# help(sklearn)
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GridSearchCV

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Download the stopwords and all nltk relevant packages.

# Download the Punkt tokenizer model
nltk.download('punkt')
# Download the stopwords
nltk.download('stopwords')
# Download the WordNet model
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('punkt_tab')

stop_words = set(stopwords.words("english"))

print(stop_words)

def load_data(
    x_column_name = "message", 
    y_start=4, 
    y_stop=None, 
    table_name='cleandata',
    database_path="DisasterTweets.db"
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
    
    return X, y, df


def tokenize(text, stop_words=None):
    """
    Tokenize a text by normalizing, lemmatizing and removing stop words.
    
    Args:
        text (list): list of strings
        stop_words (set): a set of word strings for stop words -- optional.

    Returns:
        tokens(list): list of token strings.
    """
    # Import stopwords if not imported.
    if stop_words is None:
        stop_words = set(stopwords.words("english"))
    
    lemmatizer = WordNetLemmatizer()    
    
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    
    # Replace URLs with a placeholder and normalize case.
    normalized_text = re.sub(url_regex, ' ', text.lower())

    # Replace non-alphanumeric characters with spaces.
    normalized_text = re.sub(r'[^a-zA-Z0-9]', ' ', normalized_text)
    
    tokens = word_tokenize(normalized_text)
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    return tokens

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
        try:
            
            # print("\n\nText:", text)
            sentence_list = nltk.sent_tokenize(text)
            for sentence in sentence_list:
                # Tokenize the sentence
                text_tokens = tokenize(sentence)
                
                if text_tokens:
                    # Get the POS (parts of speech) of the words in the text.
                    first_word, first_tag = nltk.pos_tag(text_tokens)[0]
                    
                    # Check if the first word is a Verb or 'RT' (retweet). 
                    if first_tag in ['VB', 'VBP', 'UH'] or first_word == 'RT':
                        # print("\nVerb, Tag: ",first_tag, ", First Word: ",first_word)
                        tag = 1
                        return tag
                    else:
                        # print(f"\nNon-verb, Tag: {first_tag}, First Word: ,{first_word}")
                        tag = 0
                        return tag
                else:
                    # print(f'Empty Text Tokens, {text_tokens} in "{sentence}"')
                    pass
            
            # If no sentences were found in the entire text.
            self.run_count += 1
            print(f"Non-Text first word({self.run_count}): ", sentence_list)
            print("Default Tag: 0\n")
            tag = 0
            return tag
        
        except IndexError as e:
            print(f"IndexError: {e}")
            print(f"Text causing issue: {text}")
            tag = 0
            return tag
    
        except Exception as e:
            print(f"Unexpected error: {e}")
            print(f"Text causing issue: {text}")
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
        print("Input feature shape: ", X.shape)
        
        return df_array
    
def build_model():
    """
    Constructs an ML pipeline using FeatureUnion to add a new binary feature.
    The feature checks if the first word in each text is a verb (1 if a verb, 0 otherwise),
    and combines this feature with the text data processed through a TF-IDF transformer pipeline.
    The combined feature matrix is used to train the model to enhance its performance.

    Returns:
        model (Pipeline): A scikit-learn pipeline object for training and evaluation, 
        which includes feature extraction, transformation, and a classifier.
    """
    
    text_pipeline = Pipeline([
        ("vect", CountVectorizer(tokenizer=tokenize)),
        ("tfidf", TfidfTransformer())
    ])
    
    features = FeatureUnion([
        ("text_pipeline",text_pipeline),
        ("starting_verb", StartingVerbExtractor())
    ])
    
    model = Pipeline([
        ("features", features),
        ("clf", MultiOutputClassifier(RandomForestClassifier()))
])
    
    # specify parameters for grid search
    parameters = {
        'features__text_pipeline__vect__ngram_range': ((1, 1), (1, 2)),
        'clf__estimator__n_estimators': [10, 25, 50],
        'clf__estimator__min_samples_split': [2, 3, 4]
    }
    
    # create grid search object
    cv = GridSearchCV(estimator=model, param_grid=parameters, verbose=2, cv=3, error_score='raise')
    
    return cv   
    
    # Predict using the pipeline.
def confusion_matrix_plot(y_test, y_pred, class_names):
    """
    Plots a confusion matrix of the actual class (output) and predicted class.

    Args:
        y_test (numpy.ndarray): actual output data.
        y_pred (numpy.ndarray): predicted output data as returned by our model,
                              classifier.
        class_names (list): list of all class names in the output dataset.
    """
    # Convert one-hot to labels
    y_test_labels = np.argmax(y_test, axis=1)
    y_pred_labels = np.argmax(y_pred, axis=1)

    # Create confusion matrix with explicit labels
    all_classes = np.arange(len(class_names))  # 0 to 35
    conf_matrix = confusion_matrix(y_test_labels, y_pred_labels, labels=all_classes)

    # Visualize
    plt.figure(figsize=(13, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.title('Confusion Matrix')
    plt.show()

    # Print class distribution
    print("-"*100)
    print("\nClass Distribution in Test Set:")
    print("-"*100)
    unique, counts = np.unique(y_test_labels, return_counts=True)

    for class_idx, count in zip(unique, counts):
        print(f'{class_names[class_idx]}: {count} samples')


def calculate_multiclass_accuracy(y_true, y_pred):
    """
    Calculate accuracy for multi-class classification
    
    Args:
        y_test (numpy.ndarray): actual output data.
        y_pred (numpy.ndarray): predicted output data as returned by our model,
                              classifier.
    
    Returns:
        summary_accuracy_score (float): summary accuracy using decoded labels.
    """
    # Convert to labels if one-hot encoded
    if (len(y_true.shape) > 1) & (len(y_pred.shape) > 1):
        y_true = np.argmax(y_true, axis=1)
        y_pred = np.argmax(y_pred, axis=1)
        summary_accuracy_score = accuracy_score(y_true, y_pred)
       
    else:
        print("Empty values for y test or y pred")
        summary_accuracy_score = np.nan
    
    return summary_accuracy_score


def per_class_accuracy(y_true, y_pred, class_names):
    """
    Calculate accuracy for each class separately

    Args
        y_test (numpy.ndarray): actual output data.
        y_pred (numpy.ndarray): predicted output data as returned by our model,
                              classifier.
        class_names (list): list of all class names in the output dataset.
    
    return
        accuracies_df (pandas.core.frame.DataFrame): a dataframe of class names 
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
        metrics_df(pandas.core.frame.DataFrame): a dataframe of class names 
                              and their correspoinding precision, recall and 
                              F1-score values, for binary classification
    """
    sample_accuracy = calculate_multiclass_accuracy(y_true, y_pred)
    print(f"Overall Sample-wise Accuracy: {sample_accuracy:.3f}\n")
    print("-"*100)

    
    # Calculate per-class accuracy
    print(per_class_accuracy(y_true, y_pred, class_names))
    print("-"*100)

    # *** Per-class classification metrics
    print('Detailed classification report per class')
    print("-"*100)

    
    results = []
    for i in range(y_true.shape[1]):
        if np.sum(y_true[:, i]) > 0:  # Check if there are true instances
            precision = precision_score(y_true[:, i], y_pred[:, i], average='macro', zero_division=0)
            recall = recall_score(y_true[:, i], y_pred[:, i],average='macro', zero_division=0)
            f1 = f1_score(y_true[:, i], y_pred[:, i],average='macro', zero_division=0)
        
            results.append({
                'Class': class_names[i],
                'Precision': precision,
                'Recall': recall,
                'F1-Score': f1
            })
        
            print(f"\nDetailed metrics for {class_names[i]}:")
            print(classification_report(y_true[:, i], y_pred[:, i], zero_division=0))
        else:
            results.append({
                'Class': class_names[i],
                'Precision': np.nan,
                'Recall': np.nan,
                'F1-Score': np.nan
            })
            print(f"\nNo true instances for {class_names[i]}, skipping detailed report.")
            
    # Create a DataFrame with all classification metrics per class
    print("-"*100)
    metrics_df = pd.DataFrame(results)
    print("\nSummary of all metrics:")
    print("-"*100)
    print(metrics_df.round(3))
    
    # Visualize metrics
    plt.figure(figsize=(10, 6))
    metrics_melted = pd.melt(metrics_df, id_vars=['Class'], 
                           value_vars=['Precision', 'Recall', 'F1-Score'])
    sns.barplot(x='Class', y='value', hue='variable', data=metrics_melted)
    plt.xticks(rotation=80)
    plt.title('Model Performance Metrics by Class')
    plt.tight_layout()
    plt.show()
    
    return metrics_df
        
        
def evaluate_model(model, x_test, y_test, class_names):
    """
    Evaluate the model and print all relevant metrics
    
    Args:
        x_test(numpy.ndarray): actual input data
        y_test(numpy.ndarray): actual output data.
        class_names (list): list of all class names in the output dataset.
    
    Returns:
        y_pred(numpy.ndarray): predicted output data as returned by our model,
                              classifier.
        metrics_df(pandas.core.frame.DataFrame): a dataframe of class names 
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
        print("Best parameters found:")
        pprint(model.best_params_)
        print("\nBest cross-validation score:", model.best_score_)
        print("-"*100)
    else:
        print('Non-Pipeline Model')
        print("-"*100)
    
    # Evaluate the model
    metrics_df = evaluate_multilabel_model(y_test, y_pred, class_names)
    
    confusion_matrix_plot(y_test, y_pred, class_names)
    print("\n")
  
    return y_pred, metrics_df


def save_model(model, filepath='train_classifier.pkl'):
    """
    Saves the ML model as a pickle file.

    Args:
        model (pipeline or GridSearchCV object): The ML model to be saved.
        filepath (str): Specified filename and path to save the ML model.
    """
    try:
        # Export the model using pickle
        with open(filepath, 'wb') as model_file:
            pickle.dump(model, model_file)

        print("Model exported successfully!")
    except Exception as e:
        print(f"Error saving model: {e}")


def main():
    """_summary_
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
        evaluate_model(model, X_test, Y_test, category_names)

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



from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
def display_results(y_test, y_pred):
    labels = np.unique(y_pred)
    confusion_mat = confusion_matrix(y_test, y_pred, labels=labels)
    accuracy = (y_pred == y_test).mean()
    print("Labels:", labels)
    print("Confusion Matrix:\n", confusion_mat)
    print("Accuracy:", accuracy)


# def load_data():
#     """ 
#     """
#     df = pd.read_csv('corporate_messaging.csv', encoding='latin-1')
#     df = df[(df["category:confidence"] == 1) & (df['category'] != 'Exclude')]
#     X = df.text.values
#     y = df.category.values
#     return X, y


# def tokenize(text):
#     """_summary_

#     Args:
#         text (_type_): _description_

#     Returns:
#         _type_: _description_
#     """
#     detected_urls = re.findall(url_regex, text)
#     for url in detected_urls:
#         text = text.replace(url, "urlplaceholder")

#     tokens = word_tokenize(text)
#     lemmatizer = WordNetLemmatizer()

#     clean_tokens = []
#     for tok in tokens:
#         clean_tok = lemmatizer.lemmatize(tok).lower().strip()
#         clean_tokens.append(clean_tok)

#     return clean_tokens


# def model_pipeline():
#     """
#     creates a pipleline which contains a feature union (text_verb_union) of 
#     a transformer and the output of another pipeline.
#     by appending a boolean of if the text is a verb or not to the last column
#     of the tfidf matrix of the text.


#     """
    
#     text_processing_pipeline = Pipeline([
#                 ('vect', CountVectorizer(tokenizer=tokenize)),
#                 ('tfidf', TfidfTransformer())
#             ])
    
    
#     text_verb_union = FeatureUnion([

#             ('text_pipeline', text_processing_pipeline),
#             ('starting_verb', StartingVerbExtractor())
#         ])
    
    
#     pipeline = Pipeline([
#         ('features', text_verb_union),
#         ('clf', RandomForestClassifier())
#     ])

#     return pipeline


# def display_results(y_test, y_pred):
#     """_summary_

#     Args:
#         y_test (_type_): _description_
#         y_pred (_type_): _description_
#     """
#     labels = np.unique(y_pred)
#     confusion_mat = confusion_matrix(y_test, y_pred, labels=labels)
#     accuracy = (y_pred == y_test).mean()

#     print("Labels:", labels)
#     print("Confusion Matrix:\n", confusion_mat)
#     print("Accuracy:", accuracy)


# def main():
#     """_summary_
#     """
#     X, y = load_data()
#     X_train, X_test, y_train, y_test = train_test_split(X, y)

#     model = model_pipeline()
#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_test)

#     display_results(y_test, y_pred)

# main()