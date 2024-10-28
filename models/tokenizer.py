import re
import nltk
import pandas as pd

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.base import BaseEstimator
from sklearn.base import BaseEstimator, TransformerMixin

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)
nltk.download('punkt_tab', quiet=True)

def tokenize(text):
    """
    Tokenize a text by normalizing, lemmatizing and removing stop words.
    
    Args:
        text (list): list of strings
        stop_words (set): a set of word strings for stop words -- optional.

    Returns:
        tokens(list): list of token strings.
    """
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    
    url_regex = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    # Replace URLs with a placeholder and normalize case.
    normalized_text = re.sub(url_regex, ' ', text.lower())
    # Replace non-alphanumeric characters with spaces.
    normalized_text = re.sub(r'[^a-zA-Z0-9]', ' ', normalized_text)
    
    tokens = word_tokenize(normalized_text)
    
    # Lemmatize with POS tagging
    clean_tokens = []
    for token in tokens:
        if token not in stop_words:
            pos_tag = nltk.pos_tag([token])[0][1]
            if pos_tag.startswith('RB') and token.endswith('ly'):
                base_form = token[:-2] # Remove 'ly'
                clean_token = lemmatizer.lemmatize(base_form, pos='a')
            elif pos_tag.startswith('VB'):
                clean_token = lemmatizer.lemmatize(token, pos='v')
            elif pos_tag.startswith('JJ'):
                clean_token = lemmatizer.lemmatize(token, pos='a')
            else:
                clean_token = lemmatizer.lemmatize(token, pos='n')
            clean_tokens.append(clean_token)
    
    return clean_tokens


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