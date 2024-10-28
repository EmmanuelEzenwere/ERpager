import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re

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