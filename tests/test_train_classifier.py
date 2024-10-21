import unittest
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from train_classifier import load_data, tokenize, StartingVerbExtractor, build_model

class TestTrainClassifier(unittest.TestCase):

    def test_load_data(self):
        """Test load_data function to ensure it correctly loads data."""
        X, y, classes = load_data('data/DisasterResponse.db', table_name='cleandata')
        self.assertIsInstance(X, np.ndarray)
        self.assertIsInstance(y, np.ndarray)
        self.assertIsInstance(classes, list)
        self.assertGreater(len(X), 0)
        self.assertGreater(len(y), 0)
        self.assertEqual(len(X), len(y))

    def test_tokenize(self):
        """Test tokenize function to ensure correct tokenization."""
        text = "This is a test message with URL http://example.com."
        tokens = tokenize(text)
        expected_tokens = ['test', 'message', 'url']
        self.assertEqual(tokens, expected_tokens)
    
    def test_starting_verb_extractor(self):
        """Test StartingVerbExtractor to ensure it identifies verbs correctly."""
        extractor = StartingVerbExtractor()
        sample_text = "Run towards the goal."
        result = extractor.starting_verb(sample_text)
        self.assertEqual(result, 1)  # First word is a verb
        
        sample_text_2 = "The goal is near."
        result_2 = extractor.starting_verb(sample_text_2)
        self.assertEqual(result_2, 0)  # First word is not a verb

    def test_build_model(self):
        """Test build_model to ensure it creates a valid pipeline."""
        model = build_model()
        self.assertIsInstance(model, Pipeline)
        self.assertTrue(hasattr(model, 'fit'))
        self.assertTrue(hasattr(model, 'predict'))

    def test_model_pipeline(self):
        """Test the end-to-end pipeline to ensure the pipeline runs and predicts."""
        X = np.array(["Run towards the goal.", "This is a test message."])
        y = np.array([[1, 0], [0, 1]])
        model = build_model()
        model.fit(X, y)
        predictions = model.predict(X)
        self.assertEqual(predictions.shape, y.shape)

if __name__ == '__main__':
    unittest.main()