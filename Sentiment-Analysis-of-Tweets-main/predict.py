import os
import sys
import pickle
import numpy as np

# Set UTF-8 encoding for stdout to prevent Windows console encoding errors
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Import preprocessing
from preprocessing.clean_text import preprocess_text

class SentimentPredictor:
    def __init__(self):
        self.workspace_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_path = os.path.join(self.workspace_dir, "models", "sentiment_model.pkl")
        self.vectorizer_path = os.path.join(self.workspace_dir, "models", "vectorizer.pkl")
        
        self.model = None
        self.vectorizer = None
        self.load_model()

    def load_model(self):
        """
        Loads the saved classification model and TF-IDF vectorizer.
        """
        if not os.path.exists(self.model_path) or not os.path.exists(self.vectorizer_path):
            print(f"Warning: Model files not found. Please train the model first by running train_model.py")
            return False
        
        try:
            with open(self.model_path, "rb") as f:
                self.model = pickle.load(f)
            with open(self.vectorizer_path, "rb") as f:
                self.vectorizer = pickle.load(f)
            return True
        except Exception as e:
            print(f"Error loading model files: {e}")
            return False

    def predict(self, tweet_text):
        """
        Cleans and predicts the sentiment of a single tweet.
        
        Returns:
        - dict: Prediction details containing:
            - 'sentiment': 'Positive', 'Neutral', or 'Negative'
            - 'confidence': confidence percentage (0-100)
            - 'clean_text': preprocessed text
            - 'probabilities': dictionary of sentiment probabilities
        """
        if self.model is None or self.vectorizer is None:
            # Try reloading if not loaded
            if not self.load_model():
                return {
                    "sentiment": "Neutral",
                    "confidence": 0.0,
                    "clean_text": "",
                    "probabilities": {"Positive": 0.33, "Neutral": 0.34, "Negative": 0.33},
                    "error": "Model files are not loaded or trained."
                }

        # 1. Preprocess the input text
        clean_tweet = preprocess_text(tweet_text, method='lemmatize')
        
        # Fallback if tweet is empty after cleaning
        if not clean_tweet.strip():
            return {
                "sentiment": "Neutral",
                "confidence": 100.0,
                "clean_text": "",
                "probabilities": {"Positive": 0.0, "Neutral": 1.0, "Negative": 0.0}
            }

        # 2. Vectorize the cleaned text
        vector = self.vectorizer.transform([clean_tweet])
        
        # 3. Predict sentiment
        sentiment = self.model.predict(vector)[0]
        
        # 4. Get confidence probabilities
        try:
            probabilities = self.model.predict_proba(vector)[0]
            classes = self.model.classes_
            
            # Map classes to their probabilities
            probs_dict = {classes[i]: float(probabilities[i]) for i in range(len(classes))}
            
            # Get the confidence score for the predicted sentiment
            confidence = float(probs_dict[sentiment]) * 100
        except Exception:
            # Fallback for models without predict_proba (like some SVC configurations)
            # SVC kernel='linear' with probability=True supports predict_proba, but we add a safety check
            confidence = 100.0
            probs_dict = {
                "Positive": 1.0 if sentiment == "Positive" else 0.0,
                "Neutral": 1.0 if sentiment == "Neutral" else 0.0,
                "Negative": 1.0 if sentiment == "Negative" else 0.0
            }

        return {
            "sentiment": sentiment,
            "confidence": round(confidence, 2),
            "clean_text": clean_tweet,
            "probabilities": {k: round(v, 4) for k, v in probs_dict.items()}
        }

if __name__ == "__main__":
    predictor = SentimentPredictor()
    
    # Quick sanity check
    test_inputs = [
        "I absolutely love this new device! It is amazing! 😊",
        "It was a boring and average day at work. Just did some filing.",
        "This software is terrible and slow! Completely waste of money.",
        "the package has arrived."
    ]
    
    print("Testing Predictor Utility:")
    for text in test_inputs:
        res = predictor.predict(text)
        print(f"\nInput: {text}")
        print(f"Cleaned: {res['clean_text']}")
        print(f"Result: {res['sentiment']} (Confidence: {res['confidence']}%)")
        print(f"Probs:  {res['probabilities']}")
