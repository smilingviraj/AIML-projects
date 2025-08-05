import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
from collections import Counter
import pickle
import json
from utils.text_preprocessing import TextPreprocessor, TFIDFVectorizer

class SimpleNaiveBayes:
    """
    Simple Naive Bayes classifier implementation for sentiment analysis
    """
    
    def __init__(self):
        self.class_priors = {}
        self.feature_probs = {}
        self.classes = []
        self.vocabulary = []
    
    def fit(self, X, y):
        """
        Train the Naive Bayes classifier
        """
        self.classes = list(set(y))
        n_samples = len(y)
        
        # Calculate class priors
        class_counts = Counter(y)
        for class_label in self.classes:
            self.class_priors[class_label] = class_counts[class_label] / n_samples
        
        # Initialize feature probabilities
        n_features = len(X[0]) if X else 0
        self.vocabulary = list(range(n_features))
        
        for class_label in self.classes:
            self.feature_probs[class_label] = []
            
            # Get samples for this class
            class_samples = [X[i] for i in range(len(X)) if y[i] == class_label]
            
            if class_samples:
                # Calculate feature probabilities for each feature
                for feature_idx in range(n_features):
                    feature_values = [sample[feature_idx] for sample in class_samples]
                    # Use Laplace smoothing
                    prob = (sum(feature_values) + 1) / (len(feature_values) + 2)
                    self.feature_probs[class_label].append(prob)
            else:
                # If no samples for this class, use uniform probability
                self.feature_probs[class_label] = [0.5] * n_features
    
    def predict_single(self, x):
        """
        Predict class for a single sample
        """
        class_scores = {}
        
        for class_label in self.classes:
            # Start with log prior
            score = np.log(self.class_priors[class_label])
            
            # Add log likelihoods
            for feature_idx, feature_value in enumerate(x):
                if feature_idx < len(self.feature_probs[class_label]):
                    prob = self.feature_probs[class_label][feature_idx]
                    # Handle binary features (TF-IDF can be treated as feature presence)
                    if feature_value > 0:
                        score += np.log(prob)
                    else:
                        score += np.log(1 - prob)
            
            class_scores[class_label] = score
        
        # Return class with highest score
        return max(class_scores, key=class_scores.get)
    
    def predict(self, X):
        """
        Predict classes for multiple samples
        """
        return [self.predict_single(x) for x in X]
    
    def predict_proba(self, X):
        """
        Predict class probabilities
        """
        probabilities = []
        
        for x in X:
            class_scores = {}
            
            for class_label in self.classes:
                score = np.log(self.class_priors[class_label])
                
                for feature_idx, feature_value in enumerate(x):
                    if feature_idx < len(self.feature_probs[class_label]):
                        prob = self.feature_probs[class_label][feature_idx]
                        if feature_value > 0:
                            score += np.log(prob)
                        else:
                            score += np.log(1 - prob)
                
                class_scores[class_label] = score
            
            # Convert to probabilities using softmax
            exp_scores = {k: np.exp(v) for k, v in class_scores.items()}
            total = sum(exp_scores.values())
            probs = {k: v / total for k, v in exp_scores.items()}
            
            probabilities.append(probs)
        
        return probabilities

class SentimentAnalyzer:
    """
    Complete sentiment analysis pipeline
    """
    
    def __init__(self, max_features=500):
        self.preprocessor = TextPreprocessor()
        self.vectorizer = TFIDFVectorizer(max_features=max_features, min_df=2)
        self.classifier = SimpleNaiveBayes()
        self.is_trained = False
        
    def prepare_features(self, texts):
        """
        Prepare features combining TF-IDF and custom features
        """
        # Get TF-IDF features
        tfidf_features = self.vectorizer.transform(texts)
        
        # Get custom features
        custom_features = []
        for text in texts:
            features = self.preprocessor.extract_features(text)
            feature_vector = [
                features['word_count'] / 100,  # Normalize
                features['char_count'] / 1000,  # Normalize
                features['avg_word_length'] / 10,  # Normalize
                features['exclamation_count'],
                features['question_count'],
                features['uppercase_count'] / 100,  # Normalize
                features['positive_keywords'],
                features['negative_keywords'],
                features['neutral_keywords']
            ]
            custom_features.append(feature_vector)
        
        # Combine features
        combined_features = []
        for i in range(len(texts)):
            combined = tfidf_features[i] + custom_features[i]
            combined_features.append(combined)
        
        return combined_features
    
    def train(self, texts, labels):
        """
        Train the sentiment analyzer
        """
        print(f"Training sentiment analyzer on {len(texts)} samples...")
        
        # Fit vectorizer
        self.vectorizer.fit(texts)
        
        # Prepare features
        X = self.prepare_features(texts)
        
        # Train classifier
        self.classifier.fit(X, labels)
        self.is_trained = True
        
        print("Training completed!")
        
        # Print training statistics
        label_counts = Counter(labels)
        print(f"Training data distribution:")
        for label, count in label_counts.items():
            print(f"  {label}: {count} ({count/len(labels)*100:.1f}%)")
    
    def predict(self, texts):
        """
        Predict sentiment for texts
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        X = self.prepare_features(texts)
        return self.classifier.predict(X)
    
    def predict_proba(self, texts):
        """
        Predict sentiment probabilities
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        X = self.prepare_features(texts)
        return self.classifier.predict_proba(X)
    
    def evaluate(self, texts, true_labels):
        """
        Evaluate the model performance
        """
        predictions = self.predict(texts)
        
        # Calculate accuracy
        correct = sum(1 for pred, true in zip(predictions, true_labels) if pred == true)
        accuracy = correct / len(true_labels)
        
        # Calculate per-class metrics
        classes = list(set(true_labels))
        metrics = {}
        
        for class_label in classes:
            tp = sum(1 for pred, true in zip(predictions, true_labels) 
                    if pred == class_label and true == class_label)
            fp = sum(1 for pred, true in zip(predictions, true_labels) 
                    if pred == class_label and true != class_label)
            fn = sum(1 for pred, true in zip(predictions, true_labels) 
                    if pred != class_label and true == class_label)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            metrics[class_label] = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            }
        
        return {
            'accuracy': accuracy,
            'class_metrics': metrics,
            'predictions': predictions
        }
    
    def save_model(self, filepath):
        """
        Save the trained model
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            'vectorizer': {
                'vocabulary': self.vectorizer.vocabulary,
                'idf_scores': self.vectorizer.idf_scores,
                'max_features': self.vectorizer.max_features,
                'min_df': self.vectorizer.min_df
            },
            'classifier': {
                'class_priors': self.classifier.class_priors,
                'feature_probs': self.classifier.feature_probs,
                'classes': self.classifier.classes,
                'vocabulary': self.classifier.vocabulary
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(model_data, f, indent=2)
        
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """
        Load a trained model
        """
        with open(filepath, 'r') as f:
            model_data = json.load(f)
        
        # Restore vectorizer
        self.vectorizer.vocabulary = model_data['vectorizer']['vocabulary']
        self.vectorizer.idf_scores = model_data['vectorizer']['idf_scores']
        self.vectorizer.max_features = model_data['vectorizer']['max_features']
        self.vectorizer.min_df = model_data['vectorizer']['min_df']
        
        # Restore classifier
        self.classifier.class_priors = model_data['classifier']['class_priors']
        self.classifier.feature_probs = model_data['classifier']['feature_probs']
        self.classifier.classes = model_data['classifier']['classes']
        self.classifier.vocabulary = model_data['classifier']['vocabulary']
        
        self.is_trained = True
        print(f"Model loaded from {filepath}")

def train_sentiment_model():
    """
    Train the sentiment analysis model on NASA comments data
    """
    # Load data
    print("Loading NASA Facebook comments data...")
    df = pd.read_csv('../data/nasa_facebook_comments.csv')
    
    # Prepare data
    texts = df['Text'].tolist()
    labels = df['Sentiment'].tolist()
    
    print(f"Loaded {len(texts)} comments")
    
    # Split data (simple train/test split)
    split_idx = int(0.8 * len(texts))
    
    train_texts = texts[:split_idx]
    train_labels = labels[:split_idx]
    test_texts = texts[split_idx:]
    test_labels = labels[split_idx:]
    
    print(f"Training set: {len(train_texts)} samples")
    print(f"Test set: {len(test_texts)} samples")
    
    # Initialize and train model
    analyzer = SentimentAnalyzer(max_features=300)
    analyzer.train(train_texts, train_labels)
    
    # Evaluate model
    print("\nEvaluating model...")
    results = analyzer.evaluate(test_texts, test_labels)
    
    print(f"\nModel Performance:")
    print(f"Accuracy: {results['accuracy']:.3f}")
    print(f"\nPer-class metrics:")
    for class_label, metrics in results['class_metrics'].items():
        print(f"  {class_label}:")
        print(f"    Precision: {metrics['precision']:.3f}")
        print(f"    Recall: {metrics['recall']:.3f}")
        print(f"    F1-Score: {metrics['f1_score']:.3f}")
    
    # Save model
    analyzer.save_model('../models/sentiment_model.json')
    
    # Test on some examples
    print(f"\nTesting on sample texts:")
    sample_texts = [
        "NASA's climate research is absolutely fantastic and crucial for our future!",
        "This climate data is just propaganda and fear mongering.",
        "Interesting analysis. Would like to understand the methodology better."
    ]
    
    predictions = analyzer.predict(sample_texts)
    probabilities = analyzer.predict_proba(sample_texts)
    
    for i, text in enumerate(sample_texts):
        print(f"\nText: {text}")
        print(f"Predicted sentiment: {predictions[i]}")
        print(f"Probabilities: {probabilities[i]}")
    
    return analyzer, results

if __name__ == "__main__":
    # Train the model
    analyzer, results = train_sentiment_model()
    print("\nSentiment analysis model training completed!")