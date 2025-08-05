import re
import string
from collections import Counter

class TextPreprocessor:
    """
    Text preprocessing utilities for sentiment analysis
    """
    
    def __init__(self):
        # Common English stop words
        self.stop_words = {
            'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
            'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
            'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
            'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
            'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
            'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
            'while', 'of', 'at', 'by', 'for', 'with', 'through', 'during', 'before', 'after',
            'above', 'below', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
            'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all',
            'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
            'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will',
            'just', 'don', 'should', 'now'
        }
        
        # Climate-specific keywords for feature extraction
        self.climate_keywords = {
            'positive': ['research', 'data', 'science', 'important', 'crucial', 'excellent', 
                        'amazing', 'great', 'fantastic', 'thank', 'helpful', 'vital',
                        'transparency', 'accessible', 'future', 'understanding'],
            'negative': ['hoax', 'propaganda', 'agenda', 'fake', 'manipulation', 'alarmism',
                        'waste', 'politics', 'fear', 'mongering', 'scam', 'lie'],
            'neutral': ['interesting', 'question', 'explain', 'methodology', 'analysis',
                       'implications', 'patterns', 'accurate', 'data', 'factors']
        }
    
    def clean_text(self, text):
        """
        Clean and normalize text
        """
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove user mentions and hashtags
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove numbers
        text = re.sub(r'\d+', '', text)
        
        return text.strip()
    
    def tokenize(self, text):
        """
        Tokenize text into words
        """
        cleaned_text = self.clean_text(text)
        tokens = cleaned_text.split()
        
        # Remove stop words
        tokens = [token for token in tokens if token not in self.stop_words]
        
        # Remove very short tokens
        tokens = [token for token in tokens if len(token) > 2]
        
        return tokens
    
    def extract_features(self, text):
        """
        Extract features from text for sentiment analysis
        """
        tokens = self.tokenize(text)
        
        features = {
            'word_count': len(tokens),
            'char_count': len(text),
            'avg_word_length': sum(len(word) for word in tokens) / len(tokens) if tokens else 0,
            'exclamation_count': text.count('!'),
            'question_count': text.count('?'),
            'uppercase_count': sum(1 for c in text if c.isupper()),
            'positive_keywords': 0,
            'negative_keywords': 0,
            'neutral_keywords': 0
        }
        
        # Count climate-specific keywords
        text_lower = text.lower()
        for keyword in self.climate_keywords['positive']:
            features['positive_keywords'] += text_lower.count(keyword)
        for keyword in self.climate_keywords['negative']:
            features['negative_keywords'] += text_lower.count(keyword)
        for keyword in self.climate_keywords['neutral']:
            features['neutral_keywords'] += text_lower.count(keyword)
        
        return features
    
    def get_word_frequency(self, texts, top_n=20):
        """
        Get most frequent words across all texts
        """
        all_tokens = []
        for text in texts:
            all_tokens.extend(self.tokenize(text))
        
        word_freq = Counter(all_tokens)
        return word_freq.most_common(top_n)
    
    def sentiment_score_simple(self, text):
        """
        Simple rule-based sentiment scoring
        """
        features = self.extract_features(text)
        
        # Simple scoring based on keyword counts
        positive_score = features['positive_keywords']
        negative_score = features['negative_keywords']
        
        if positive_score > negative_score:
            return 'positive'
        elif negative_score > positive_score:
            return 'negative'
        else:
            return 'neutral'

class TFIDFVectorizer:
    """
    Simple TF-IDF implementation for text vectorization
    """
    
    def __init__(self, max_features=1000, min_df=2):
        self.max_features = max_features
        self.min_df = min_df
        self.vocabulary = {}
        self.idf_scores = {}
        self.preprocessor = TextPreprocessor()
    
    def fit(self, documents):
        """
        Fit the vectorizer on documents
        """
        # Tokenize all documents
        tokenized_docs = [self.preprocessor.tokenize(doc) for doc in documents]
        
        # Count document frequency for each word
        word_doc_count = Counter()
        for tokens in tokenized_docs:
            unique_tokens = set(tokens)
            for token in unique_tokens:
                word_doc_count[token] += 1
        
        # Filter words by minimum document frequency
        filtered_words = {word: count for word, count in word_doc_count.items() 
                         if count >= self.min_df}
        
        # Select top features
        top_words = sorted(filtered_words.items(), key=lambda x: x[1], reverse=True)
        top_words = top_words[:self.max_features]
        
        # Create vocabulary
        self.vocabulary = {word: idx for idx, (word, _) in enumerate(top_words)}
        
        # Calculate IDF scores
        total_docs = len(documents)
        for word in self.vocabulary:
            df = word_doc_count[word]
            self.idf_scores[word] = np.log(total_docs / df)
    
    def transform(self, documents):
        """
        Transform documents to TF-IDF matrix
        """
        if not self.vocabulary:
            raise ValueError("Vectorizer must be fitted before transform")
        
        matrix = []
        
        for doc in documents:
            tokens = self.preprocessor.tokenize(doc)
            
            # Calculate TF scores
            tf_scores = Counter(tokens)
            total_tokens = len(tokens)
            
            # Create feature vector
            vector = [0.0] * len(self.vocabulary)
            
            for word, tf in tf_scores.items():
                if word in self.vocabulary:
                    idx = self.vocabulary[word]
                    tf_normalized = tf / total_tokens
                    tfidf_score = tf_normalized * self.idf_scores[word]
                    vector[idx] = tfidf_score
            
            matrix.append(vector)
        
        return matrix
    
    def fit_transform(self, documents):
        """
        Fit and transform in one step
        """
        self.fit(documents)
        return self.transform(documents)

# Import numpy for TF-IDF calculations
import numpy as np

if __name__ == "__main__":
    # Test the preprocessing
    preprocessor = TextPreprocessor()
    
    sample_texts = [
        "Thank you NASA for your incredible work on climate research!",
        "This climate change stuff is just fear mongering.",
        "Interesting climate data. Would like to see more analysis."
    ]
    
    print("Testing Text Preprocessing:")
    for i, text in enumerate(sample_texts):
        print(f"\nText {i+1}: {text}")
        print(f"Cleaned: {preprocessor.clean_text(text)}")
        print(f"Tokens: {preprocessor.tokenize(text)}")
        print(f"Features: {preprocessor.extract_features(text)}")
        print(f"Simple Sentiment: {preprocessor.sentiment_score_simple(text)}")
    
    # Test TF-IDF
    print("\n\nTesting TF-IDF Vectorizer:")
    vectorizer = TFIDFVectorizer(max_features=10)
    tfidf_matrix = vectorizer.fit_transform(sample_texts)
    
    print(f"Vocabulary: {list(vectorizer.vocabulary.keys())}")
    print(f"TF-IDF Matrix shape: {len(tfidf_matrix)} x {len(tfidf_matrix[0])}")
    print(f"First document vector: {tfidf_matrix[0][:5]}...")  # Show first 5 features