from flask import Flask, render_template, request, jsonify, redirect, url_for
import pandas as pd
import numpy as np
import json
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(__file__))

from models.sentiment_analysis import SentimentAnalyzer
from models.climate_prediction import ClimatePredictor
from utils.text_preprocessing import TextPreprocessor

app = Flask(__name__)

# Global variables to store models
sentiment_analyzer = None
climate_predictor = None
climate_data = None
comments_data = None

def load_models():
    """Load trained models and data"""
    global sentiment_analyzer, climate_predictor, climate_data, comments_data
    
    try:
        # Load sentiment analyzer
        sentiment_analyzer = SentimentAnalyzer()
        if os.path.exists('models/sentiment_model.json'):
            sentiment_analyzer.load_model('models/sentiment_model.json')
            print("Sentiment model loaded successfully")
        else:
            print("Sentiment model not found, will train on first use")
        
        # Load climate data
        if os.path.exists('data/climate_data.csv'):
            climate_data = pd.read_csv('data/climate_data.csv')
            print(f"Climate data loaded: {len(climate_data)} years")
        
        # Load comments data
        if os.path.exists('data/nasa_facebook_comments.csv'):
            comments_data = pd.read_csv('data/nasa_facebook_comments.csv')
            print(f"Comments data loaded: {len(comments_data)} comments")
        
        # Initialize climate predictor
        climate_predictor = ClimatePredictor()
        if climate_data is not None:
            climate_predictor.train(climate_data)
            print("Climate predictor trained successfully")
        
    except Exception as e:
        print(f"Error loading models: {e}")

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('index.html')

@app.route('/sentiment')
def sentiment_page():
    """Sentiment analysis page"""
    return render_template('sentiment.html')

@app.route('/climate')
def climate_page():
    """Climate prediction page"""
    return render_template('climate.html')

@app.route('/data')
def data_page():
    """Data exploration page"""
    return render_template('data.html')

@app.route('/api/sentiment/analyze', methods=['POST'])
def analyze_sentiment():
    """API endpoint for sentiment analysis"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        if not text.strip():
            return jsonify({'error': 'No text provided'}), 400
        
        # Train model if not loaded
        if sentiment_analyzer is None or not sentiment_analyzer.is_trained:
            train_sentiment_model()
        
        # Analyze sentiment
        prediction = sentiment_analyzer.predict([text])[0]
        probabilities = sentiment_analyzer.predict_proba([text])[0]
        
        # Get text features
        preprocessor = TextPreprocessor()
        features = preprocessor.extract_features(text)
        
        return jsonify({
            'text': text,
            'sentiment': prediction,
            'probabilities': probabilities,
            'features': features
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/climate/predict', methods=['POST'])
def predict_climate():
    """API endpoint for climate prediction"""
    try:
        data = request.get_json()
        year = data.get('year', 2030)
        
        if climate_predictor is None or not climate_predictor.is_trained:
            return jsonify({'error': 'Climate model not available'}), 500
        
        # Make prediction
        prediction = climate_predictor.predict_future([year])
        
        result = {
            'year': year,
            'predictions': {}
        }
        
        for col in prediction.columns:
            if col != 'Year':
                result['predictions'][col] = round(prediction[col].iloc[0], 2)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/climate/trends')
def get_climate_trends():
    """API endpoint for climate trends analysis"""
    try:
        if climate_predictor is None or climate_data is None:
            return jsonify({'error': 'Climate data not available'}), 500
        
        # Get trend analysis
        analysis, future_predictions = climate_predictor.analyze_trends(climate_data, future_years=20)
        
        # Prepare historical data
        historical_data = []
        for _, row in climate_data.iterrows():
            historical_data.append({
                'year': int(row['Year']),
                'temperature_anomaly': round(row['Temperature_Anomaly'], 2),
                'co2_level': round(row['CO2_Level'], 1),
                'sea_level_rise': round(row['Sea_Level_Rise'], 1),
                'arctic_ice_extent': round(row['Arctic_Ice_Extent'], 2),
                'extreme_weather_events': int(row['Extreme_Weather_Events'])
            })
        
        # Prepare future predictions
        future_data = []
        for _, row in future_predictions.iterrows():
            future_data.append({
                'year': int(row['Year']),
                'temperature_anomaly': round(row['Temperature_Anomaly'], 2),
                'co2_level': round(row['CO2_Level'], 1),
                'sea_level_rise': round(row['Sea_Level_Rise'], 1),
                'arctic_ice_extent': round(row['Arctic_Ice_Extent'], 2),
                'extreme_weather_events': int(row['Extreme_Weather_Events'])
            })
        
        return jsonify({
            'analysis': analysis,
            'historical_data': historical_data,
            'future_predictions': future_data
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/data/comments')
def get_comments_data():
    """API endpoint for comments data"""
    try:
        if comments_data is None:
            return jsonify({'error': 'Comments data not available'}), 500
        
        # Get summary statistics
        sentiment_counts = comments_data['Sentiment'].value_counts().to_dict()
        
        # Get sample comments
        sample_comments = []
        for sentiment in ['positive', 'negative', 'neutral']:
            sample = comments_data[comments_data['Sentiment'] == sentiment].head(3)
            for _, row in sample.iterrows():
                sample_comments.append({
                    'text': row['Text'],
                    'sentiment': row['Sentiment'],
                    'likes': int(row['LikesCount']),
                    'date': row['Date']
                })
        
        return jsonify({
            'total_comments': len(comments_data),
            'sentiment_distribution': sentiment_counts,
            'sample_comments': sample_comments,
            'date_range': {
                'start': comments_data['Date'].min(),
                'end': comments_data['Date'].max()
            }
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/data/climate')
def get_climate_data():
    """API endpoint for climate data"""
    try:
        if climate_data is None:
            return jsonify({'error': 'Climate data not available'}), 500
        
        # Get recent data (last 10 years)
        recent_data = climate_data.tail(10)
        
        data_summary = {
            'total_years': len(climate_data),
            'year_range': {
                'start': int(climate_data['Year'].min()),
                'end': int(climate_data['Year'].max())
            },
            'recent_trends': [],
            'statistics': {}
        }
        
        # Recent trends
        for _, row in recent_data.iterrows():
            data_summary['recent_trends'].append({
                'year': int(row['Year']),
                'temperature_anomaly': round(row['Temperature_Anomaly'], 2),
                'co2_level': round(row['CO2_Level'], 1),
                'sea_level_rise': round(row['Sea_Level_Rise'], 1),
                'arctic_ice_extent': round(row['Arctic_Ice_Extent'], 2),
                'extreme_weather_events': int(row['Extreme_Weather_Events'])
            })
        
        # Statistics
        for col in ['Temperature_Anomaly', 'CO2_Level', 'Sea_Level_Rise', 'Arctic_Ice_Extent']:
            data_summary['statistics'][col] = {
                'mean': round(climate_data[col].mean(), 2),
                'min': round(climate_data[col].min(), 2),
                'max': round(climate_data[col].max(), 2),
                'trend': 'increasing' if climate_data[col].iloc[-1] > climate_data[col].iloc[0] else 'decreasing'
            }
        
        return jsonify(data_summary)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def train_sentiment_model():
    """Train sentiment model if not already trained"""
    global sentiment_analyzer
    
    try:
        if comments_data is None:
            return False
        
        print("Training sentiment model...")
        sentiment_analyzer = SentimentAnalyzer()
        
        texts = comments_data['Text'].tolist()
        labels = comments_data['Sentiment'].tolist()
        
        sentiment_analyzer.train(texts, labels)
        sentiment_analyzer.save_model('models/sentiment_model.json')
        
        print("Sentiment model trained and saved")
        return True
    
    except Exception as e:
        print(f"Error training sentiment model: {e}")
        return False

@app.route('/api/train/sentiment', methods=['POST'])
def train_sentiment_api():
    """API endpoint to train sentiment model"""
    try:
        success = train_sentiment_model()
        if success:
            return jsonify({'message': 'Sentiment model trained successfully'})
        else:
            return jsonify({'error': 'Failed to train sentiment model'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_server_error(e):
    return render_template('500.html'), 500

if __name__ == '__main__':
    # Load models and data on startup
    load_models()
    
    # Run the app
    app.run(debug=True, host='0.0.0.0', port=5000)