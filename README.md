# Climate Change Modeling & Sentiment Analysis Project

## Project Overview
This comprehensive data science project combines climate change modeling with sentiment analysis of NASA Facebook comments. The project aims to:

1. **Sentiment Analysis**: Analyze public sentiment from NASA's Facebook page comments about climate change
2. **Climate Prediction**: Build predictive models for climate change indicators
3. **Web Application**: Deploy models through a Flask web interface

## Dataset Description
- **Source**: NASA Facebook page comments (2020-2023)
- **Size**: 500+ user comments
- **Columns**:
  - `Date`: Comment timestamp
  - `LikesCount`: Number of likes received
  - `ProfileName`: Anonymized username
  - `CommentsCount`: Number of responses
  - `Text`: Comment content

## Project Structure
```
├── data/                   # Data files
├── models/                 # Trained models
├── notebooks/              # Jupyter notebooks for analysis
├── static/                 # CSS, JS files for web app
├── templates/              # HTML templates
├── utils/                  # Utility functions
├── app.py                  # Flask application
├── requirements.txt        # Dependencies
└── README.md              # Project documentation
```

## Technologies Used
- **Python**: Core programming language
- **Pandas & NumPy**: Data manipulation and analysis
- **Scikit-learn**: Machine learning models
- **Seaborn & Matplotlib**: Data visualization
- **Flask**: Web application framework
- **NLTK & TextBlob**: Natural language processing

## Installation & Setup
```bash
pip install -r requirements.txt
python app.py
```

## Features
1. **Exploratory Data Analysis** with comprehensive visualizations
2. **Sentiment Analysis** using machine learning
3. **Climate Change Prediction** models
4. **Interactive Web Dashboard** for model interaction
5. **Model Performance Metrics** and evaluation

## Usage
1. Run the Jupyter notebooks for analysis
2. Train models using the provided scripts
3. Launch the Flask app for interactive predictions

## Author
Data Science End-to-End Project - Climate Change Modeling
