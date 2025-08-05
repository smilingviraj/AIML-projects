#!/usr/bin/env python3
"""
Training script for Climate Change Modeling project
Trains both sentiment analysis and climate prediction models
"""

import os
import sys
import time

# Add project root to path
sys.path.append(os.path.dirname(__file__))

def train_all_models():
    """Train all models in the project"""
    print("=" * 60)
    print("CLIMATE CHANGE MODELING - MODEL TRAINING")
    print("=" * 60)
    print()
    
    start_time = time.time()
    
    try:
        # Train sentiment analysis model
        print("1. Training Sentiment Analysis Model...")
        print("-" * 40)
        
        from models.sentiment_analysis import train_sentiment_model
        analyzer, results = train_sentiment_model()
        
        print(f"✓ Sentiment model trained successfully!")
        print(f"  - Accuracy: {results['accuracy']:.3f}")
        print()
        
        # Train climate prediction models
        print("2. Training Climate Prediction Models...")
        print("-" * 40)
        
        from models.climate_prediction import train_climate_models
        predictor, analysis, future_predictions = train_climate_models()
        
        print(f"✓ Climate models trained successfully!")
        print(f"  - Models saved for future use")
        print()
        
        # Summary
        total_time = time.time() - start_time
        print("=" * 60)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"Total training time: {total_time:.2f} seconds")
        print()
        print("Models trained:")
        print("  ✓ Sentiment Analysis (Naive Bayes + TF-IDF)")
        print("  ✓ Climate Prediction (Polynomial Regression)")
        print()
        print("Files created:")
        print("  - models/sentiment_model.json")
        print("  - models/climate_models.json")
        print("  - data/climate_predictions.csv")
        print()
        print("Next steps:")
        print("  1. Run 'python3 app.py' to start the web application")
        print("  2. Open http://localhost:5000 in your browser")
        print("  3. Explore the sentiment analysis and climate prediction tools")
        print()
        
        return True
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("Make sure all required dependencies are installed.")
        print("Try running: pip install -r requirements.txt")
        return False
        
    except FileNotFoundError as e:
        print(f"❌ File Error: {e}")
        print("Make sure the data files exist:")
        print("  - data/nasa_facebook_comments.csv")
        print("  - data/climate_data.csv")
        print("Run the data generation script first if needed.")
        return False
        
    except Exception as e:
        print(f"❌ Training Error: {e}")
        print("Check the error details above and try again.")
        return False

def check_data_files():
    """Check if required data files exist"""
    required_files = [
        'data/nasa_facebook_comments.csv',
        'data/climate_data.csv'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ Missing required data files:")
        for file_path in missing_files:
            print(f"  - {file_path}")
        print()
        print("Please run the data generation script first:")
        print("  cd utils && python3 simple_data_generator.py")
        return False
    
    return True

def check_directories():
    """Create required directories if they don't exist"""
    required_dirs = ['models', 'data']
    
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            print(f"Created directory: {dir_path}")

if __name__ == "__main__":
    print("Climate Change Modeling - Training Script")
    print("=========================================")
    print()
    
    # Check and create directories
    check_directories()
    
    # Check if data files exist
    if not check_data_files():
        print("Generating missing data files...")
        try:
            # Generate data if missing
            from utils.simple_data_generator import generate_climate_comments_dataset, generate_climate_data, save_to_csv
            
            print("Generating NASA Facebook comments dataset...")
            comments_data = generate_climate_comments_dataset(500)
            comments_fieldnames = ['Date', 'LikesCount', 'ProfileName', 'CommentsCount', 'Text', 'Sentiment']
            save_to_csv(comments_data, 'data/nasa_facebook_comments.csv', comments_fieldnames)
            
            print("Generating climate data...")
            climate_data = generate_climate_data(50)
            climate_fieldnames = ['Year', 'Temperature_Anomaly', 'Global_Temperature', 'CO2_Level', 
                                 'Sea_Level_Rise', 'Arctic_Ice_Extent', 'Extreme_Weather_Events']
            save_to_csv(climate_data, 'data/climate_data.csv', climate_fieldnames)
            
            print("✓ Data files generated successfully!")
            print()
            
        except Exception as e:
            print(f"❌ Error generating data: {e}")
            sys.exit(1)
    
    # Train all models
    success = train_all_models()
    
    if success:
        print("🎉 All models trained successfully!")
        print("You can now run the web application: python3 app.py")
        sys.exit(0)
    else:
        print("❌ Training failed. Please check the errors above.")
        sys.exit(1)