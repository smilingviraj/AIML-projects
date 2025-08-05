import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta

class SimpleLinearRegression:
    """
    Simple linear regression implementation
    """
    
    def __init__(self):
        self.coefficients = None
        self.intercept = None
        self.is_fitted = False
    
    def fit(self, X, y):
        """
        Fit linear regression model
        """
        # Add bias term
        X_with_bias = np.column_stack([np.ones(len(X)), X])
        
        # Calculate coefficients using normal equation
        # θ = (X^T X)^(-1) X^T y
        XTX = np.dot(X_with_bias.T, X_with_bias)
        XTy = np.dot(X_with_bias.T, y)
        
        try:
            theta = np.linalg.solve(XTX, XTy)
            self.intercept = theta[0]
            self.coefficients = theta[1:]
            self.is_fitted = True
        except np.linalg.LinAlgError:
            # If matrix is singular, use pseudo-inverse
            theta = np.dot(np.linalg.pinv(XTX), XTy)
            self.intercept = theta[0]
            self.coefficients = theta[1:]
            self.is_fitted = True
    
    def predict(self, X):
        """
        Make predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        
        return self.intercept + np.dot(X, self.coefficients)
    
    def score(self, X, y):
        """
        Calculate R² score
        """
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)

class TimeSeriesPredictor:
    """
    Time series prediction using polynomial features and trend analysis
    """
    
    def __init__(self, degree=2):
        self.degree = degree
        self.models = {}
        self.scalers = {}
        self.is_fitted = False
        
    def create_polynomial_features(self, X, degree):
        """
        Create polynomial features
        """
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        
        features = [X]
        for d in range(2, degree + 1):
            features.append(X ** d)
        
        return np.column_stack(features)
    
    def fit(self, years, target_data):
        """
        Fit time series models for each target variable
        """
        # Normalize years to start from 0
        self.base_year = min(years)
        normalized_years = np.array(years) - self.base_year
        
        for column in target_data.columns:
            y = target_data[column].values
            
            # Create polynomial features
            X_poly = self.create_polynomial_features(normalized_years, self.degree)
            
            # Fit model
            model = SimpleLinearRegression()
            model.fit(X_poly, y)
            
            self.models[column] = model
            
            # Store scaling info
            self.scalers[column] = {
                'mean': np.mean(y),
                'std': np.std(y)
            }
        
        self.is_fitted = True
    
    def predict(self, future_years):
        """
        Predict future values
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Normalize future years
        normalized_years = np.array(future_years) - self.base_year
        X_poly = self.create_polynomial_features(normalized_years, self.degree)
        
        predictions = {}
        for column, model in self.models.items():
            predictions[column] = model.predict(X_poly)
        
        return predictions
    
    def evaluate(self, years, target_data):
        """
        Evaluate model performance
        """
        normalized_years = np.array(years) - self.base_year
        X_poly = self.create_polynomial_features(normalized_years, self.degree)
        
        scores = {}
        for column in target_data.columns:
            y_true = target_data[column].values
            model = self.models[column]
            score = model.score(X_poly, y_true)
            scores[column] = score
        
        return scores

class ClimatePredictor:
    """
    Complete climate prediction system
    """
    
    def __init__(self, degree=2):
        self.temperature_model = TimeSeriesPredictor(degree=degree)
        self.co2_model = TimeSeriesPredictor(degree=degree)
        self.sea_level_model = TimeSeriesPredictor(degree=degree)
        self.ice_model = TimeSeriesPredictor(degree=degree)
        self.extreme_events_model = TimeSeriesPredictor(degree=degree)
        self.is_trained = False
        
    def train(self, climate_data):
        """
        Train all climate prediction models
        """
        print(f"Training climate prediction models on {len(climate_data)} years of data...")
        
        years = climate_data['Year'].values
        
        # Train individual models
        temp_data = climate_data[['Temperature_Anomaly', 'Global_Temperature']]
        self.temperature_model.fit(years, temp_data)
        
        co2_data = climate_data[['CO2_Level']]
        self.co2_model.fit(years, co2_data)
        
        sea_level_data = climate_data[['Sea_Level_Rise']]
        self.sea_level_model.fit(years, sea_level_data)
        
        ice_data = climate_data[['Arctic_Ice_Extent']]
        self.ice_model.fit(years, ice_data)
        
        extreme_data = climate_data[['Extreme_Weather_Events']]
        self.extreme_events_model.fit(years, extreme_data)
        
        self.is_trained = True
        print("Climate models training completed!")
        
        # Evaluate models
        self.evaluate_models(climate_data)
    
    def evaluate_models(self, climate_data):
        """
        Evaluate all models
        """
        years = climate_data['Year'].values
        
        print("\nModel Performance (R² scores):")
        
        # Temperature model
        temp_data = climate_data[['Temperature_Anomaly', 'Global_Temperature']]
        temp_scores = self.temperature_model.evaluate(years, temp_data)
        print(f"Temperature Model:")
        for var, score in temp_scores.items():
            print(f"  {var}: {score:.3f}")
        
        # CO2 model
        co2_data = climate_data[['CO2_Level']]
        co2_scores = self.co2_model.evaluate(years, co2_data)
        print(f"CO2 Model:")
        for var, score in co2_scores.items():
            print(f"  {var}: {score:.3f}")
        
        # Sea level model
        sea_level_data = climate_data[['Sea_Level_Rise']]
        sea_level_scores = self.sea_level_model.evaluate(years, sea_level_data)
        print(f"Sea Level Model:")
        for var, score in sea_level_scores.items():
            print(f"  {var}: {score:.3f}")
        
        # Ice model
        ice_data = climate_data[['Arctic_Ice_Extent']]
        ice_scores = self.ice_model.evaluate(years, ice_data)
        print(f"Arctic Ice Model:")
        for var, score in ice_scores.items():
            print(f"  {var}: {score:.3f}")
        
        # Extreme events model
        extreme_data = climate_data[['Extreme_Weather_Events']]
        extreme_scores = self.extreme_events_model.evaluate(years, extreme_data)
        print(f"Extreme Events Model:")
        for var, score in extreme_scores.items():
            print(f"  {var}: {score:.3f}")
    
    def predict_future(self, future_years):
        """
        Predict climate variables for future years
        """
        if not self.is_trained:
            raise ValueError("Models must be trained before prediction")
        
        predictions = {}
        
        # Temperature predictions
        temp_pred = self.temperature_model.predict(future_years)
        predictions.update(temp_pred)
        
        # CO2 predictions
        co2_pred = self.co2_model.predict(future_years)
        predictions.update(co2_pred)
        
        # Sea level predictions
        sea_level_pred = self.sea_level_model.predict(future_years)
        predictions.update(sea_level_pred)
        
        # Ice predictions
        ice_pred = self.ice_model.predict(future_years)
        predictions.update(ice_pred)
        
        # Extreme events predictions
        extreme_pred = self.extreme_events_model.predict(future_years)
        predictions.update(extreme_pred)
        
        # Create DataFrame
        future_df = pd.DataFrame({
            'Year': future_years,
            'Temperature_Anomaly': predictions['Temperature_Anomaly'],
            'Global_Temperature': predictions['Global_Temperature'],
            'CO2_Level': predictions['CO2_Level'],
            'Sea_Level_Rise': predictions['Sea_Level_Rise'],
            'Arctic_Ice_Extent': predictions['Arctic_Ice_Extent'],
            'Extreme_Weather_Events': predictions['Extreme_Weather_Events']
        })
        
        return future_df
    
    def analyze_trends(self, climate_data, future_years=10):
        """
        Analyze current trends and make future projections
        """
        if not self.is_trained:
            raise ValueError("Models must be trained before analysis")
        
        current_year = climate_data['Year'].max()
        future_year_list = list(range(current_year + 1, current_year + future_years + 1))
        
        # Get current values
        current_data = climate_data[climate_data['Year'] == current_year].iloc[0]
        
        # Predict future values
        future_predictions = self.predict_future(future_year_list)
        future_data = future_predictions.iloc[-1]  # Last year of predictions
        
        # Calculate changes
        analysis = {
            'current_year': current_year,
            'future_year': current_year + future_years,
            'projections': {}
        }
        
        variables = ['Temperature_Anomaly', 'Global_Temperature', 'CO2_Level', 
                    'Sea_Level_Rise', 'Arctic_Ice_Extent', 'Extreme_Weather_Events']
        
        for var in variables:
            current_val = current_data[var]
            future_val = future_data[var]
            change = future_val - current_val
            percent_change = (change / current_val * 100) if current_val != 0 else 0
            
            analysis['projections'][var] = {
                'current_value': round(current_val, 2),
                'future_value': round(future_val, 2),
                'absolute_change': round(change, 2),
                'percent_change': round(percent_change, 1)
            }
        
        return analysis, future_predictions
    
    def save_models(self, filepath):
        """
        Save trained models
        """
        if not self.is_trained:
            raise ValueError("Models must be trained before saving")
        
        model_data = {
            'temperature_model': {
                'base_year': self.temperature_model.base_year,
                'degree': self.temperature_model.degree,
                'models': {},
                'scalers': self.temperature_model.scalers
            },
            'co2_model': {
                'base_year': self.co2_model.base_year,
                'degree': self.co2_model.degree,
                'models': {},
                'scalers': self.co2_model.scalers
            },
            'sea_level_model': {
                'base_year': self.sea_level_model.base_year,
                'degree': self.sea_level_model.degree,
                'models': {},
                'scalers': self.sea_level_model.scalers
            },
            'ice_model': {
                'base_year': self.ice_model.base_year,
                'degree': self.ice_model.degree,
                'models': {},
                'scalers': self.ice_model.scalers
            },
            'extreme_events_model': {
                'base_year': self.extreme_events_model.base_year,
                'degree': self.extreme_events_model.degree,
                'models': {},
                'scalers': self.extreme_events_model.scalers
            }
        }
        
        # Save model coefficients
        for model_name, model_obj in [
            ('temperature_model', self.temperature_model),
            ('co2_model', self.co2_model),
            ('sea_level_model', self.sea_level_model),
            ('ice_model', self.ice_model),
            ('extreme_events_model', self.extreme_events_model)
        ]:
            for var_name, var_model in model_obj.models.items():
                model_data[model_name]['models'][var_name] = {
                    'coefficients': var_model.coefficients.tolist(),
                    'intercept': var_model.intercept
                }
        
        with open(filepath, 'w') as f:
            json.dump(model_data, f, indent=2)
        
        print(f"Climate models saved to {filepath}")

def train_climate_models():
    """
    Train climate prediction models
    """
    # Load climate data
    print("Loading climate data...")
    climate_df = pd.read_csv('../data/climate_data.csv')
    
    print(f"Loaded {len(climate_df)} years of climate data ({climate_df['Year'].min()}-{climate_df['Year'].max()})")
    
    # Initialize and train predictor
    predictor = ClimatePredictor(degree=2)
    predictor.train(climate_df)
    
    # Analyze trends and make future projections
    print("\nAnalyzing climate trends...")
    analysis, future_predictions = predictor.analyze_trends(climate_df, future_years=20)
    
    print(f"\nClimate Change Projections ({analysis['current_year']} to {analysis['future_year']}):")
    for var, proj in analysis['projections'].items():
        print(f"\n{var}:")
        print(f"  Current: {proj['current_value']}")
        print(f"  Projected: {proj['future_value']}")
        print(f"  Change: {proj['absolute_change']} ({proj['percent_change']:+.1f}%)")
    
    # Save future predictions
    future_predictions.to_csv('../data/climate_predictions.csv', index=False)
    print(f"\nFuture predictions saved to climate_predictions.csv")
    
    # Save models
    predictor.save_models('../models/climate_models.json')
    
    # Test specific year prediction
    print(f"\nSample prediction for year 2030:")
    sample_prediction = predictor.predict_future([2030])
    for col in sample_prediction.columns:
        if col != 'Year':
            print(f"  {col}: {sample_prediction[col].iloc[0]:.2f}")
    
    return predictor, analysis, future_predictions

if __name__ == "__main__":
    # Train climate models
    predictor, analysis, future_predictions = train_climate_models()
    print("\nClimate prediction models training completed!")