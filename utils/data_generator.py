import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_climate_comments_dataset(num_comments=500):
    """
    Generate synthetic NASA Facebook comments dataset about climate change
    """
    
    # Set random seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Sample climate change related comments (positive, negative, neutral)
    positive_comments = [
        "Thank you NASA for your incredible work on climate research! This data is crucial for our future.",
        "Amazing work! This helps us understand climate change better. Keep it up NASA!",
        "This is exactly the kind of research we need to combat climate change. Excellent job!",
        "NASA's climate data is invaluable for policy makers. Thank you for your transparency!",
        "Great visualization of climate trends. This makes the data accessible to everyone.",
        "Love seeing NASA's commitment to climate science. This research is vital!",
        "Fantastic work on climate modeling. This helps us prepare for the future.",
        "NASA continues to lead in climate research. This data is eye-opening!",
        "Thank you for making climate science accessible to the public. Great work!",
        "This climate data visualization is incredibly helpful for understanding trends."
    ]
    
    negative_comments = [
        "I don't believe this climate data. NASA is pushing an agenda.",
        "This climate change stuff is just fear mongering. Not buying it.",
        "NASA wasting taxpayer money on climate propaganda instead of space exploration.",
        "Climate has always changed naturally. This is not human-caused.",
        "Stop the climate alarmism! Focus on real space missions instead.",
        "This climate data is manipulated to support political narratives.",
        "NASA should stick to space, not climate politics.",
        "Climate change is a hoax. NASA knows better than this.",
        "Why is NASA involved in climate activism? Stick to rockets!",
        "This climate fear mongering needs to stop. Focus on actual science."
    ]
    
    neutral_comments = [
        "Interesting climate data. Would like to see more analysis on regional variations.",
        "Can someone explain how this climate model works? Curious about the methodology.",
        "What are the implications of this climate data for coastal cities?",
        "How does this climate trend compare to historical patterns?",
        "Would be helpful to see uncertainty ranges in this climate projection.",
        "What's the timeframe for these climate predictions?",
        "How accurate have previous NASA climate models been?",
        "Can this climate data be used for local planning decisions?",
        "What factors are included in this climate analysis?",
        "How often is this climate dataset updated?"
    ]
    
    # Profile names (anonymized)
    profile_names = [
        f"User_{i:03d}" for i in range(1, 201)
    ]
    
    # Generate dataset
    data = []
    
    for i in range(num_comments):
        # Random comment selection
        comment_type = np.random.choice(['positive', 'negative', 'neutral'], 
                                      p=[0.4, 0.3, 0.3])
        
        if comment_type == 'positive':
            text = random.choice(positive_comments)
        elif comment_type == 'negative':
            text = random.choice(negative_comments)
        else:
            text = random.choice(neutral_comments)
        
        # Generate random date between 2020 and 2023
        start_date = datetime(2020, 1, 1)
        end_date = datetime(2023, 12, 31)
        random_date = start_date + timedelta(
            days=random.randint(0, (end_date - start_date).days)
        )
        
        # Generate likes count (influenced by sentiment)
        if comment_type == 'positive':
            likes_count = np.random.poisson(25)  # Higher likes for positive
        elif comment_type == 'negative':
            likes_count = np.random.poisson(8)   # Lower likes for negative
        else:
            likes_count = np.random.poisson(15)  # Medium likes for neutral
        
        # Generate comments count
        comments_count = np.random.poisson(3)
        
        # Random profile name
        profile_name = random.choice(profile_names)
        
        data.append({
            'Date': random_date.strftime('%Y-%m-%d %H:%M:%S'),
            'LikesCount': max(0, likes_count),
            'ProfileName': profile_name,
            'CommentsCount': max(0, comments_count),
            'Text': text,
            'Sentiment': comment_type  # Adding ground truth for evaluation
        })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Sort by date
    df = df.sort_values('Date').reset_index(drop=True)
    
    return df

def generate_climate_data(num_years=50):
    """
    Generate synthetic climate data for modeling
    """
    np.random.seed(42)
    
    # Generate years from 1974 to 2023
    years = list(range(2024 - num_years, 2024))
    
    # Base temperature with warming trend
    base_temp = 14.0  # Global average temperature baseline
    warming_rate = 0.02  # Degrees per year
    
    data = []
    
    for i, year in enumerate(years):
        # Temperature with warming trend + noise
        temp_anomaly = warming_rate * i + np.random.normal(0, 0.3)
        temperature = base_temp + temp_anomaly
        
        # CO2 levels (increasing trend)
        co2_base = 330  # ppm baseline
        co2_increase = 2.0  # ppm per year
        co2_level = co2_base + co2_increase * i + np.random.normal(0, 5)
        
        # Sea level (rising trend)
        sea_level_base = 0  # mm baseline
        sea_level_rise = 3.2  # mm per year
        sea_level = sea_level_base + sea_level_rise * i + np.random.normal(0, 2)
        
        # Arctic ice extent (decreasing trend)
        ice_base = 7.0  # million km²
        ice_decline = 0.05  # million km² per year
        ice_extent = ice_base - ice_decline * i + np.random.normal(0, 0.3)
        ice_extent = max(3.0, ice_extent)  # Minimum threshold
        
        # Extreme weather events (increasing trend)
        extreme_events = max(0, int(10 + 0.3 * i + np.random.normal(0, 3)))
        
        data.append({
            'Year': year,
            'Temperature_Anomaly': round(temp_anomaly, 2),
            'Global_Temperature': round(temperature, 2),
            'CO2_Level': round(co2_level, 1),
            'Sea_Level_Rise': round(sea_level, 1),
            'Arctic_Ice_Extent': round(ice_extent, 2),
            'Extreme_Weather_Events': extreme_events
        })
    
    return pd.DataFrame(data)

if __name__ == "__main__":
    # Generate and save datasets
    print("Generating NASA Facebook comments dataset...")
    comments_df = generate_climate_comments_dataset(500)
    comments_df.to_csv('../data/nasa_facebook_comments.csv', index=False)
    print(f"Generated {len(comments_df)} comments")
    
    print("Generating climate data...")
    climate_df = generate_climate_data(50)
    climate_df.to_csv('../data/climate_data.csv', index=False)
    print(f"Generated {len(climate_df)} years of climate data")
    
    print("Data generation completed!")