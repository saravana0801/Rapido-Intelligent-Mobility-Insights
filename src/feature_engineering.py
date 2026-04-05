import pandas as pd
import numpy as np

def create_features(df):
    df_feat = df.copy()
    
    # Fare_per_KM
    df_feat['Fare_per_KM'] = df_feat['booking_value'] / df_feat['ride_distance_km'].replace(0, np.nan)
    df_feat['Fare_per_KM'] = df_feat['Fare_per_KM'].fillna(0)
    
    # Fare_per_Min
    df_feat['Fare_per_Min'] = df_feat['booking_value'] / df_feat['estimated_ride_time_min'].replace(0, np.nan)
    df_feat['Fare_per_Min'] = df_feat['Fare_per_Min'].fillna(0)
    
    # Rush_Hour_Flag (assume 8-10 AM, 5-8 PM)
    df_feat['Rush_Hour_Flag'] = df_feat['hour_of_day'].apply(lambda x: 1 if (8 <= x <= 10) or (17 <= x <= 20) else 0)
    
    # Long_Distance_Flag (> 15 km)
    df_feat['Long_Distance_Flag'] = (df_feat['ride_distance_km'] > 15).astype(int)
    
    # City_Pair = Pickup + Drop
    df_feat['City_Pair'] = df_feat['pickup_location'].astype(str) + "_" + df_feat['drop_location'].astype(str)
    
    # Driver_Reliability_Score
    if 'acceptance_rate' in df_feat.columns and 'delay_rate' in df_feat.columns:
        df_feat['Driver_Reliability_Score'] = (df_feat['acceptance_rate'] * (1 - df_feat['delay_rate'])) * 100
    else:
        df_feat['Driver_Reliability_Score'] = 0
        
    # Customer_Loyalty_Score
    if 'total_bookings' in df_feat.columns and 'cancellation_rate' in df_feat.columns:
        df_feat['Customer_Loyalty_Score'] = df_feat['total_bookings'] * (1 - df_feat['cancellation_rate'])
    else:
        df_feat['Customer_Loyalty_Score'] = 0
        
    return df_feat

if __name__ == "__main__":
    from src import data_loader, preprocessing
    dfs = data_loader.load_data()
    merged = data_loader.merge_data(dfs)
    cleaned = preprocessing.clean_data(merged)
    engineered = create_features(cleaned)
    print(f"Engineered Features Shape: {engineered.shape}")
    print("New Features Example:")
    print(engineered[['Fare_per_KM', 'Fare_per_Min', 'Rush_Hour_Flag', 'Long_Distance_Flag', 'City_Pair', 'Driver_Reliability_Score', 'Customer_Loyalty_Score']].head())
