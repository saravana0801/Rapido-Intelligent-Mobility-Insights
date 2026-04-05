import pandas as pd

def clean_data(df):
    df_clean = df.copy()
    
    # Target missing actual_ride_time_min, typical for uncompleted rides
    if 'actual_ride_time_min' in df_clean.columns and 'estimated_ride_time_min' in df_clean.columns:
        df_clean['actual_ride_time_min'] = df_clean['actual_ride_time_min'].fillna(df_clean['estimated_ride_time_min'])
    
    # Handle incomplete_ride_reason 
    if 'incomplete_ride_reason' in df_clean.columns:
        df_clean['incomplete_ride_reason'] = df_clean['incomplete_ride_reason'].fillna('None')
        
    # Fill remaining numeric missing values with median
    numeric_cols = df_clean.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_cols:
        if df_clean[col].isnull().sum() > 0:
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
            
    # Fill remaining categorical missing values with mode
    cat_cols = df_clean.select_dtypes(include=['object']).columns
    for col in cat_cols:
        if df_clean[col].isnull().sum() > 0:
            df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
            
    return df_clean

if __name__ == "__main__":
    from src import data_loader
    dfs = data_loader.load_data()
    merged = data_loader.merge_data(dfs)
    cleaned = clean_data(merged)
    print(f"Cleaned Dataset Shape: {cleaned.shape}")
    print(f"Missing Values After Cleaning:\n{cleaned.isnull().sum().sum()}")
