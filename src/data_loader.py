import pandas as pd
import os

def load_data(data_dir="Rapido_dataset"):
    bookings = pd.read_csv(os.path.join(data_dir, "bookings.csv"))
    customers = pd.read_csv(os.path.join(data_dir, "customers.csv"))
    drivers = pd.read_csv(os.path.join(data_dir, "drivers.csv"))
    location_demand = pd.read_csv(os.path.join(data_dir, "location_demand.csv"))
    
    try:
        time_features = pd.read_csv(os.path.join(data_dir, "time_features.csv"))
    except:
        time_features = pd.read_excel(os.path.join(data_dir, "time_features.xlsx"))
        
    return {
        "bookings": bookings,
        "customers": customers,
        "drivers": drivers,
        "location_demand": location_demand,
        "time_features": time_features
    }

def merge_data(dfs):
    bookings = dfs["bookings"].copy()
    customers = dfs["customers"].copy()
    drivers = dfs["drivers"].copy()
    location_demand = dfs["location_demand"].copy()
    time_features = dfs["time_features"].copy()
    
    # Create a merged datetime feature
    bookings['datetime'] = pd.to_datetime(bookings['booking_date'] + ' ' + bookings['booking_time'])
    time_features['datetime'] = pd.to_datetime(time_features['datetime'])
    
    # Merge bookings with customers
    merged_df = pd.merge(bookings, customers, on="customer_id", how="left")
    
    # Merge with drivers
    merged_df = pd.merge(merged_df, drivers.drop(columns=['vehicle_type']), on="driver_id", how="left")
    
    # Merge with location demand
    merged_df = pd.merge(merged_df, location_demand, 
                         on=['city', 'pickup_location', 'hour_of_day', 'vehicle_type'], 
                         how="left", suffixes=('', '_demand'))
    
    # Merge with time features, keeping only non-overlapping columns
    cols_to_use = time_features.columns.difference(merged_df.columns).tolist() + ['datetime']
    merged_df = pd.merge(merged_df, time_features[cols_to_use], on="datetime", how="left")
    
    return merged_df

if __name__ == "__main__":
    print("Loading data...")
    dfs = load_data()
    print("Merging data...")
    merged_df = merge_data(dfs)
    print(f"Merged Dataset Shape: {merged_df.shape}")
    print("Merged Dataset Columns:", merged_df.columns.tolist())
