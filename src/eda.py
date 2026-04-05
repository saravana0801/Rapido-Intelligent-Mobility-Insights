import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from src import data_loader, preprocessing
import warnings
warnings.filterwarnings('ignore')

print("Loading and cleaning data for EDA...")
dfs = data_loader.load_data()
merged = data_loader.merge_data(dfs)
df = preprocessing.clean_data(merged)

# Set style
sns.set_theme(style="whitegrid")

# ──────────────────────────────────────────────
# 1. Ride volume by hour
# ──────────────────────────────────────────────
plt.figure()
sns.countplot(x='hour_of_day', data=df, palette='viridis')
plt.title("Ride Volume by Hour of Day")
plt.xlabel("Hour of Day")
plt.ylabel("Number of Rides")
plt.tight_layout()
plt.savefig('plots/ride_volume_by_hour.png')
plt.close()
print("✅ Plot 1: Ride volume by hour saved.")

# ──────────────────────────────────────────────
# 2. Ride volume by weekday
# ──────────────────────────────────────────────
weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
plt.figure(figsize=(9, 4))
day_counts = df['day_of_week'].value_counts().reindex(weekday_order, fill_value=0)
sns.barplot(x=day_counts.index, y=day_counts.values, palette='magma')
plt.title("Ride Volume by Weekday")
plt.xlabel("Day of Week")
plt.ylabel("Number of Rides")
plt.tight_layout()
plt.savefig('plots/ride_volume_by_weekday.png')
plt.close()
print("✅ Plot 2: Ride volume by weekday saved.")

# ──────────────────────────────────────────────
# 3. Distance vs Fare correlation
# ──────────────────────────────────────────────
plt.figure()
sns.scatterplot(x='ride_distance_km', y='booking_value', data=df.sample(min(2000, len(df))), alpha=0.5)
plt.title("Distance vs Fare")
plt.xlabel("Ride Distance (km)")
plt.ylabel("Booking Value (₹)")
plt.tight_layout()
plt.savefig('plots/distance_vs_fare.png')
plt.close()
print("✅ Plot 3: Distance vs Fare saved.")

# ──────────────────────────────────────────────
# 4. Cancellation heatmap across cities (city × hour)
# ──────────────────────────────────────────────
df['is_cancelled'] = (df['booking_status'] == 'Cancelled').astype(int)
pivot = df.pivot_table(values='is_cancelled', index='city', columns='hour_of_day', aggfunc='mean')
plt.figure(figsize=(16, 5))
sns.heatmap(pivot, cmap='YlOrRd', linewidths=0.3, annot=False, fmt='.2f',
            cbar_kws={'label': 'Cancellation Rate'})
plt.title("Cancellation Heatmap — City × Hour of Day")
plt.xlabel("Hour of Day")
plt.ylabel("City")
plt.tight_layout()
plt.savefig('plots/cancellation_heatmap_city.png')
plt.close()
print("✅ Plot 4: Cancellation heatmap (city × hour) saved.")

# ──────────────────────────────────────────────
# 5. Cancellation rate by city (bar)
# ──────────────────────────────────────────────
cancel_by_city = df.groupby('city')['is_cancelled'].mean().reset_index()
plt.figure()
sns.barplot(x='city', y='is_cancelled', data=cancel_by_city, palette='coolwarm')
plt.title("Cancellation Rate by City")
plt.xlabel("City")
plt.ylabel("Cancellation Rate")
plt.tight_layout()
plt.savefig('plots/cancellation_by_city.png')
plt.close()
print("✅ Plot 5: Cancellation by city saved.")

# ──────────────────────────────────────────────
# 6. Rating distribution
# ──────────────────────────────────────────────
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
sns.histplot(df['avg_customer_rating'].dropna(), bins=20, kde=True, color='blue')
plt.title("Customer Rating Distribution")
plt.subplot(1, 2, 2)
sns.histplot(df['avg_driver_rating'].dropna(), bins=20, kde=True, color='green')
plt.title("Driver Rating Distribution")
plt.tight_layout()
plt.savefig('plots/rating_distribution.png')
plt.close()
print("✅ Plot 6: Rating distribution saved.")

# ──────────────────────────────────────────────
# 7. Customer vs Driver behaviour comparison
# ──────────────────────────────────────────────
plt.figure()
behavior_data = pd.DataFrame({
    'Metric': ['Customer Cancel Rate', 'Driver Delay Rate'],
    'Average %': [df['cancellation_rate'].mean() * 100, df['delay_rate'].mean() * 100]
})
sns.barplot(x='Metric', y='Average %', data=behavior_data, palette='Set2')
plt.title("Customer vs Driver Behavior")
plt.tight_layout()
plt.savefig('plots/customer_vs_driver_behavior.png')
plt.close()
print("✅ Plot 7: Customer vs Driver behaviour saved.")

# ──────────────────────────────────────────────
# 8. Payment method usage patterns
# ──────────────────────────────────────────────
if 'payment_method' in df.columns:
    plt.figure()
    payment_counts = df['payment_method'].value_counts()
    sns.barplot(x=payment_counts.index, y=payment_counts.values, palette='pastel')
    plt.title("Payment Method Usage Patterns")
    plt.xlabel("Payment Method")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig('plots/payment_method_usage.png')
    plt.close()
    print("✅ Plot 8: Payment method usage saved.")
else:
    print("⚠️  Plot 8: 'payment_method' column not present in dataset — skipping plot.")
    # Save a placeholder note image
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.text(0.5, 0.5,
            "Payment Method data not available in the current dataset.\n"
            "This field is not collected in bookings.csv.",
            ha='center', va='center', fontsize=13, color='#666',
            transform=ax.transAxes, wrap=True)
    ax.axis('off')
    fig.patch.set_facecolor('#f9f9f9')
    plt.tight_layout()
    plt.savefig('plots/payment_method_usage.png')
    plt.close()

# ──────────────────────────────────────────────
# 9. Traffic/Weather vs Cancellation
# ──────────────────────────────────────────────
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
traffic_cancel = df.groupby('traffic_level')['is_cancelled'].mean().reset_index()
sns.barplot(x='traffic_level', y='is_cancelled', data=traffic_cancel, palette='Reds')
plt.title("Cancellation Rate by Traffic")

plt.subplot(1, 2, 2)
weather_cancel = df.groupby('weather_condition')['is_cancelled'].mean().reset_index()
sns.barplot(x='weather_condition', y='is_cancelled', data=weather_cancel, palette='Blues')
plt.title("Cancellation Rate by Weather")
plt.tight_layout()
plt.savefig('plots/traffic_weather_vs_cancellation.png')
plt.close()
print("✅ Plot 9: Traffic/Weather vs Cancellation saved.")

# ──────────────────────────────────────────────
# 10. Pickup / Drop city heatmaps (top locations)
# ──────────────────────────────────────────────
top_n = 10

# Top pickup locations
top_pickups = df['pickup_location'].value_counts().head(top_n)
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
sns.barplot(x=top_pickups.values, y=top_pickups.index, palette='Blues_r', ax=axes[0])
axes[0].set_title(f"Top {top_n} Pickup Locations")
axes[0].set_xlabel("Number of Rides")
axes[0].set_ylabel("Location")

# Top drop locations
top_drops = df['drop_location'].value_counts().head(top_n)
sns.barplot(x=top_drops.values, y=top_drops.index, palette='Greens_r', ax=axes[1])
axes[1].set_title(f"Top {top_n} Drop Locations")
axes[1].set_xlabel("Number of Rides")
axes[1].set_ylabel("")

plt.tight_layout()
plt.savefig('plots/pickup_drop_heatmap.png')
plt.close()
print("✅ Plot 10: Pickup/Drop city heatmap saved.")

# ──────────────────────────────────────────────
# 11. Surge behavior patterns by hour
# ──────────────────────────────────────────────
plt.figure(figsize=(10, 4))
surge_by_hour = df.groupby('hour_of_day')['surge_multiplier'].mean().reset_index()
sns.lineplot(x='hour_of_day', y='surge_multiplier', data=surge_by_hour,
             marker='o', color='#f9a825', linewidth=2.5)
plt.fill_between(surge_by_hour['hour_of_day'], surge_by_hour['surge_multiplier'],
                 alpha=0.15, color='#f9a825')
plt.title("Avg Surge Multiplier by Hour of Day")
plt.xlabel("Hour of Day")
plt.ylabel("Avg Surge Multiplier")
plt.tight_layout()
plt.savefig('plots/surge_behavior_by_hour.png')
plt.close()
print("✅ Plot 11: Surge behavior by hour saved.")

# ──────────────────────────────────────────────
# 12. Customer vs Driver cancellation reasons
# ──────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Customer: cancellation rate bucketed by loyalty score
df['loyalty_bucket'] = pd.cut(df['Customer_Loyalty_Score'] if 'Customer_Loyalty_Score' in df.columns
                               else df['cancellation_rate'],
                               bins=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
cancel_by_loyalty = df.groupby('loyalty_bucket', observed=False)['is_cancelled'].mean().reset_index()
sns.barplot(x='loyalty_bucket', y='is_cancelled', data=cancel_by_loyalty,
            palette='Reds', ax=axes[0])
axes[0].set_title("Customer Cancel Rate by Loyalty Bucket")
axes[0].set_xlabel("Customer Loyalty")
axes[0].set_ylabel("Cancellation Rate")

# Driver: delay rate by driver experience / reliability
if 'driver_experience_years' in df.columns:
    df['exp_bucket'] = pd.cut(df['driver_experience_years'], bins=4,
                               labels=['0-1 yr', '1-3 yrs', '3-6 yrs', '6+ yrs'])
    delay_by_exp = df.groupby('exp_bucket', observed=False)['driver_delay_flag'].mean().reset_index()
    sns.barplot(x='exp_bucket', y='driver_delay_flag', data=delay_by_exp,
                palette='Blues', ax=axes[1])
    axes[1].set_title("Driver Delay Rate by Experience")
    axes[1].set_xlabel("Driver Experience")
    axes[1].set_ylabel("Delay Rate")
else:
    delay_by_traffic = df.groupby('traffic_level')['driver_delay_flag'].mean().reset_index()
    sns.barplot(x='traffic_level', y='driver_delay_flag', data=delay_by_traffic,
                palette='Blues', ax=axes[1])
    axes[1].set_title("Driver Delay Rate by Traffic Level")
    axes[1].set_xlabel("Traffic Level")

plt.tight_layout()
plt.savefig('plots/customer_driver_cancel_reasons.png')
plt.close()
print("✅ Plot 12: Customer vs Driver cancellation reasons saved.")

print("\n✅ All EDA plots saved in plots/ directory.")
