import pandas as pd
import numpy as np
import os

DAYS = 150
LOCATIONS = 5
START_DATE = (pd.Timestamp.now().normalize() - pd.Timedelta(days=150)).strftime("%Y-%m-%d")

np.random.seed(42)

location_profiles = {
    1: {"type": "Commercial District", "weekday_multiplier": 1.4, "weekend_multiplier": 0.6, "morning_peak": True, "evening_peak": True},
    2: {"type": "Residential Area", "weekday_multiplier": 0.9, "weekend_multiplier": 1.2, "morning_peak": True, "evening_peak": True},
    3: {"type": "Shopping Mall", "weekday_multiplier": 1.0, "weekend_multiplier": 1.8, "morning_peak": False, "evening_peak": True},
    4: {"type": "Industrial Zone", "weekday_multiplier": 1.3, "weekend_multiplier": 0.3, "morning_peak": True, "evening_peak": False},
    5: {"type": "Entertainment District", "weekday_multiplier": 0.8, "weekend_multiplier": 1.5, "morning_peak": False, "evening_peak": True}
}

timestamps = pd.date_range(
    start=START_DATE,
    periods=DAYS * 24,
    freq="h"
)

data = []

# Random events (accidents, road closures, etc.)
random_events = {}
for _ in range(10):  # 10 random events across the month
    event_day = np.random.randint(0, DAYS)
    event_hour = np.random.randint(0, 24)
    event_location = np.random.randint(1, LOCATIONS + 1)
    event_key = (event_day, event_hour, event_location)
    random_events[event_key] = np.random.choice(["accident", "event", "closure"])


for location in range(1, LOCATIONS + 1):
    profile = location_profiles[location]
    
    for idx, time in enumerate(timestamps):
        hour = time.hour
        day_of_week = time.dayofweek  
        is_weekend = day_of_week >= 5
        day_index = idx // 24
        
        # Base activity with some random variation
        base_activity = np.random.randint(15, 35)
        
        # Apply weekend/weekday multiplier
        if is_weekend:
            base_activity = int(base_activity * profile["weekend_multiplier"])
        else:
            base_activity = int(base_activity * profile["weekday_multiplier"])
        
        activity = base_activity
        
        # Morning peak (7-9 AM)
        if 7 <= hour <= 9 and profile["morning_peak"]:
            if is_weekend:
                activity += np.random.randint(20, 40)
            else:
                activity += np.random.randint(45, 75)
        
        # Lunch hour (12-13)
        elif 12 <= hour <= 13:
            activity += np.random.randint(15, 30)
        
        # Evening peak (17-19)
        elif 17 <= hour <= 19 and profile["evening_peak"]:
            if is_weekend:
                activity += np.random.randint(35, 60)
            else:
                activity += np.random.randint(50, 85)
        
        # Night activity (20-23)
        elif 20 <= hour <= 23:
            if profile["type"] == "Entertainment District":
                activity += np.random.randint(30, 50)
            else:
                activity += np.random.randint(0, 15)
        
        # Late night low traffic (0-5 AM)
        elif 0 <= hour <= 5:
            activity = max(5, activity - np.random.randint(10, 20))
        
        # Random daily variation
        daily_variation = np.random.randint(-10, 15)
        activity += daily_variation
        
        # Check for random events
        event_key = (day_index, hour, location)
        if event_key in random_events:
            event_type = random_events[event_key]
            if event_type == "accident":
                activity = int(activity * 1.5)  # Traffic jam
            elif event_type == "event":
                activity = int(activity * 1.8)  # Special event
            elif event_type == "closure":
                activity = max(5, int(activity * 0.3))  # Road closure
        
        # Gradual trend (slight increase over time)
        trend = int((idx / (DAYS * 24)) * 10)
        activity += trend
        
        # Ensure activity is within reasonable bounds
        activity = max(5, min(activity, 150))
        
        if activity < 40:
            traffic_level = "Low"
        elif activity < 80:
            traffic_level = "Medium"
        else:
            traffic_level = "High"
        
        data.append([
            time,
            location,
            activity,
            traffic_level
        ])

df = pd.DataFrame(
    data,
    columns=[
        "timestamp",
        "location_id",
        "activity_count",
        "traffic_density"
    ]
)

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
output_path = os.path.join(project_root, "data", "raw", "simulated_traffic_data.csv")
df.to_csv(output_path, index=False)

print("✓ Dataset created successfully!")
print("\n--- Location Profiles ---")
for loc_id, profile in location_profiles.items():
    print(f"Location {loc_id}: {profile['type']}")
print(f"\n--- Dataset Summary ---")
print(f"Total records: {len(df)}")
print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
print(f"Locations: {df['location_id'].nunique()}")
print(f"\n--- Traffic Distribution ---")
print(df['traffic_density'].value_counts())
print(f"\n--- Sample Data ---")
print(df.head(10))
