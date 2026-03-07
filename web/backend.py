from flask import Flask, render_template, request
import numpy as np
import joblib
import os
import pandas as pd
from datetime import datetime
from tensorflow.keras.models import load_model

app = Flask(__name__)

# ---------------------
# Load model & scaler
# ---------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model = load_model(os.path.join(BASE_DIR, 'results', 'traffic_model.h5'), compile=False)
scaler = joblib.load(os.path.join(BASE_DIR, 'data', 'processed', 'scaler.pkl'))

# Load raw data for building input sequences
raw_df = pd.read_csv(os.path.join(BASE_DIR, 'data', 'raw', 'simulated_traffic_data.csv'))
raw_df['timestamp'] = pd.to_datetime(raw_df['timestamp'])
raw_df = raw_df.sort_values(by=['location_id', 'timestamp'])

PLACE_TO_ID = {
    'Commercial Districts': 1,
    'Residential Area': 2,
    'Shopping Mall': 3,
    'Industrial Zone': 4,
    'Entertainment Districts': 5,
}

LOOKBACK = 24


def build_input_sequence(location_id, hour_24, day_of_week):
    """Build a 24-step input sequence for the LSTM using historical data."""
    loc_df = raw_df[raw_df['location_id'] == location_id].copy()
    loc_df = loc_df.sort_values('timestamp').reset_index(drop=True)
    loc_df['hour'] = loc_df['timestamp'].dt.hour
    loc_df['dow'] = loc_df['timestamp'].dt.dayofweek
    loc_df['is_weekend'] = loc_df['dow'] >= 5

    is_weekend = day_of_week >= 5

    # Find a row matching the target hour and weekday/weekend pattern
    matches = loc_df[(loc_df['hour'] == hour_24) & (loc_df['is_weekend'] == is_weekend)]
    if len(matches) == 0:
        matches = loc_df[loc_df['hour'] == hour_24]
    if len(matches) == 0:
        matches = loc_df

    # Pick the last matching row that has enough history before it
    window = None
    for idx in matches.index[::-1]:
        pos = loc_df.index.get_loc(idx)
        if pos >= LOOKBACK - 1:
            window = loc_df.iloc[pos - LOOKBACK + 1:pos + 1]
            break

    if window is None:
        window = loc_df.iloc[-LOOKBACK:]

    # Build feature array: [activity_count, hour, day_of_week, loc_1..loc_5]
    features = []
    for _, row in window.iterrows():
        h = row['timestamp'].hour
        dow = row['timestamp'].dayofweek
        ac = row['activity_count']

        cont = scaler.transform([[ac, h, dow]])[0]
        one_hot = [1.0 if i == location_id else 0.0 for i in range(1, 6)]
        features.append(list(cont) + one_hot)

    return np.array(features, dtype=np.float32).reshape(1, LOOKBACK, 8)


@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        place = request.form.get("place", "")
        hour_str = request.form.get("hour", "")
        period = request.form.get("period", "")

        # Server-side validation
        if not place or not hour_str or not period:
            return render_template("index.html", error="Please select all fields")

        hour_12 = int(hour_str)

        # Convert 12-hour to 24-hour
        if period == "AM":
            hour_24 = 0 if hour_12 == 12 else hour_12
        else:
            hour_24 = 12 if hour_12 == 12 else hour_12 + 12

        location_id = PLACE_TO_ID.get(place, 1)
        day_of_week = (datetime.now().weekday() + 1) % 7  # Tomorrow's day

        # Build input and predict
        X_input = build_input_sequence(location_id, hour_24, day_of_week)
        pred_normalized = model.predict(X_input, verbose=0)[0][0]

        # Inverse-transform to original activity_count scale
        dummy = np.zeros((1, 3))
        dummy[0, 0] = float(pred_normalized)
        value = int(scaler.inverse_transform(dummy)[0, 0])
        value = max(5, min(value, 150))

        if value < 40:
            prediction = "Low"
        elif value < 80:
            prediction = "Medium"
        else:
            prediction = "High"

        return render_template("result.html",
                               prediction=prediction,
                               value=value)

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)