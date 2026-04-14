from flask import Flask, render_template, request
import numpy as np
import joblib
import os
import pandas as pd
from datetime import datetime, timedelta
from tensorflow.keras.models import load_model

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model = load_model(os.path.join(BASE_DIR, 'results', 'traffic_model.h5'), compile=False)
scaler = joblib.load(os.path.join(BASE_DIR, 'data', 'processed', 'scaler.pkl'))

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
    loc_df = raw_df[raw_df['location_id'] == location_id].copy()
    loc_df = loc_df.sort_values('timestamp').reset_index(drop=True)
    loc_df['hour'] = loc_df['timestamp'].dt.hour
    loc_df['dow'] = loc_df['timestamp'].dt.dayofweek
    loc_df['is_weekend'] = loc_df['dow'] >= 5

    is_weekend = day_of_week >= 5

    matches = loc_df[(loc_df['hour'] == hour_24) & (loc_df['is_weekend'] == is_weekend)]
    if len(matches) == 0:
        matches = loc_df[loc_df['hour'] == hour_24]
    if len(matches) == 0:
        matches = loc_df

    window = None
    for idx in matches.index[::-1]:
        pos = loc_df.index.get_loc(idx)
        if pos >= LOOKBACK - 1:
            window = loc_df.iloc[pos - LOOKBACK + 1:pos + 1]
            break

    if window is None:
        window = loc_df.iloc[-LOOKBACK:]

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
        date_str = request.form.get("date", "")

        if not place or not hour_str or not period or not date_str:
            return render_template("index.html", error="Please select all fields")

        try:
            hour_12 = int(hour_str)
        except ValueError:
            return render_template("index.html", error="Invalid hour")

        try:
            selected_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        except ValueError:
            return render_template("index.html", error="Invalid date")

        today = datetime.now().date()
        min_date = today
        max_date = today + timedelta(days=7)
        if selected_date < min_date or selected_date > max_date:
            return render_template("index.html", error="Date must be within today and the next 7 days")

        if period == "AM":
            hour_24 = 0 if hour_12 == 12 else hour_12
        else:
            hour_24 = 12 if hour_12 == 12 else hour_12 + 12

        location_id = PLACE_TO_ID.get(place, 1)
        day_of_week = selected_date.weekday()

        X_input = build_input_sequence(location_id, hour_24, day_of_week)
        pred_normalized = model.predict(X_input, verbose=0)[0][0]

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
                               value=value,
                               place=place,
                               hour=hour_12,
                               period=period,
                               selected_date=selected_date.strftime("%Y-%m-%d"))

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)