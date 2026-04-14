# UrbanFlow AI - Traffic Density Prediction

## Overview
UrbanFlow AI is an intelligent prediction system designed to forecast traffic density across various urban zones (such as Commercial Districts, Residential Areas, Shopping Malls, Industrial Zones, and Entertainment Districts). By leveraging an LSTM (Long Short-Term Memory) deep learning model trained on simulated mobile activity and temporal data, it provides accurate, time-based traffic density predictions (Low, Medium, High). 

The project includes a complete pipeline from realistic data simulation and preprocessing to deep learning model training and a fully functioning Flask web application.

## Key Features
- **Data Simulation Engine**: Generates realistic time-series mobility data reflecting weekday/weekend patterns, morning/evening peak hours, and random events (accidents, road closures).
- **Deep Learning Prediction Model**: Utilizes a robust LSTM neural network via TensorFlow/Keras tailored to analyze sequences of activity and predict future traffic loads.
- **Interactive Web Interface**: A Flask application where users can input a location, date, and time to receive real-time traffic density insights to optimize travel.

## Tech Stack
- **Languages**: Python, HTML, CSS
- **Machine Learning & Data Processing**: TensorFlow (Keras), Scikit-Learn, Pandas, NumPy
- **Web Framework**: Flask
- **Tools**: Joblib (for scaler persistence), Matplotlib (for training visualization)

## Project Structure
```text
├── data/
│   ├── raw/             # Generated simulated traffic CSVs
│   └── processed/       # Scaled datasets, numpy arrays, and scaler.pkl
├── src/
│   ├── data_simulation.py  # Script for generating base simulated dataset
│   ├── preprocessing.py    # Data cleaning and sequence generation
│   └── model.py            # LSTM model definition, training, and evaluation
├── web/
│   ├── backend.py       # Flask backend application serving predictions
│   ├── static/          # Web app CSS and static assets
│   └── templates/       # HTML view templates
├── results/             # Trained models (.h5), evaluation metrics, and plots
├── requirements.txt     # Python dependency list
└── README.md
```

## Running the Project Locally

### 1. Installation
Clone the repository and install the required dependencies:
```bash
pip install -r requirements.txt
```

### 2. Data Generation & Training (Optional)
If you wish to re-generate the data and re-train the model from scratch:
```bash
# Generate the raw data
python src/data_simulation.py

# Preprocess the data and build sequences
python src/preprocessing.py

# Train the LSTM model
python src/model.py
```

### 3. Launching the Web Application
Start the Flask server to interact with the predictions:
```bash
python web/backend.py
```
Open your browser and navigate to `http://127.0.0.1:5000` to interact with the UI.