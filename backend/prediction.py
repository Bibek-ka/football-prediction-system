from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)

# Load your trained model
model = joblib.load('your_saved_model.pkl')

# Maps to convert strings to integers for categorical features
venue_map = {'home': 0, 'away': 1, 'neutral': 2}
weather_map = {'sunny': 0, 'rainy': 1, 'cloudy': 2, 'snowy': 3}
league_map = {'premier_league': 0, 'la_liga': 1, 'serie_a': 2, 'bundesliga': 3, 'ligue_1': 4}

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    try:
        # Extract and preprocess inputs
        home_strength = float(data.get('homeStrength', 0))
        away_strength = float(data.get('awayStrength', 0))

        venue_str = data.get('venue', '').lower()
        weather_str = data.get('weather', '').lower()
        league_str = data.get('league', '').lower()

        # Convert categorical to numeric (default to -1 if unknown)
        venue_encoded = venue_map.get(venue_str, -1)
        weather_encoded = weather_map.get(weather_str, -1)
        league_encoded = league_map.get(league_str, -1)

        # Check if any encoding failed (unknown category)
        if -1 in (venue_encoded, weather_encoded, league_encoded):
            return jsonify({'error': 'Invalid category in input'}), 400

        # Construct feature array in the order your model expects
        features = np.array([[home_strength, away_strength, venue_encoded, weather_encoded, league_encoded]])

        # Predict (assuming model output is class label or int)
        prediction_raw = model.predict(features)[0]

        # If you want to map prediction back to meaningful label (optional)
        # For example, if model predicts 0=Home Win,1=Draw,2=Away Win:
        label_map = {0: 'Home Win', 1: 'Draw', 2: 'Away Win'}
        prediction = label_map.get(prediction_raw, str(prediction_raw))

        return jsonify({'prediction': prediction})

    except Exception as e:
        return jsonify({'error': f'Error during prediction: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)
