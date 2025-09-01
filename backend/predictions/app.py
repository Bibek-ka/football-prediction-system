from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Allow CORS so your frontend can call this API

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # Extract features from input
    home_strength = float(data.get('homeStrength', 0))
    away_strength = float(data.get('awayStrength', 0))
    venue = data.get('venue', '').lower()
    weather = data.get('weather', '').lower()
    league = data.get('league', '').lower()
    if home_strength > away_strength + 1:
        prediction = 'Home Win'
    elif away_strength > home_strength + 1:
        prediction = 'Away Win'
    else:
        prediction = 'Draw'

    # Return the prediction
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)
