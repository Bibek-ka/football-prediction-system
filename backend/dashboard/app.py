from flask import Flask, jsonify, render_template
import pandas as pd
from datetime import datetime
import os

app = Flask(__name__)

def load_data():
    csv_file = 'combined_output.csv'
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
        return df.to_dict(orient='records')
    else:
        # Default sample data
        return [
            {"player": "Max Aarons", "nation": "ENG", "position": "DF", "team": "Bournemouth",
             "age_x": 25, "goals_x": 0, "assists_x": 0, "minutes": 86,
             "expected_goals_x": 0, "successful_take_ons_x": 0, "progressive_passes_x": 8},
            {"player": "Tyler Adams", "nation": "USA", "position": "MF", "team": "Bournemouth",
             "age_x": 26, "goals_x": 0, "assists_x": 3, "minutes": 1875,
             "expected_goals_x": 1.6, "successful_take_ons_x": 3, "progressive_passes_x": 71},
            {"player": "Simon Adingra", "nation": "CIV", "position": "FW,MF", "team": "Brighton",
             "age_x": 23, "goals_x": 2, "assists_x": 2, "minutes": 1052,
             "expected_goals_x": 2.4, "successful_take_ons_x": 21, "progressive_passes_x": 18}
            # Add more data rows as needed
        ]

@app.route('/')
def dashboard():
    return render_template('dashboard.html')

@app.route('/api/players')
def api_players():
    players = load_data()
    return jsonify({
        'last_update': datetime.utcnow().isoformat(),
        'players': players
    })

if __name__ == '__main__':
    app.run(debug=True)
