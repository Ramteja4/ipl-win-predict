from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load your trained pipeline
with open('win_prob_model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from form
    batting_team = request.form.get('batting_team')
    bowling_team = request.form.get('bowling_team')
    venue = request.form.get('venue')
    target_score = float(request.form.get('target_score'))
    current_score = float(request.form.get('current_score'))
    wickets_lost = int(request.form.get('wickets_lost'))
    overs_completed = float(request.form.get('overs_completed'))
    
    # Create a DataFrame to match the training columns
    input_df = pd.DataFrame({
        'batting_team': [batting_team],
        'bowling_team': [bowling_team],
        'venue': [venue],
        'target': [target_score],
        'current_score': [current_score],
        'wickets': [wickets_lost],
        'overs': [overs_completed]
    })
    
    # Predict probability of the batting team winning (class=1)
    probability = model.predict_proba(input_df)[0][1] * 100
    probability = round(probability, 2)

    # Render the result page with the probability bar
    return render_template(
        'result.html', 
        probability=probability,
        batting_team=batting_team,
        bowling_team=bowling_team
    )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
