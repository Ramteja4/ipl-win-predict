import pickle
import pandas as pd

with open('pipe.pkl', 'rb') as f:
    pipeline = pickle.load(f)

def predict_win_probability(batting_team,bowling_team, city, target, current_score, wickets, overs):
    """
    Predict the win probability for the batting team given the match state.
    """

    crr = current_score / overs if overs > 0 else 0
    runs_left = target - current_score
    overs_remaining = 20 - overs
    rrr = runs_left / overs_remaining if overs_remaining > 0 else runs_left
    balls_left=overs_remaining*6
    wickets_left=10 -wickets

    input_df = pd.DataFrame({
        'batting_team': [batting_team],
        'bowling_team': [bowling_team],
        'city': [city],
        'runs_left': [runs_left],
        'balls_left': [balls_left],
        'wickets_left': [wickets_left],
        'total_runs_x': [target],
        'crr': [crr],
        'rrr': [rrr]
    })


    probability = pipeline.predict_proba(input_df)[0][1] * 100

    return round(probability, 2)
