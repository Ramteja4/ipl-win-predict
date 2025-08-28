import pickle
import pandas as pd

# Load the pipeline (preprocessor + classifier)
with open('win_prob_model.pkl', 'rb') as f:
    pipeline = pickle.load(f)

def predict_win_probability(batting_team, bowling_team, venue, target, current_score, wickets, overs):
    """
    Predict the win probability for the batting team given the match state.
    """

    # Calculate any extra features if needed
    run_rate = current_score / overs if overs > 0 else 0
    runs_remaining = target - current_score
    overs_remaining = 20 - overs
    req_run_rate = runs_remaining / overs_remaining if overs_remaining > 0 else runs_remaining

    # Create a DataFrame matching the training columns
    input_df = pd.DataFrame({
        'batting_team': [batting_team],
        'bowling_team': [bowling_team],
        'venue': [venue],
        'target': [target],
        'current_score': [current_score],
        'wickets': [wickets],
        'overs': [overs],
        'run_rate': [run_rate],
        'req_run_rate': [req_run_rate]
    })

    # Get predicted probability (class=1: batting team wins)
    probability = pipeline.predict_proba(input_df)[0][1] * 100

    return round(probability, 2)  # e.g., 53.91
