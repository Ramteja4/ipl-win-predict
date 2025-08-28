import pandas as pd

# Load matches and deliveries data
matches = pd.read_csv('data/matches.csv')
deliveries = pd.read_csv('data/deliveries.csv')

# Filter out matches with no result or tie
matches = matches.dropna(subset=['winner'])
matches = matches[matches['result'] != 'no result']

training_data = []

# Iterate over each match
for idx, match in matches.iterrows():
    match_id = match['id']
    # Get deliveries for this match
    match_deliveries = deliveries[deliveries['match_id'] == match_id]
    
    # Assume first innings is played by team1 and second innings by team2
    first_innings_team = match['team1']
    second_innings_team = match['team2']
    
    # Get first innings deliveries (inning == 1)
    first_innings = match_deliveries[match_deliveries['inning'] == 1]
    if first_innings.empty:
        continue
    # Calculate first innings total score
    first_innings_total = first_innings['total_runs'].sum()
    target = first_innings_total + 1  # runs required to win
    
    # Get second innings deliveries (inning == 2)
    second_innings = match_deliveries[match_deliveries['inning'] == 2]
    if second_innings.empty:
        continue
    
    # Check if at least 15 overs have been played (15 overs = 90 balls)
    second_innings = second_innings.sort_values(by=['over', 'ball'])
    if len(second_innings) < 90:
        continue
    
    # Snapshot at 15 overs: filter deliveries where over <= 15
    snapshot = second_innings[second_innings['over'] <= 15]
    
    # Compute current score and wickets lost at 15 overs
    current_score = snapshot['total_runs'].sum()
    wickets = snapshot['player_dismissed'].notnull().sum()
    
    # Calculate additional features
    overs = 15
    run_rate = current_score / overs
    runs_remaining = target - current_score
    overs_remaining = 20 - overs
    req_run_rate = runs_remaining / overs_remaining if overs_remaining > 0 else runs_remaining
    
    # Determine outcome: 1 if batting (chasing) team won, else 0
    result = 1 if match['winner'] == second_innings['batting_team'].iloc[0] else 0
    
    training_data.append({
        'batting_team': second_innings['batting_team'].iloc[0],
        'bowling_team': second_innings['bowling_team'].iloc[0],
        'venue': match['venue'],
        'target': target,
        'current_score': current_score,
        'wickets': wickets,
        'overs': overs,
        'run_rate': run_rate,
        'req_run_rate': req_run_rate,
        'result': result
    })

df = pd.DataFrame(training_data)
df.to_csv('data/training_data.csv', index=False)
print("Training data created and saved to data/training_data.csv")
