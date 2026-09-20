from flask import Flask, render_template, request, jsonify
from predicter import predict_win_probability

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # The website submits a normal HTML form. JSON is retained so the
        # endpoint can also be used as an API.
        data = request.get_json(silent=True) or request.form

        batting_team = data['batting_team']
        bowling_team = data['bowling_team']
        city = data.get('city') or data['venue']
        target = int(data['target'])
        current_score = int(data['current_score'])
        wickets = int(data['wickets'])
        overs = float(data['overs'])

        probability = predict_win_probability(
            batting_team, bowling_team, city,
            target, current_score, wickets, overs
        )

        losing_team = bowling_team
        losing_prob = round(100 - probability, 2)

        result = {
            "batting_team": batting_team,
            "batting_prob": probability,
            "bowling_team": losing_team,
            "bowling_prob": losing_prob
        }

        if request.is_json:
            return jsonify(result)

        return render_template('result.html', **result)

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    import os
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=False
    )