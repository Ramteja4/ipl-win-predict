import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import pickle

# Load the training data
data = pd.read_csv('data/training_data.csv')

# Check available columns
print("Columns in dataset:", data.columns)

# Define features and target variable with correct column names
features = ['batting_team', 'bowling_team', 'venue', 'target', 'current_score', 'wickets', 'overs']
X = data[features]
y = data['result']

# Set up preprocessing for categorical and numerical features
categorical_features = ['batting_team', 'bowling_team', 'venue']
numerical_features = ['target', 'current_score', 'wickets', 'overs']

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ('num', StandardScaler(), numerical_features)
    ]
)

# Create a pipeline that first preprocesses the data then applies logistic regression
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000))
])

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
pipeline.fit(X_train, y_train)

# Evaluate the model
score = pipeline.score(X_test, y_test)
print("Model accuracy on test set: {:.2f}%".format(score * 100))

# Save the model to disk
with open('win_prob_model.pkl', 'wb') as f:
    pickle.dump(pipeline, f)

print("Model saved as win_prob_model.pkl")
