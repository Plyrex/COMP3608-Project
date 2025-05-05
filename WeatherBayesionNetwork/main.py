import pandas as pd
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import K2Score, HillClimbSearch, BayesianEstimator
from pgmpy.inference import VariableElimination
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import matthews_corrcoef
import numpy as np

data = pd.read_csv("F1 Weather(2023-2018).csv")

# Data Preprocessing
data['RainBinary'] = data['Rainfall'].apply(lambda x: 1 if x > 0 else 0)

# Select relevant features
features = ['AirTemp', 'Humidity', 'Pressure', 'RainBinary', 'TrackTemp', 'WindSpeed']
data = data[features]

# Fill missing values (if any)
data.fillna(method='ffill', inplace=True)
data.fillna(method='bfill', inplace=True)

# Normalize numerical features
for col in features[:-1]:  # Exclude 'RainBinary'
    data[col] = (data[col] - data[col].mean()) / data[col].std()

# Discretize continuous variables for Bayesian Network compatibility
data['AirTemp'] = pd.cut(data['AirTemp'], bins=5, labels=False)
data['Humidity'] = pd.cut(data['Humidity'], bins=5, labels=False)
data['Pressure'] = pd.cut(data['Pressure'], bins=5, labels=False)
data['TrackTemp'] = pd.cut(data['TrackTemp'], bins=5, labels=False)
data['WindSpeed'] = pd.cut(data['WindSpeed'], bins=5, labels=False)

# Split into train-test sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Define Bayesian Network structure
model = BayesianNetwork([
    ('Humidity', 'RainBinary'),
    ('AirTemp', 'RainBinary'),
    ('Pressure', 'RainBinary'),
    ('RainBinary', 'TrackTemp'),
    ('WindSpeed', 'RainBinary')
])

# Train Bayesian Network with BayesianEstimator (smoothing applied)
hc = HillClimbSearch(train_data)
best_model_structure = hc.estimate(scoring_method=K2Score(train_data))
model = BayesianNetwork(best_model_structure.edges())
model.fit(train_data, estimator=BayesianEstimator, prior_type="BDeu", equivalent_sample_size=10)

# Check if the model is valid
try:
    model.check_model()
    print("Model is valid!")
except ValueError as e:
    print(f"Model validation failed: {e}")

# Perform inference
inference = VariableElimination(model)

# Predictions for RainBinary (for MCC evaluation)
predictions = []
for _, row in test_data.iterrows():
    evidence = row.drop('RainBinary').to_dict()
    validated_evidence = {key: int(value) for key, value in evidence.items() if key in model.nodes}
    query_result = inference.map_query(variables=['RainBinary'], evidence=validated_evidence)
    predictions.append(int(query_result['RainBinary']))

# Ensure true_labels are integers
true_labels = test_data['RainBinary'].astype(int)

# K-Fold Cross Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
mcc_scores = []

for train_index, test_index in kf.split(data):
    train_fold = data.iloc[train_index]
    test_fold = data.iloc[test_index]

    hc = HillClimbSearch(train_fold)
    best_model_structure = hc.estimate(scoring_method=K2Score(train_fold))
    model = BayesianNetwork(best_model_structure.edges())
    model.fit(train_fold, estimator=BayesianEstimator, prior_type="BDeu", equivalent_sample_size=10)
    inference = VariableElimination(model)

    fold_predictions = []
    for _, row in test_fold.iterrows():
        evidence = row.drop('RainBinary').to_dict()
        validated_evidence = {key: int(value) for key, value in evidence.items() if key in model.nodes}
        query_result = inference.map_query(variables=['RainBinary'], evidence=validated_evidence)
        fold_predictions.append(int(query_result['RainBinary']))

    fold_mcc = matthews_corrcoef(test_fold['RainBinary'].astype(int), fold_predictions)
    mcc_scores.append(fold_mcc)

# print("Example Evidence:", validated_evidence)

# Evaluate performance using MCC
mcc_score = matthews_corrcoef(true_labels, predictions)
print(f"Matthew Correlation Coefficient (MCC): {mcc_score}")

# Convert MCC results to a DataFrame
mcc_table = pd.DataFrame({"True Labels": true_labels.values, "Predictions": predictions})
print("\n\nMCC Results:")
print(mcc_table)    

# Convert Cross-Validation Results to a DataFrame
cv_table = pd.DataFrame({"Fold": list(range(1, 6)), "MCC Score": mcc_scores})
print("K-Fold Cross Validation Results:")
print(cv_table)

# 1. Probability Distributions
probabilities = inference.query(variables=['RainBinary'], evidence={'Humidity': 3, 'AirTemp': 2})
prob_table = pd.DataFrame({"State": probabilities.state_names['RainBinary'], "Probability": probabilities.values})
print("\n\nProbability Distribution for RainBinary:")
print(prob_table)

# 2. Impact Analysis
impact_analysis = inference.query(variables=['RainBinary'], evidence={'Humidity': 4})
impact_table = pd.DataFrame({"State": impact_analysis.state_names['RainBinary'], "Probability": impact_analysis.values})
print("\n\nImpact Analysis Table:")
print(impact_table)

# 3. Most Likely Explanation (MLE)
mle = inference.map_query(variables=['Humidity', 'AirTemp'], evidence={'RainBinary': 1})
mle_table = pd.DataFrame(list(mle.items()), columns=["Variable", "Most Likely State"])
print("\n\nMost Likely Explanation Table:")
print(mle_table)

# 4. Variable Dependencies
dependencies = model.edges()
dependencies_table = pd.DataFrame(dependencies, columns=["Parent", "Child"])
print("\n\nVariable Dependencies Table:")
print(dependencies_table)

# 5. Scenario Simulation
scenario_result = inference.query(variables=['RainBinary'], evidence={'WindSpeed': 4, 'TrackTemp': 2})
scenario_table = pd.DataFrame({"State": scenario_result.state_names['RainBinary'], "Probability": scenario_result.values})
print("\n\nScenario Simulation Table:")
print(scenario_table)
