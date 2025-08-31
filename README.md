# healthRiskPrediction
## (Ryan Welzel 8/30/2025)

A machine learning pipeline that classifies patient health risk based on vital signs using Python and scikit-learn.

--- 

Features:
- ML model training pipeline: Loads a local vitals dataset (CSV), trains a classification model (Random Forest), and evaluates performance
- User input: Accepts input of vital signs (heart rate, respiratory rate, oxygen saturation, and temperature) for risk prediction
- Risk category classification: Predicts LOW, MEDIUM, or HIGH risk level based on trained model
- Input validation: Enforces realistic physiological bounds to reject extreme or invalid vitals
- Visual explanation: Uses matplotlib to generate a bar chart showing each feature's contribution to the overall prediction, with a color gradient and labeled importance values
- Flexible model options: Easily swap between RandomForestClassifier, LogisticRegression, or other classifiers

Requirements:
- Python 3.8+

Dependencies:
- pandas
- scikit-learn
- matplotlib
