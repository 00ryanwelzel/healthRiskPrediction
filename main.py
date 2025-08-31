import os
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

CSV_PATH = 'Health_Risk_Dataset.csv'
TARGET_COLUMN = 'Risk_Level'
IS_CLASSIFICATION = True
VITAL_FEATURES = ['Respiratory_Rate', 'Oxygen_Saturation', 'Heart_Rate', 'Temperature']

def load_data():
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV file not found at: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    #print(f"Loaded {CSV_PATH} with shape {df.shape}")
    return df

def train_model(df):
    df = df.dropna(subset=[TARGET_COLUMN] + VITAL_FEATURES)

    # Filter junk
    df = df[VITAL_FEATURES + [TARGET_COLUMN]]

    # Skew results towards high risk
    high_rows = df[df[TARGET_COLUMN] == 'HIGH']
    df = pd.concat([df, high_rows, high_rows], ignore_index=True)

    x = df[VITAL_FEATURES]
    y = df[TARGET_COLUMN]

    # Str to numeric
    if y.dtype == 'object':
        y = LabelEncoder().fit_transform(y)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Tested models
    '''
    model = LogisticRegression(class_weight='balanced', max_iter=5000)
    '''

    model = RandomForestClassifier()
    model.fit(x_train, y_train)

    # For stats / debugging
    '''
    y_pred = model.predict(x_test)
    if IS_CLASSIFICATION:
        acc = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {acc:.4f}")
    else:
        print("Error printing accuracy.")
        return
    '''

    return model

def plot_feature_importance(model, feature_names):
    # Sort vitals score
    importances = model.feature_importances_
    sorted_idx = importances.argsort()[::-1]
    sorted_importances = importances[sorted_idx]

    # 0.0 -> 1.0
    norm = (sorted_importances - np.min(sorted_importances)) / (np.max(sorted_importances) - np.min(sorted_importances))

    # Calculate bar colors based on vitals score
    cmap = plt.colormaps['coolwarm']
    colors = [cmap(val) for val in norm]

    # Chart
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(range(len(importances)), sorted_importances, align='center', color=colors)
    ax.set_xticks(range(len(importances)))
    ax.set_xticklabels([feature_names[i] for i in sorted_idx], rotation=45)
    ax.set_ylabel("Contribution to Overall Risk")
    ax.set_title("Feature Contribution to Risk Prediction")
    ax.set_ylim(0, max(sorted_importances) * 1.2)

    # Annotate bars with values
    for i, bar in enumerate(bars):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{sorted_importances[i]:.3f}",
            ha='center',
            va='bottom',
            fontsize=9
        )

    # Color scale sidebar
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=mpl.colors.Normalize(vmin=np.min(sorted_importances), vmax=np.max(sorted_importances)))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label('Feature Importance Value')

    plt.tight_layout()

    output_file = "feature_importance.png"
    plt.savefig(output_file)
    print("Breakdown of high risk vitals saved to", os.path.abspath(output_file))

    plt.close(fig)

def predict_user_input(model):
    print("\nEnter patient vitals:")

    # Widest acceptable vitals margins
    # If vitals are outside these bounds your guy is dead
    bounds = {
        'Respiratory_Rate': (10, 50),
        'Oxygen_Saturation': (70, 100),
        'Heart_Rate': (30, 250),
        'Temperature': (25.0, 45.0)
    }

    try:
        respiratory_rate = float(input("Respiratory Rate (10-50): "))
        if not (bounds['Respiratory_Rate'][0] <= respiratory_rate <= bounds['Respiratory_Rate'][1]):
            raise ValueError("Respiratory Rate out of bounds.")

        oxygen_saturation = float(input("Oxygen Saturation (70–100): "))
        if not (bounds['Oxygen_Saturation'][0] <= oxygen_saturation <= bounds['Oxygen_Saturation'][1]):
            raise ValueError("Oxygen Saturation out of bounds.")

        heart_rate = float(input("Heart Rate (30–250): "))
        if not (bounds['Heart_Rate'][0] <= heart_rate <= bounds['Heart_Rate'][1]):
            raise ValueError("Heart Rate out of bounds.")

        temperature = float(input("Temperature (25.0–45.0 °C): "))
        if not (bounds['Temperature'][0] <= temperature <= bounds['Temperature'][1]):
            raise ValueError("Temperature out of bounds.")

    except ValueError as e:
        print(f"Invalid input: {e}")
        return


    input_data = pd.DataFrame([{
        'Respiratory_Rate': respiratory_rate,
        'Oxygen_Saturation': oxygen_saturation,
        'Heart_Rate': heart_rate,
        'Temperature': temperature
    }])

    prediction = model.predict(input_data)[0]

    # Classify model prediction so it has interpretability
    if IS_CLASSIFICATION:
        if prediction == 0:
            prediction_classification = "LOW"
        elif prediction == 1:
            prediction_classification = "MEDIUM"
        else:
            prediction_classification = "HIGH"

        print("Predicted Risk Category: {}".format(prediction_classification))
        plot_feature_importance(model, VITAL_FEATURES)
    else:
        print("Error predicting Risk Category.")

def main():
    df = load_data()
    model = train_model(df)
    #print("Model training complete.")

    while True:
        predict_user_input(model)
        again = input("\nWould you like to enter another case? (y/n): ").strip().lower()
        if again != 'y':
            break

if __name__ == '__main__':
    main()