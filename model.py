import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score



# Load datasets
dataset1 = pd.read_csv("Dataset_1.csv")  # Medical Report-Based
dataset2 = pd.read_csv("Dataset_2.csv")  # Lifestyle & Symptoms-Based

# Keep all columns
X1 = dataset1.drop(columns=["target"])
y1 = dataset1["target"]
X2 = dataset2.drop(columns=["Heart_Risk"])
y2 = dataset2["Heart_Risk"]

# Split datasets
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.2, random_state=42)
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=42)

# Train models
model1 = GaussianNB()
model1.fit(X1_train, y1_train)

model2 = GradientBoostingClassifier()
model2.fit(X2_train, y2_train)

# Evaluate models
accuracy1 = accuracy_score(y1_test, model1.predict(X1_test))
accuracy2 = accuracy_score(y2_test, model2.predict(X2_test))

print(f"Medical Dataset Model (Naïve Bayes) Accuracy: {accuracy1:.2%}")
print(f"Lifestyle Dataset Model (Gradient Boosting) Accuracy: {accuracy2:.2%}")

# Prediction function
def predict_cardiac_disease(input_values):
    medical_input_len = len(X1.columns)
    lifestyle_input_len = len(X2.columns)

    if len(input_values) != medical_input_len + lifestyle_input_len:
        raise ValueError(f"Expected {medical_input_len + lifestyle_input_len} inputs, but got {len(input_values)}.")

    medical_input = pd.DataFrame([input_values[:medical_input_len]], columns=X1.columns)
    lifestyle_input = pd.DataFrame([input_values[medical_input_len:]], columns=X2.columns)

    prob1 = model1.predict_proba(medical_input)[0][1]
    prob2 = model2.predict_proba(lifestyle_input)[0][1]

    # Apply rule-based decision without printing probabilities
    if prob1 >= 0.95:
        final_prediction = 1
    elif prob1 >= 0.8 and prob2 >= 0.6:
        final_prediction = 1
    elif prob1 >= 0.7 and prob2 >= 0.8:
        final_prediction = 1
    elif prob2 >= 0.95:
        final_prediction = 1
    else:
        final_prediction = 0

    return {
        "Final Prediction": "Heart Disease Detected" if final_prediction == 1 else "No Heart Disease"
    }

# Example test input (dummy values of correct length)
test_input = [0.5] * (X1.shape[1] + X2.shape[1])
print(predict_cardiac_disease(test_input))
