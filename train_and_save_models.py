import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier

# Load datasets
dataset1 = pd.read_csv("Dataset_1.csv")  # Medical
dataset2 = pd.read_csv("Dataset_2.csv")  # Lifestyle

X1 = dataset1.drop(columns=["target"])
y1 = dataset1["target"]
X2 = dataset2.drop(columns=["Heart_Risk"])
y2 = dataset2["Heart_Risk"]

X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.2, random_state=42)
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=42)

model1 = GaussianNB()
model1.fit(X1_train, y1_train)

model2 = GradientBoostingClassifier()
model2.fit(X2_train, y2_train)

# Save models and feature names
joblib.dump(model1, "model1_nb.pkl")
joblib.dump(model2, "model2_gb.pkl")
joblib.dump(X1.columns.tolist(), "features1.pkl")
joblib.dump(X2.columns.tolist(), "features2.pkl")
