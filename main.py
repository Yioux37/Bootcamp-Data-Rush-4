import pandas as pd
import numpy as np

from datetime import date

from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt



data = pd.read_csv("csv/Camp_Market.csv", sep=';')

#Data Cleaning
data = data[~data['Marital_Status'].isin(['YOLO', 'Absurd'])]
data = data.replace({'Marital_Status': {'Alone': 'Single'}})

data = data[~data['Year_Birth'].isin([1893, 1899, 1900])]

median = data['Income'].median()
data['Income'] = data['Income'].fillna(median)

data['Age'] = 2015 - data['Year_Birth']
data['Is_Parent'] = ((data['Kidhome'] + data['Teenhome']) > 0).astype(int)

totalSpent = ['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']
data['Total_Spent'] = data[totalSpent].sum(axis=1)

data['Dt_Customer'] = pd.to_datetime(data['Dt_Customer'])
data['Customer_Seniority'] = (pd.Timestamp.today() - data['Dt_Customer']).dt.days
data = data.drop(columns=['Dt_Customer'])

bins = [0, 25, 35, 50, 65, 100]
groups = ['18-25', '26-35', '36-50', '51-65', '65+']
data['Age_Group'] = pd.cut(data['Age'], bins=bins, labels=groups, right=False)



X = data.drop(columns=["Response"])
y = data["Response"]
X = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_bal)
X_test_scaled = scaler.transform(X_test)

# Create the model
# forest = RandomForestClassifier(
#     n_estimators=200,      # number of trees (you can tune)
#     max_depth=10,          # prevent overfitting
#     random_state=42,
#     n_jobs=-1              # use all CPU cores for speed
# )

forest = RandomForestClassifier(
    n_estimators=500,
    max_depth=None,
    min_samples_leaf=10,
    class_weight={0:1, 1:1.5},
    random_state=42,
    n_jobs=-1
)

forest.fit(X_train_scaled, y_train_bal)

y_pred = forest.predict(X_test_scaled)
y_pred_proba = forest.predict_proba(X_test_scaled)[:, 1]
threshold = 0.35  # or whatever value you want
y_pred_adjusted = (y_pred_proba >= threshold).astype(int)

auc = roc_auc_score(y_test, y_pred_adjusted)
print(f"💖 AUC Score: {auc:.3f}")

print("🌲 Random Forest Results 🌲")
print("Accuracy:", round(accuracy_score(y_test, y_pred_adjusted), 3))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_adjusted))
print("\nClassification Report:\n", classification_report(y_test, y_pred_adjusted))


print("Before SMOTE:", y_train.value_counts())
print("After SMOTE:", y_train_bal.value_counts())


#TEST CLIENTS

import pandas as pd

# Create 5 clients with real dataset columns
new_clients = pd.DataFrame([
    {"Year_Birth": 1988, "Education": "Graduation", "Marital_Status": "Single", "Income": 34000,
     "Kidhome": 0, "Teenhome": 0, "Recency": 12, "MntWines": 50, "MntFruits": 5, "MntMeatProducts": 100,
     "MntFishProducts": 15, "MntSweetProducts": 10, "MntGoldProds": 20},
    
    {"Year_Birth": 1975, "Education": "Master", "Marital_Status": "Married", "Income": 72000,
     "Kidhome": 1, "Teenhome": 1, "Recency": 38, "MntWines": 400, "MntFruits": 20, "MntMeatProducts": 900,
     "MntFishProducts": 70, "MntSweetProducts": 60, "MntGoldProds": 180},
    
    {"Year_Birth": 1968, "Education": "PhD", "Marital_Status": "Together", "Income": 95000,
     "Kidhome": 0, "Teenhome": 0, "Recency": 25, "MntWines": 700, "MntFruits": 60, "MntMeatProducts": 1400,
     "MntFishProducts": 90, "MntSweetProducts": 80, "MntGoldProds": 250},
    
    {"Year_Birth": 1993, "Education": "2n Cycle", "Marital_Status": "Single", "Income": 26000,
     "Kidhome": 0, "Teenhome": 1, "Recency": 8, "MntWines": 20, "MntFruits": 2, "MntMeatProducts": 60,
     "MntFishProducts": 5, "MntSweetProducts": 3, "MntGoldProds": 10},
    
    {"Year_Birth": 1982, "Education": "Graduation", "Marital_Status": "Married", "Income": 52000,
     "Kidhome": 2, "Teenhome": 1, "Recency": 60, "MntWines": 200, "MntFruits": 25, "MntMeatProducts": 450,
     "MntFishProducts": 35, "MntSweetProducts": 25, "MntGoldProds": 70}
])

new_clients['Age'] = 2015 - new_clients['Year_Birth']
new_clients['Total_Spent'] = new_clients[['MntWines','MntFruits','MntMeatProducts',
                                          'MntFishProducts','MntSweetProducts','MntGoldProds']].sum(axis=1)

new_clients = pd.get_dummies(new_clients, drop_first=True)
new_clients = new_clients.reindex(columns=X_train_bal.columns, fill_value=0)

new_clients_scaled = scaler.transform(new_clients)

calibrated_forest = CalibratedClassifierCV(forest, method='isotonic', cv=5)
calibrated_forest.fit(X_train_scaled, y_train_bal)

predictions = forest.predict(new_clients_scaled)
probabilities = forest.predict_proba(new_clients_scaled)[:, 1]

results = pd.DataFrame({
    "Client": [1, 2, 3, 4, 5],
    "Predicted_Response": predictions,
    "Response_Probability": probabilities
})
print(results)


