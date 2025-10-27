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
from sklearn.metrics import make_scorer, roc_auc_score, f1_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt



data = pd.read_csv("Bootcamp-Data-Rush-4/csv/Camp_Market.csv", sep=';')

#Data Cleaning
data = data[~data['Marital_Status'].isin(['YOLO', 'Absurd'])]
data = data.replace({'Marital_Status': {'Alone': 'Single'}})

data = data[~data['Year_Birth'].isin([1893, 1899, 1900])]

median = data['Income'].median()
data['Income'] = data['Income'].fillna(median)

data['Age'] = 2014 - data['Year_Birth']
data['Is_Parent'] = ((data['Kidhome'] + data['Teenhome']) > 0).astype(int)

totalSpent = ['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']
data['Total_Spent'] = data[totalSpent].sum(axis=1)

data['Dt_Customer'] = pd.to_datetime(data['Dt_Customer'])
data['Customer_Seniority'] = (pd.Timestamp('2014-12-31') - data['Dt_Customer']).dt.days
data = data.drop(columns=['Dt_Customer'])

bins = [0, 25, 35, 50, 65, 100]
groups = ['18-25', '26-35', '36-50', '51-65', '65+']
data['Age_Group'] = pd.cut(data['Age'], bins=bins, labels=groups, right=False)

accepted_cols = ['AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5']
data['Total_Accepted_Campaigns'] = data[accepted_cols].sum(axis=1)  

data['Has_Accepted_Before'] = (data['Total_Accepted_Campaigns'] > 0).astype(int)


X = data.drop(columns=["Response"])
y = data["Response"]
X = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_bal)
X_test_scaled = scaler.transform(X_test)


rf = RandomForestClassifier(random_state=42, n_jobs=-1)



param_grid = {
    'n_estimators': [200, 400],          # keep 2 levels of complexity
    'max_depth': [6, 8, 10, None],       # test shallow vs full
    'min_samples_split': [2, 5],         # fine control on splits
    'min_samples_leaf': [1, 5, 10],      # controls smoothness
    'class_weight': [None, 'balanced', {0:1, 1:1.5}]  # 3 variants
}

grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    scoring='roc_auc',   # works natively with RandomForest
    cv=3,                # 3 folds instead of 5 = faster
    verbose=2,
    n_jobs=-1
)


# grid_search.fit(X_train_scaled, y_train_bal)
# print("🔥 Best parameters:", grid_search.best_params_)
# print("💎 Best AUC:", grid_search.best_score_)
forest = RandomForestClassifier(
    n_estimators=500,
    max_depth=25,        
    min_samples_leaf=35,    
    min_samples_split=2,
    class_weight={0:1, 1:1.4},  
    random_state=42,
    n_jobs=-1
)

forest.fit(X_train_scaled, y_train_bal)

y_pred = forest.predict(X_test_scaled)
y_pred_proba = forest.predict_proba(X_test_scaled)[:, 1]
threshold = 0.4 
y_pred_adjusted = (y_pred_proba >= threshold).astype(int)

auc = roc_auc_score(y_test, y_pred_adjusted)
print(f"AUC Score: {auc:.3f}")

print("Accuracy:", round(accuracy_score(y_test, y_pred_adjusted), 3))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_adjusted))
print("\nClassification Report:\n", classification_report(y_test, y_pred_adjusted))

# feature_importances = pd.DataFrame({
#     'Feature': X_train_bal.columns,
#     'Importance': forest.feature_importances_
# }).sort_values(by='Importance', ascending=False)

# print(feature_importances.head(15))

print("Before SMOTE:", y_train.value_counts())
print("After SMOTE:", y_train_bal.value_counts())


#TEST CLIENTS

import pandas as pd

# Create 5 clients with real dataset columns
new_clients = pd.DataFrame([
    {"Year_Birth": 1970, "Education": "PhD", "Marital_Status": "Married", "Income": 110000,
     "Kidhome": 0, "Teenhome": 0, "Recency": 2,
     "MntWines": 1200, "MntFruits": 100, "MntMeatProducts": 1500,
     "MntFishProducts": 300, "MntSweetProducts": 250, "MntGoldProds": 400,
     "Total_Accepted_Campaigns": 4},

    {"Year_Birth": 2000, "Education": "Basic", "Marital_Status": "Single", "Income": 10000,
     "Kidhome": 0, "Teenhome": 0, "Recency": 95,
     "MntWines": 0, "MntFruits": 0, "MntMeatProducts": 0,
     "MntFishProducts": 0, "MntSweetProducts": 0, "MntGoldProds": 0,
     "Total_Accepted_Campaigns": 0}
])

new_clients['Age'] = 2015 - new_clients['Year_Birth']
new_clients['Total_Spent'] = new_clients[['MntWines','MntFruits','MntMeatProducts',
                                          'MntFishProducts','MntSweetProducts','MntGoldProds']].sum(axis=1)

new_clients = pd.get_dummies(new_clients, drop_first=True)
new_clients = new_clients.reindex(columns=X_train_bal.columns, fill_value=0)

new_clients_scaled = scaler.transform(new_clients)

predictions = forest.predict(new_clients_scaled)
probabilities = forest.predict_proba(new_clients_scaled)[:, 1]

results = pd.DataFrame({
    "Client": range(1, 3),
    "Predicted_Response": predictions,
    "Response_Probability": probabilities
})
print(results)


