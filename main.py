import pandas as pd
import numpy as np

from datetime import date

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

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


print("Before SMOTE:", y_train.value_counts())
print("After SMOTE:", y_train_bal.value_counts())






