import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

data = pd.read_csv("csv/Camp_Market.csv", sep=';')

#Data Cleaning
data = data[~data['Marital_Status'].isin(['YOLO', 'Absurd'])]
data = data[~data['Year_Birth'].isin([1893, 1899, 1900])]

median_income = data['Income'].median()
data['Income'] = data['Income'].fillna(median_income)


print(data['Marital_Status'].unique())
print(sorted(data['Year_Birth'].unique())[:10])  
print(data['Income'].isna().sum())  



