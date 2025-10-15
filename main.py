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
import matplotlib.pyplot as plt
import seaborn as sns

output_folder = "Bootcamp-Data-Rush-4/Graphs"
data = pd.read_csv("Bootcamp-Data-Rush-4/csv/Camp_Market.csv", sep=';')

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
data['Customer_Seniority'] = (pd.Timestamp('2014-12-31') - data['Dt_Customer']).dt.days
data = data.drop(columns=['Dt_Customer'])

bins = [0, 25, 35, 50, 65, 100]
groups = ['18-25', '26-35', '36-50', '51-65', '65+']
data['Age_Group'] = pd.cut(data['Age'], bins=bins, labels=groups, right=False)


campaign_cols = ['AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5']

# Acceptance rate per campaign
for cmp in campaign_cols:
    rate = data[cmp].mean() * 100
    print(f"{cmp}: {rate:.2f}% accepted")
    print(data.groupby(['Age_Group'])[cmp].mean() * 100)
    print(data.groupby(cmp)[['Age', 'Income', 'Total_Spent', 'Customer_Seniority', 'Is_Parent']].mean())

       # 1️⃣ Age Group acceptance bar chart
    plt.figure(figsize=(6,4))
    sns.barplot(x='Age_Group', y=cmp, data=data, estimator=lambda x: 100*sum(x)/len(x), errorbar=None, palette="Blues_d")
    plt.title(f"{cmp} - Acceptance Rate by Age Group (%)")
    plt.ylabel("Acceptance Rate (%)")
    plt.xlabel("Age Group")
    plt.tight_layout()
    plt.savefig(f"{output_folder}/{cmp}_AgeGroup.png")
    plt.close()

    # 2️⃣ Income distribution boxplot
    avg_income = data.groupby(cmp)['Income'].mean().reset_index()
    plt.figure(figsize=(5,4))
    sns.barplot(x=cmp, y='Income', data=avg_income, palette="pastel")
    plt.title(f"{cmp} - Average Income by Acceptance")
    plt.xlabel("Accepted (0 = No, 1 = Yes)")
    plt.ylabel("Average Income (€)")
    plt.tight_layout()
    plt.savefig(f"{output_folder}/{cmp}_AvgIncome.png")
    plt.close()

      # 3️⃣ Average Total Spending comparison (bar chart)
    avg_spent = data.groupby(cmp)['Total_Spent'].mean().reset_index()
    plt.figure(figsize=(5,4))
    sns.barplot(x=cmp, y='Total_Spent', data=avg_spent, palette="muted")
    plt.title(f"{cmp} - Average Total Spending by Acceptance")
    plt.xlabel("Accepted (0 = No, 1 = Yes)")
    plt.ylabel("Average Total Spent (€)")
    plt.tight_layout()
    plt.savefig(f"{output_folder}/{cmp}_AvgSpent.png")
    plt.close()

    # 4️⃣ Age histogram (accepted only)
    plt.figure(figsize=(6,4))
    sns.histplot(data[data[cmp] == 1]['Age'], bins=10, kde=True, color='salmon', alpha=0.7)
    plt.title(f"{cmp} - Age Distribution of Accepted Clients")
    plt.xlabel("Age")
    plt.ylabel("Number of Accepted Clients")
    plt.tight_layout()
    plt.savefig(f"{output_folder}/{cmp}_AgeHistogram.png")
    plt.close()





    




