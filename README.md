# Python-Project
#Python project by utilizing Pandas, Scikit-learn, Matplotlib, and Seaborn for making a model and visualization 
# -*- coding: utf-8 -*-
"""
Created on Mon May 26 15:04:11 2025

@author: Tanvi
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

loans = pd.read_csv(r"C:\Users\Tanvi\OneDrive\Documents\Python\loan_data.csv")
print(loans.info())
print(loans.describe())
print(loans.head())

# Plot histograms
plt.figure(figsize=(10,6))
loans[loans['credit.policy']==1]['fico'].hist(alpha=0.5,color='blue',
                                              bins=30,label='Credit.Policy=1')
loans[loans['credit.policy']==0]['fico'].hist(alpha=0.5,color='red',
                                              bins=30,label='Credit.Policy=0')
plt.legend()
plt.xlabel('FICO')
plt.title('FICO Score Distribution by Credit Policy')
plt.grid(True)
plt.show()

plt.figure(figsize=(10,6))
loans[loans['not.fully.paid']==1]['fico'].hist(alpha=0.5,color='blue',
                                              bins=30,label='Credit.Policy=1')
loans[loans['not.fully.paid']==0]['fico'].hist(alpha=0.5,color='red',
                                              bins=30,label='Credit.Policy=0')
plt.legend()
plt.xlabel('FICO')
plt.title('FICO distribution by Payments')
plt.grid(True)
plt.show()

plt.figure(figsize=(10,6))
sns.countplot(data=loans, x='purpose', hue='not.fully.paid')
plt.title('Count of Loans by Purpose and Not Fully Paid Status')
plt.xlabel('Loan Purpose')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

sns.jointplot(data=loans, x='fico', y='int.rate', kind='scatter', height=8, color='purple')

# Show the plot
plt.show()

#Decision tree model 
Bank_loans = ['purpose']
final_data= pd.get_dummies(loans, columns = Bank_loans, drop_first = True)

final_data.info()

from sklearn.model_selection import train_test_split
X = final_data.drop('not.fully.paid',axis=1)
y = final_data['not.fully.paid']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state = 101)

from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()
dtree.fit(X_train, y_train)

#Predictions and Evaluation of Decision Tree BY Classification and Confusion matrix
predictions = dtree.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))

#Random Forest model
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators = 300)
rfc = RandomForestClassifier(class_weight='balanced', random_state=101)
rfc.fit(X_train, y_train)

rfc_pred = rfc.predict(X_test)
print(classification_report(y_test,rfc_pred))
print(confusion_matrix(y_test,rfc_pred))

#Test the model by adding one person info
new_data = {
    'credit.policy': 1,
    'int.rate': 0.13,
    'installment': 250.0,
    'log.annual.inc': 10.5,
    'dti': 15.3,
    'fico': 720,
    'days.with.cr.line': 5000.0,
    'revol.bal': 12000,
    'revol.util': 30.0,
    'inq.last.6mths': 1,
    'delinq.2yrs': 0,
    'pub.rec': 0,
    'purpose_credit_card': 0,
    'purpose_debt_consolidation': 1,
    'purpose_educational': 0,
    'purpose_home_improvement': 0,
    'purpose_major_purchase': 0,
    'purpose_small_business': 0,
    'purpose_vacation': 0,
    'purpose_wedding': 0
}
new_df = pd.DataFrame([new_data])
new_df = new_df[X_train.columns]  # Ensures the order matches
# Use Decision Tree
dtree_prediction = dtree.predict(new_df)
print("Decision Tree Prediction:", "Not Fully Paid" if dtree_prediction[0] == 1 else "Fully Paid")

# Use Random Forest
rfc_prediction = rfc.predict(new_df)
print("Random Forest Prediction:", "Not Fully Paid" if rfc_prediction[0] == 1 else "Fully Paid")


