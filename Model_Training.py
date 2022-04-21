from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Internal Libaries 
import sys
import os


# Importing the dataset
df_train = pd.read_csv('loan_engineered.csv', index_col=0)

# Getting rid of infinite data from df_train dataset
df_train = df_train.replace([np.inf, -np.inf], np.nan)
df_train = df_train.dropna()

# Split df_train into x_train and y_train, and x_test and y_test
x_train, x_test, y_train, y_test = train_test_split(df_train.drop('Loan_Status', axis=1), df_train['Loan_Status'], test_size=0.2, random_state=42)

# Random Forest Classification.
rfc = RandomForestClassifier(n_estimators=1000, random_state=1, max_leaf_nodes=10, max_depth = 10)

# XGB Classifier.
xgb = XGBClassifier(n_estimators=1000, random_state=1, max_leaf_nodes=10, max_depth = 10)

# Modular Classification Models
def Model(model, x_train, y_train, x_test, y_test):
    model.fit(x_train, y_train) # Fitting the model. Training the models.
    y_pred = model.predict(x_test) # Predicting the test variables

    print(model.__class__.__name__)
    # Calculating the accuracy of the model
    print('Accuracy Score: ', accuracy_score(y_test, y_pred))

    # Calculating the mean squared error
    print('Mean Squared Error: ', mean_squared_error(y_test, y_pred))

    # Calculating the confusion matrix
    print('Confusion Matrix: \n', confusion_matrix(y_test, y_pred))

    # Plotting the confusion matrix
    plt.figure(figsize=(10,10))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(model.__class__.__name__ + '.png')
    plt.show()
    print('\n')

    # Downlod .png of confusion matrix with the title 'Confusion Matrix'
    

Model(rfc, x_train, y_train, x_test, y_test)
Model(xgb, x_train, y_train, x_test, y_test)