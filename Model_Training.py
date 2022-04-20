from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Internal Libaries 
import sys
import os


# Importing the dataset
df_train = pd.read_csv('loan_training_engineered.csv', index_col=0)

# Getting rid of infinite data from df_train dataset
df_train = df_train.replace([np.inf, -np.inf], np.nan)
df_train = df_train.dropna()

# Split df_train into x_train and y_train, and x_test and y_test
x_train, x_test, y_train, y_test = train_test_split(df_train.drop('Loan_Status', axis=1), df_train['Loan_Status'], test_size=0.2, random_state=42)

# Random Forest Classifier
rfc = RandomForestClassifier(n_estimators=1000, random_state=1, max_leaf_nodes=10, max_depth = 10)

# Fitting dataset to model
rfc.fit(x_train, y_train)

# Predicting the test set results
predicted_rfc = rfc.predict(x_test)

# Turning data into numpy array
predicted_rfc = np.array(predicted_rfc).reshape(-1, 1)
y_test = np.array(y_test).reshape(-1, 1)

# Scoring Testing Set from predicted_rfc
score_rfc = round(accuracy_score(y_test, predicted_rfc)*100, 3)
print(str(score_rfc) + '%')