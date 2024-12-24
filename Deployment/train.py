#!/usr/bin/env python
# coding: utf-8

#load libraries and packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

# PARAMETERS
C = 1.0
n_splits = 5


# DATA PREPARATION AND CLEANING

df = pd.read_csv(r"C:\Users\H P\Desktop\Machine Learning Zoomcamp\Logistic Regression\Telco-Customer-Churn.csv") #load dataset from copy downloaded from kaggle

###formats column names
df.columns = df.columns.str.lower().str.replace(' ', '_')

###formats all index by removing spaces and making index lowercase. ie formats rows
categorical_columns = list(df.dtypes[df.dtypes == 'object'].index)

for c in categorical_columns:
    df[c] = df[c].str.lower().str.replace(' ', '_')

df.totalcharges = pd.to_numeric(df.totalcharges, errors = 'coerce')  #changes object dtypes to int and second input ignores nulls, and  other dtypes included

df.totalcharges = df.totalcharges.fillna(0)

df.churn = (df.churn == 'yes').astype(int)  #assigns 0 and 1 to yes and no

#SPLITTING DATA
df_full_train, df_test = train_test_split(df, test_size = 0.2, random_state = 1) ##splits test to 20%


numerical = ['tenure','monthlycharges', 'totalcharges' ]

categorical = ['gender', 'seniorcitizen', 'partner', 'dependents',
     'phoneservice', 'multiplelines', 'internetservice',
       'onlinesecurity', 'onlinebackup', 'deviceprotection', 'techsupport',
       'streamingtv', 'streamingmovies', 'contract', 'paperlessbilling',
       'paymentmethod']


# TRAINING DATA FUNC


def train(df_train, y_train, C=1.0):
    dicts = df_train[categorical + numerical].to_dict(orient = 'records')

    dv = DictVectorizer(sparse = False)
    X_train = dv.fit_transform(dicts)

    model = LogisticRegression(solver='lbfgs', max_iter=10000, C=C)
    model.fit(X_train, y_train)

    return dv, model


# PREDICTING FUNC


def predict(df, dv, model):
    dicts = df[categorical + numerical].to_dict(orient = 'records')
    
    X = dv.transform(dicts)
    y_pred = model.predict_proba(X)[:,1]

    return y_pred


#TRAINING AND VALIDATION 

print(f'doing validation with C= {C}')
kfold = KFold(n_splits = n_splits, shuffle=True, random_state = 1)

#train_idx, val_idx = next(kfold.split(df_full_train)) #divides the dataset into 10 and trains 9 parts and validates 1 parts and iterates 10 times.
#returns the index for the trained parts and validated parts)
scores = []
fold = 0
# Compute formula to calculate AUC for each of the 5 groups
for train_idx, val_idx in (kfold.split(df_full_train)):
    df_train = df_full_train.iloc[train_idx]
    df_val = df_full_train.iloc[val_idx]

    y_train = df_train.churn.values
    y_val = df_val.churn.values

    #uses func created for train and predict to train and predict each part
    dv, model = train(df_train, y_train, C=C ) #takes each part of the train and does DV, and fit the model
    y_pred = predict(df_val, dv, model) #predicts on val for each model 

    auc = roc_auc_score(y_val, y_pred) #find auc score for each iteration
    scores.append(auc) #stores all in the scores array above

    print(f'auc on fold {fold} is {auc}')
    fold = fold + 1

print('validation results')
print('C=%s %.3f +- %.3f' % (C, np.mean(scores), np.std(scores)))

scores


#TRAIN FINAL MODEL
#trains full dataset and test on test dataset

print('training the final model')
dv, model = train(df_full_train, df_full_train.churn.values, C=C )  
y_pred = predict(df_test, dv, model) #predicts on val for each model 

y_test = df_test.churn.values
auc = roc_auc_score(y_test, y_pred)
auc

print (f'auc = {auc}')


#SAVE MODEL

output_file = f'model_C={C}.bin' #create file name to be exported
output_file

#alternate way
with open(output_file, 'wb') as f_out:
    pickle.dump((dv,model), f_out)


print(f'model saved to {output_file}')



