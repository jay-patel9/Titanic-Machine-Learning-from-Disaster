# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 02:08:39 2018

@author: JAY
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Load Dataset
test_df = pd.read_csv('test.csv')

#Cleaning Data

#Missing Values

sns.heatmap(data=test_df.isnull(),cmap='viridis',yticklabels=False,cbar=False)
sns.boxplot(x='Pclass',y='Age',data=test_df)
sns.boxplot(x='Pclass',y='Fare',data=test_df)

#Apply mean value to missing data
def missing_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):

        if Pclass == 1:
            return 42 #Mean Value from above boxplot

        elif Pclass == 2:
            return 26 #Mean Value from above boxplot

        else:
            return 24 #Mean Value from above boxplot

    else:
        return Age
    
test_df['Age'] = test_df[['Age','Pclass']].apply(missing_age,axis =1)
test_df.drop(152,inplace=True)
sns.heatmap(data=test_df.isnull(),cmap='viridis',yticklabels=False,cbar=False)

#Drop certain column becauce data cannot be processed
test_df.drop('Cabin',axis=1,inplace=True)
test_df.drop('Ticket',axis=1,inplace=True)
test_df.drop('Name',axis=1,inplace=True)

#Converting Categorical Feature
sex = pd.get_dummies(test_df['Sex'],drop_first=True)
embark = pd.get_dummies(test_df['Embarked'],drop_first=True)

test_df.drop(['Sex','Embarked'],axis=1,inplace=True)

#Add Converted Feature
test_df = pd.concat([test_df,sex,embark],axis=1)

#Create test data
X_test = test_df.iloc[:, 2:10].values

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_test = sc.fit_transform(X_test)

# Applying PCA for Dimensionality Reduction of Independent Variable
from sklearn.decomposition import PCA
pca = PCA(n_components=3)
X_test = pca.fit_transform(X_test)
explained_variance = pca.explained_variance_ratio_

#Import Classifier
from sklearn.externals import joblib
lg = joblib.load('model')

pred = lg.predict(X_test)
pid = test_df['PassengerId'].values
dpid = pd.DataFrame(pid)
dpred = pd.DataFrame(pred)

answer = pd.concat([dpid,dpred],axis=1)

gen_sub = pd.read_csv('gender_submission.csv')
gen_sub.drop(152,inplace=True)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(gen_sub.iloc[:,1],answer.iloc[:,1])

answer.to_csv('output.csv',index=None,header=True)
#Accuracy 0.7697













