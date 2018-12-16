# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 01:10:34 2018

@author: JAY
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Load Dataset
df = pd.read_csv('train.csv')
#test_df = pd.read_csv('test.csv')

#Cleaning Data
#Missing Values

sns.heatmap(data=df.isnull(),cmap='viridis',yticklabels=False,cbar=False)
#sns.heatmap(data=test_df.isnull(),cmap='viridis',yticklabels=False,cbar=False)
sns.boxplot(x='Pclass',y='Age',data=df)
#sns.boxplot(x='Pclass',y='Age',data=test_df)

#Apply mean value to missing data
def missing_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):

        if Pclass == 1:
            return 37 #Mean Value from above boxplot

        elif Pclass == 2:
            return 29 #Mean Value from above boxplot

        else:
            return 24#Mean Value from above boxplot

    else:
        return Age
    
df['Age'] = df[['Age','Pclass']].apply(missing_age,axis =1)

#No missing data in age column
sns.heatmap(data=df.isnull(),cmap='viridis',yticklabels=False,cbar=False)

#Drop certain column becauce data cannot be processed
df.drop('Cabin',axis=1,inplace=True)
df.drop('Ticket',axis=1,inplace=True)
df.drop('Name',axis=1,inplace=True)

#Converting Categorical Feature
sex = pd.get_dummies(df['Sex'],drop_first=True)
embark = pd.get_dummies(df['Embarked'],drop_first=True)

df.drop(['Sex','Embarked'],axis=1,inplace=True)

#Add Converted Feature
df = pd.concat([df,sex,embark],axis=1)

#Create Train data
X_train = df.iloc[:, 2:11].values
y_train = df.iloc[:, 1].values

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)

# Applying PCA for Dimensionality Reduction of Independent Variable
from sklearn.decomposition import PCA
pca = PCA(n_components=3)
X_train = pca.fit_transform(X_train)
explained_variance = pca.explained_variance_ratio_

# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 101)
classifier.fit(X_train, y_train)

#Save Classifier
from sklearn.externals import joblib
joblib.dump(classifier, 'model')













