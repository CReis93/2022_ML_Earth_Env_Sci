# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 10:29:15 2022

@author: Christophe Reis
"""

#%%-------------------------------loading package------------------------------

# load the package
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

#%%--------------------------------loading data--------------------------------

# import data as dataframe

df = pd.read_csv('Master_day6.csv',low_memory=False)

#%%--------------------------------data processing-----------------------------

# drop the raw with concentration value of DMSO (solute test)
df = df[df["Nom_Conc"].str.contains("DMSO") == False]

# change type for conc_name
df['Nom_Conc'] = pd.to_numeric(df['Nom_Conc'])

df.info() # checking the columns

# keeping only the needed columns
data = df[['Area','Perimeter','Major','Minor','step_length','step_speed','abs_angle','rel_angle','Nom_Conc']].copy()
data.info()

#%%---take 5 random line for the report & export it (no use for the code)------

#sampleDF = data.sample(n = 5)
#sampleDF.to_csv('sampleDF.csv')

#%%-----------------------delet infinite and nan values------------------------

# turn inf values in nan
data.replace([np.inf, -np.inf], np.nan, inplace=True)

# check that no raw has NaN values
data.isna().values.any()
# if true -> deleting all raw containing NaN
data_noNAN = data.dropna(axis=0)
# check again that no raw has NaN values
data_noNAN.isna().values.any()


#%%-------------------------------represent data-------------------------------

X1 = data_noNAN[['step_speed']].copy()
X2 = data_noNAN[['step_length']].copy()
c = data_noNAN[['Nom_Conc']].copy()
c = label_encoder.fit_transform(c) 

fig = plt.figure()

plt.scatter(X1, X2, c=c)

# Set figure title and axis labels
plt.title('step_speed vs abs_angle for each measurement point')
plt.xlabel("step_speed [pixel/sec]")
plt.ylabel("step_length [pixel]")

#%%------------------------------split the data--------------------------------

# select the variables
X = data_noNAN[['step_speed','Major']].copy()
Y = data_noNAN[['Nom_Conc']].copy()

# split the dataset for train, validation and test
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, train_size=0.7)
X_test, X_val, y_test, y_val = train_test_split(X, Y, test_size=0.15, train_size=0.15)

# label encoding in order to normalise the target variable
label_encoder = preprocessing.LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.fit_transform(y_test)

print(len(X_train), len(X_test), len(X_val))

#%%--------------------------- train and test the model------------------------
#with RandomForestClassifier and no hyperparam 

rfc = RandomForestClassifier()

# fit the data (training)
rfc.fit(X_train, y_train)

# predict after training on test set
RFC_test = rfc.predict(X_test)

# print the matrix and the accuracy
print('\nMatrix confusion for RandomForestClassifier (without HP) :')
print(confusion_matrix(y_test, RFC_test))

acc_RFC = accuracy_score(y_test, RFC_test)
print(f'\nThe accuracy of the model RandomForestClassifier is (without HP) {acc_RFC:.1%}')

#%%--------------------------------- search HP---------------------------------

# initiate the gridsearch (inspired from https://www.kaggle.com/code/sociopath00/random-forest-using-gridsearchcv/notebook)
params = { 
    'n_estimators': [150],
    'max_features': ['auto','sqrt'],
    'max_depth' : [4,8],
    'criterion' :['gini', 'entropy']
}
gsc = GridSearchCV(rfc, params, cv=5)

gsc.fit(X_train, y_train)

#Results key
print('\nResults keys :')
sorted(gsc.cv_results_.keys())

# print the best hyperparameters
print('\nBest params :')
print(gsc.best_params_)

# and score
print('\nBest score :')
print(gsc.best_score_)

#Best estimator
print('\nBest estimator :')
gsc.best_estimator_

#%%--------------------------- train and test the model------------------------
#with DecisionTreesClassifier and no hyperparam 

dtc = DecisionTreeClassifier(min_impurity_decrease=0,criterion='gini', splitter='best', min_samples_split=2, min_samples_leaf = 1,min_weight_fraction_leaf=0) 

# fit the data (training)
dtc.fit(X_train, y_train)

# predict after training on test set
dtc_test = dtc.predict(X_test)

# print the matrix and the accuracy
print('\nMatrix confusion for RandomForestClassifier (without HP) :')
print(confusion_matrix(y_test, dtc_test))

acc_RFC = accuracy_score(y_test, dtc_test)
print(f'\nThe accuracy of the model RandomForestClassifier is (without HP) {acc_RFC:.1%}')

#%%--------------------------------- search HP---------------------------------

parameters = {'max_features': ['auto', 'sqrt', 'log2'],
              'ccp_alpha': [0.1, .01, .001],
              'max_depth' : [5, 6, 7, 8, 9]
             }

gsc1 = GridSearchCV(dtc, parameters,cv=5)

# fit the data (training)
gsc1.fit(X_train,y_train)

#Results key
print('\nResults keys :')
sorted(gsc1.cv_results_.keys())

# print the best hyperparameters
print('\nBest params :')
print(gsc1.best_params_)

# and score
print('\nBest score :')
print(gsc1.best_score_)

#Best estimator
print('\nBest estimator :')
gsc1.best_estimator_

#%%--------------------------- train and test the model------------------------
#with DecisionTreesClassifier and hyperparam 

dtc = DecisionTreeClassifier(min_impurity_decrease=0,criterion='gini', splitter='best', min_samples_split=2, min_samples_leaf = 1,min_weight_fraction_leaf=0,ccp_alpha = 0.001, max_depth= 9,max_features ='sqrt') 

# fit the data (training)
dtc.fit(X_train, y_train)

# predict after training on test set
dtc_test = dtc.predict(X_test)

# print the matrix and the accuracy
print('\nMatrix confusion for RandomForestClassifier (without HP) :')
print(confusion_matrix(y_test, dtc_test))

acc_RFC = accuracy_score(y_test, dtc_test)
print(f'\nThe accuracy of the model RandomForestClassifier is (without HP) {acc_RFC:.1%}')

