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
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()
import matplotlib.pyplot as plt

#%%--------------------------------loading data--------------------------------

# import data as dataframe
df = pd.read_csv('Master_day6.csv',low_memory=False)

#%%--------------------------------data processing-----------------------------

# drop the raw with concentration value of DMSO
df = df[df["Nom_Conc"].str.contains("DMSO") == False]

# change type for conc_name
df['Nom_Conc'] = pd.to_numeric(df['Nom_Conc'])

df.info() # checking the columns

# keeping only the needed columns
data = df[['X','Y','Area','Mean','Min','Max','Perimeter','Major','Minor','Angle','step_length','step_speed','abs_angle','rel_angle','jell','name','Nom_Conc']].copy()
data.info()

#%%-----------------------delet infinite and nan values------------------------

# turn inf values in nan
data.replace([np.inf, -np.inf], np.nan, inplace=True)

# check that no raw has NaN values
data.isna().values.any()
# if true -> deleting all raw containing NaN
data_noNAN = data.dropna(axis=0)
# check again that no raw has NaN values
data_noNAN.isna().values.any()

#%%---------------------------------classification-----------------------------

# data split
Y = data_noNAN[['Nom_Conc']].copy()
X = data_noNAN[['step_speed']].copy()

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=7000, train_size=20000)
X_test, X_val, y_test, y_val = train_test_split(X_train, y_train, test_size=7000, train_size=7000)

print(len(X_train), len(X_test), len(X_val))

# label encoding in order to normalise the target variable
label_encoder = preprocessing.LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.fit_transform(y_test)
y_val = label_encoder.fit_transform(y_val)

# RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)

RFC_val = rfc.predict(X_val)

print('\nMatrix confusion for RandomForestClassifier :')
print(confusion_matrix(y_val, RFC_val))

acc_RFC = accuracy_score(y_val, RFC_val)
print(f'\nThe accuracy of the model RandomForestClassifier is {acc_RFC:.1%}')

print(RFC_val)

#%%



#%%--------------------------------data visualization--------------------------

#x = data.iloc[0:1552]

#fig, plt = plt.subplots()
#plt.scatter(x['X'],x['Y'])


# create a DF for each 
#j = 0 #initiate new idex position

#for i in range (1,len(data)):
    #if data(data[i]['Nom_Conc']) == 0:
        #Conc_0[j] = data[i]

        


