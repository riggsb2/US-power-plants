#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 15:27:38 2018

@author: riggs
"""


import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn import preprocessing
from sklearn import base
from sklearn.neighbors import KNeighborsClassifier
from sklearn import utils 
from sklearn import model_selection
from sklearn.feature_extraction.text import CountVectorizer,HashingVectorizer
import numpy as np

def LoadSource(filename):
    os.chdir(os.path.join('Sources'))
    df = pd.read_csv('.'.join((filename,'csv')))
    os.chdir('..')
    return(df)
    
def SecondClean(df):
    eff_upper = 1
    eff_lower = 0
    trouble_df = df[df['Efficiency']>eff_upper]
    trouble_df.append(df[df['Efficiency']<eff_lower])
    df.dropna(axis=0, how = 'any',inplace =True)
    
    print(len(trouble_df), ' entries have been removed')
    
    df = df[df['Efficiency']<eff_upper]
    df = df[df['Efficiency']>eff_lower]
    df = df[df['Fuel Consumption MWh']>0]
    df = df[df['Net Generation MWh']>0]
    
    df['Zipcode'] = (df['Zipcode'].astype(str).str.zfill(5)).astype(str)
    return df

eng = LoadSource('Engineered Features')
eng = SecondClean(eng)

setup = eng
setup.drop(labels=['Plant Code','Zipcode'], axis=1, inplace=True)

to_norm =['Fuel Consumption MWh', 'Net Generation MWh', '5 power', '5 neighbors',
       '10 power', '10 neighbors', '25 power', '25 neighbors', '50 power',
       '50 neighbors', '100 power', '100 neighbors', '150 power',
       '150 neighbors', '200 power', '200 neighbors']

for column in to_norm:
    x = eng[[column]].values.astype(float)
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    x_scaled = [float(x) for x in x_scaled]
    setup[column] = pd.Series(x_scaled)
    
correlation = setup.corr()

print(eng['Primary Mover'].unique())
#Trying to do modeling of Lat Long for fuel

Fuel_label =  eng['AER Fuel Code']
Lat_long_data = eng[['Latitude','Longitude']]

code = 1
for fuel in Fuel_label.unique():
    Fuel_label.replace(fuel,code,inplace = True)
    code+=1

knn = KNeighborsClassifier()

#knn.fit(shuf_data, shuf_label)

cv = model_selection.StratifiedKFold(n_splits=5,shuffle=True)

gs = model_selection.GridSearchCV(
    knn,
    {"n_neighbors": range(10,20)},
    cv=cv,
    n_jobs=4,
    scoring='neg_mean_squared_error'
    )

gs.fit(Lat_long_data, Fuel_label)

print(gs.best_params_)

new_fuels = gs.predict(Lat_long_data)

print(new_fuels)

plt.scatter(Fuel_label,new_fuels)
plt.show()





