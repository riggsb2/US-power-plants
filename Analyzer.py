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
from sklearn import utils,linear_model
from sklearn import model_selection
from sklearn.feature_extraction.text import CountVectorizer,HashingVectorizer
import numpy as np
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.cluster import KMeans

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


setup = eng.copy(deep=True)
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

print(setup.columns)
correlation = setup.corr()


#Trying to do modeling of Lat Long for fuel
def LatLongModel():
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
    
def StateCapacityModel():
    Gen_label = setup['Net Generation MWh']
    train_data = setup.copy(deep=True)
    train_data.drop(labels = ['Efficiency','Fuel Consumption MWh','Primary Mover',
                              'AER Fuel Code',
                              'County','Latitude','Longitude'], axis=1,
                                inplace=True)
        
    state_data = CountVectorizer().fit_transform(train_data['Plant State'])
    
    linreg = linear_model.Ridge()  
    linreg.fit(state_data, Gen_label)
    
    
    pred = sorted(train_data['Plant State'].unique())
    pred_std = setup.groupby(['Plant State'])['Net Generation MWh'].mean()

    pred_data = CountVectorizer().fit_transform(pred)
    result = linreg.predict(pred_data)
    plt.scatter(pred_std,result)
    plt.show()
    
def NNeighbor():
    Gen_label = setup['Net Generation MWh']
    train_data = setup.copy(deep=True)
    train_data.drop(labels = ['Efficiency','Fuel Consumption MWh','Primary Mover',
                              'AER Fuel Code','County','Latitude','Longitude',
                              'Plant State','Utility','Net Generation MWh'], axis=1,
                                inplace=True)
    
    print(train_data.head())
    linreg = linear_model.Ridge()  
    linreg.fit(train_data, Gen_label)
    
    result = linreg.predict(train_data)
    plt.scatter(Gen_label,result)
    plt.scatter(Gen_label,Gen_label)

    plt.show() 
    
def Clustering():
    import seaborn as sns
    groups = 20
    tmp = eng[eng['Year']==2016]
    Lat_long_data = tmp[['Latitude','Longitude']]
    Lat_long_data.reset_index(inplace = True, drop = True)
    km = KMeans(n_clusters=groups, max_iter=30, n_init=1)
    km.fit(Lat_long_data)
    print(Lat_long_data.head())
    results = pd.concat([Lat_long_data,pd.Series(km.labels_, name = 'color')],axis = 1)  
    print(results.head(20))
    colors = sns.color_palette("hls", groups).as_hex()
    
    results.plot.scatter(x = 'Longitude',y='Latitude', c=results['color'].apply(lambda x: colors[x]))
    
    #plt.scatter(*Lat_long_data, c=(km.labels_), cmap=plt.cm.plasma)

Clustering()