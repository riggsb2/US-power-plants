#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 15:27:38 2018

@author: riggs
"""


import pandas as pd
import matplotlib.pyplot as plt
import os
from collections import defaultdict
from sklearn import utils,linear_model, base, preprocessing
from sklearn.svm import SVR, LinearSVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import ShuffleSplit
import numpy as np
from sklearn.cluster import KMeans

def LoadSource(filename):
    os.chdir(os.path.join('Sources'))
    df = pd.read_csv('.'.join((filename,'csv')))
    os.chdir('..')
    return(df)
    
def SecondClean(df):

    df.dropna(axis=0, how = 'any',inplace =True)
    
    df = df[df['Fuel Consumption MWh']>0]
    df = df[df['Net Generation MWh']>0]
    
    df['Zipcode'] = (df['Zipcode'].astype(str).str.zfill(5)).astype(str)
    return df

def CheckDensity(df):
    density = defaultdict()
    for col in df.columns.values:
        density[col] = 1-df[col].isnull().sum()/len(df[col])
    Dense = pd.DataFrame.from_dict(density, orient  ='index')
    issues = Dense[Dense[0]<0.9]
    print(issues)
        
eng = LoadSource('Engineered_Features')
#both of these should be moved to data compiler
eng = eng[pd.notna(eng['Year'])] #removes entires that don't have a plant
eng.drop_duplicates(inplace=True) #removes duplicates
print(eng.columns)

todrop = ['Utility','Zipcode','Plant_Code','Plant_Id','Fuel_Group','County']
    
def Capacity(df):
    train_df = pd.DataFrame()
    
    scaler = preprocessing.MinMaxScaler()

    train_df['Label'] =  df['Net_Generation_MWh']
    train_df['Fuel_Cost_Mean']=df['Fuel_Cost_Mean'].fillna(0)
    train_df['Fuel_Cost_StDev']=df['Fuel_Cost_StDev'].fillna(0)
    train_df['Fuel_Cost_Var']=df['Fuel_Cost_Var'].fillna(0)

    fuel = df['AER_Fuel_Code']
    fuel.dropna(axis=0, how = 'any',inplace =True)
    fuels_OH = pd.get_dummies(fuel)
    
    sector = df['Sector']
    sector.dropna(axis=0, how = 'any',inplace =True)
    sector_OH = pd.get_dummies(sector)
    
    power_list = ['5 power', '10 power','25 power', '50 power','100 power',
                '150 power', '200 power', '300 power']
    power = df[power_list]
    power.dropna(axis=0, how = 'any',inplace =True)
    power_scale = pd.DataFrame(scaler.fit_transform(power),columns = power_list)
    
    pop_list = ['5 pop', '10 pop','25 pop', '50 pop','100 pop',
                '150 pop', '200 pop', '300 pop']
    pop = df[pop_list]
    pop.dropna(axis=0, how = 'any',inplace =True)
    pop_scale = pd.DataFrame(scaler.fit_transform(power),columns = pop_list)
    
    
    train_df = train_df.join([sector_OH,power_scale,fuels_OH,pop_scale], how='outer') 
    
    years = df['Year'].values.reshape(len(df['Year']),1)
    years_scaled = scaler.fit_transform(years)
    train_df['Year']=pd.Series(years_scaled.flatten())
    
    train_df.dropna(axis=0, how = 'any',inplace =True)

    
    train_label = train_df['Label']
    train_df.drop(['Label'],axis =1,inplace=True)
    
    Forest = RandomForestRegressor(n_estimators = 50)
    Forest.fit(train_df, train_label)
    #importance =linreg.coef_
    importance = pd.DataFrame(Forest.feature_importances_,
                              index = train_df.columns.values,
                              columns = ['Import'])
    #test_label = train_df['Label']
    result = pd.Series(Forest.predict(train_df))
    
    print(len(train_df))
    print(Forest.score(train_df, train_label))
    
    print(importance.sort_values(by=['Import'], ascending = False))
    plt.loglog(train_label,result, linestyle='None', marker = '.')
    plt.xlabel('From Data')
    plt.ylabel('Prediction')
    plt.show()
    plt.close()
    resid = train_label-result
    plt.plot(resid)
    plt.show()
    plt.close()

    
Capacity(eng)   
    
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