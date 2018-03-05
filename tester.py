#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 27 10:53:01 2018

@author: riggs
"""

import pandas as pd
import numpy as np
import os
import sys
import math as m

def Clean(df,keys):
    
    for key in keys:
        head_loc = np.where(df==key)[0]
        if len(head_loc)>0:
            head_loc = head_loc.item()
            break

    df.columns = df.iloc[head_loc]
    todrop = np.arange(0,head_loc+1,1)
    df.drop(todrop,inplace=True)
    
    df.columns = df.columns.str.replace('\n',' ')
    df.columns = df.columns.str.replace('  ',' ')
    mask = df.isin(['.'])   
    df = df.where(~mask, other=0)
    mask = df.isin([0])   
    df = df.where(~mask, other=np.nan)
    return(df)

IDNames = ['Plant Id', 'Plant Code','Plant ID']
Fuel_Trans = {'Natural Gas': 'NG', 'Coal': 'COL', 'Petroleum':'DFO',
              'Petroleum Coke': 'PC','Other Gas': 'OOG'}

Fuel_Trans = {'Natural Gas': 'NG', 'Coal': 'COL', 'Petroleum':'DFO',
              'Petroleum Coke': 'PC','Other Gas': 'OOG'}

def LoadSource(filename):
    os.chdir(os.path.join('Sources'))
    df = pd.read_csv('.'.join((filename,'csv')))
    os.chdir('..')
    return(df)
#GFDFile = 'EIA923_Schedules_2_3_4_5_2009_Final_Revision.XLS'
#xls = pd.ExcelFile(GFDFile)
#sheets = xls.sheet_names

    
def GeoMile(lat):
    km_mile = 0.621
    try:
        rad_lat = m.radians(float(lat))
    except:
        print('THIS ONE BROKE:',lat)
    lat_mi = km_mile*(111132.954-559.822*m.cos(2*rad_lat) +1.175*m.cos(4*rad_lat))/1000
    long_mi = km_mile*(m.pi*6378137.0*m.cos(rad_lat)/(180*(1-0.00669437999014*m.sin(rad_lat)**2)**0.5))/1000
    return(lat_mi,long_mi)
    
def FindNeighbors(x):    
    clat = x['Latitude']
    clong = x['Longitude']
    #dlat = x['dlat']
    #dlong = x['dlong']
    results = []
    for dist in dists:
        GM = GeoMile(clat)
        dlat,dlong = dist/GM[0],dist/GM[1]
        #print(clat, clong)
        #print(dlat,dlong)
        #print(clat+dlat,clat-dlat,clong+dlong,clong-dlong)
        #print(py_df['Estimate; Total'].sum())
        t = (py_df['Estimate; Total'][(py_df['INTPTLAT']<=clat+dlat) & (py_df['INTPTLAT']>=clat-dlat) &
                     (py_df['INTPTLONG']<=clong+dlong) & (py_df['INTPTLONG']>=clong-dlong)])
        #print(t.sum())
        population = py_df['Estimate; Total'][(py_df['INTPTLAT']<=clat+dlat) & (py_df['INTPTLAT']>=clat-dlat) &
                     (py_df['INTPTLONG']<=clong+dlong) & (py_df['INTPTLONG']>=clong-dlong)]
        results.extend([population.sum()])
    return results
            
df = LoadSource('Compiled Dataset')
pop_df = LoadSource('Population Set').groupby(['Year','INTPTLAT','INTPTLONG'],as_index=False).sum()
tdf = df.groupby(['Year','Plant_Code','Latitude','Longitude'],as_index=False).sum()

dists = [15]
#tdf[['dlat','dlong']] = dist/tdf['Latitude'].apply(GeoMile).apply(pd.Series)

years = tdf['Year'].unique()
power_df = pd.DataFrame()
years = [2008]
for year in years:
    py_df = pop_df[pop_df['Year']==year]
    print(year)
    print(py_df['Estimate; Total'].sum())
    power_df=power_df.append(tdf[tdf['Year']==year].apply(FindNeighbors,axis=1).apply(pd.Series)) 
power_df.dropna(axis = 1, how = 'all', inplace = True) #for some reason power_df has Year, Utility, etc. all Nan so they need to be dropped

#col = []
#for dist in dists:
#    col.extend(['{0:.0f} power'.format(dist),'{0:.0f} neighbors'.format(dist),'{0:.0f} pop'.format(dist)])
#power_df.columns = col
power_df[['Year','Plant_Code']] = tdf[['Year','Plant_Code']]

print('Neighbors Compiled: Resulting DF.head()')

print(power_df.head())

    
