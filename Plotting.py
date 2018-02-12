#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 15:12:06 2018

@author: riggs
"""

import Analyzer as An
import matplotlib.pyplot as plt
import seaborn as sns
import math as m
import pandas as pd

'''Master Headers               Cost Headers
        'Year'                  'Plant Code'
        'Utility'               'Fuel Group'
        'Plant State'           'Cost' 
        'Plant Code'            'Fuel MMBtu'
        'Primary Mover'         'Elec MWh'
        'AER Fuel Code'         'Elec Pric c/kWh'
        'Fuel Consumption MWh'  'Year'
        'Net Generation MWh'    'Month'
        'Efficiency'
'''

master_df = An.LoadMaster()
cost_df = An.LoadCost()


fuels = cost_df['Fuel Group'].unique()
for i in fuels:
    fuel_df = cost_df[cost_df['Fuel Group']==i]
    plt.plot(fuel_df['Year'].unique(),fuel_df.groupby(['Year'])['Elec Pric c/kWh'].mean(), label = i)

plt.legend()
plt.title('Electricity Price by Fuel Type')
plt.xlabel('Year')
plt.ylabel('Electricity Price c/kWh')
plt.savefig('Elec price by fuel', dpi = 300)
plt.close()

fuels = ['Coal','Natural Gas','Petroleum']
for fuel in fuels:
    fuel_df = cost_df['Elec Pric c/kWh'][(cost_df['Fuel Group']==fuel)&(cost_df['Year']==2015)]
    fuel_df.dropna(inplace=True)
    fuel_df = fuel_df[(fuel_df>=0)&(fuel_df<=100)]
    plt.hist(fuel_df, bins = int(m.sqrt(len(fuel_df))), alpha = 0.5,label = fuel)

plt.ylabel('Counts')
plt.xlabel('Electricity Cost cents/kWh')
plt.title('2015')
plt.legend()
plt.savefig('2015 fuel comparison', dpi = 300)
plt.close()

for year in cost_df['Year'].unique().tolist():
    fuel_df = cost_df['Elec Pric c/kWh'][(cost_df['Year']==year)&(cost_df['Fuel Group']=='Coal')]
    fuel_df.dropna(inplace=True)
    fuel_df = fuel_df[(fuel_df>=0)&(fuel_df<=100)]
    plt.hist(fuel_df, bins = int(m.sqrt(len(fuel_df))), alpha = 0.5,label = str(year))

plt.ylabel('Counts')
plt.xlabel('Electricity Cost cents/kWh')
plt.title('Coal')
plt.legend()
plt.savefig('Coal over time', dpi = 300)
plt.close()

for year in cost_df['Year'].unique().tolist():
    fuel_df = cost_df['Elec Pric c/kWh'][(cost_df['Year']==year)&(cost_df['Fuel Group']=='Natural Gas')]
    fuel_df.dropna(inplace=True)
    fuel_df = fuel_df[(fuel_df>=0)&(fuel_df<=100)]
    plt.hist(fuel_df, bins = int(m.sqrt(len(fuel_df))), alpha = 0.5,label = str(year))

plt.ylabel('Counts')
plt.xlabel('Electricity Cost cents/kWh')
plt.title('Natural Gas')
plt.legend()
plt.savefig('Natural Gas over time', dpi = 300)
plt.close()

for year in cost_df['Year'].unique().tolist():
    fuel_df = cost_df['Elec Pric c/kWh'][(cost_df['Year']==year)&(cost_df['Fuel Group']=='Petroleum')]
    fuel_df.dropna(inplace=True)
    fuel_df = fuel_df[(fuel_df>=0)&(fuel_df<=100)]
    plt.hist(fuel_df, bins = int(m.sqrt(len(fuel_df))), alpha = 0.5,
             label = str(year))

plt.ylabel('Counts')
plt.xlabel('Electricity Cost cents/kWh')
plt.title('Petroleum Fuel')
plt.legend()
plt.savefig('Petroleum over time', dpi = 300)
plt.close()
