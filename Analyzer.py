#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 14:33:41 2018

@author: riggs
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

compiled = pd.merge(Eng, Pop.groupby(['Year','Plant State'], as_index=False)['Estimate; Total'].sum()
                    , on=['Year', 'Plant State'], how='left')

#print(compiled[(compiled['Plant Code']==3)&(compiled['AER Fuel Code']=='COL')])
tdf = compiled[(compiled['Plant Code']==3)&(compiled['AER Fuel Code']=='COL')&(compiled['Year']==2016)]

#print(tdf.columns)
a = []  
for n in [5, 10, 25, 50,100,150,200]:
    a.append(tdf.iloc[0]['{0:.0f} neighbors'.format(n)])
        
plt.plot(a)
plt.show()