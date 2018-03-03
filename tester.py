#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 27 10:53:01 2018

@author: riggs
"""

import pandas as pd
import numpy as np
import os

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

os.chdir(os.path.join('Sources/In Use'))

GFDFile = 'EIA923_Schedules_2_3_4_5_2009_Final_Revision.XLS'
xls = pd.ExcelFile(GFDFile)
sheets = xls.sheet_names
#sheets = [x.lower() for x in sheets]
for i in sheets:
    if 'Receipts' in i:
        df = Clean(xls.parse(i), IDNames)
        break