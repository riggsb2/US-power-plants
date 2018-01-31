#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 27 10:53:01 2018

@author: riggs
"""

import pandas as pd
import numpy as np


lat_rng = [19.50139, 64,85694]
long_rng = [-161.75583, -68.01197]

df = pd.DataFrame(dtype = str)

df['lat'] = pd.Series(np.random.uniform(lat_rng[0],lat_rng[1],100), dtype = object)
df['long'] = np.random.uniform(long_rng[0],long_rng[1],100)

print(df['lat'].dtype)

df.set_value(3,'lat', 'word')
df.set_value(4,'lat', 'a phrase')

print(df.head())

df['lat'].replace(regex=True,inplace=True,to_replace=r'[a-zA-Z]+',value=np.nan)

print(df.head())

df1 = pd.DataFrame([[1, np.nan]])
df2 = pd.DataFrame([[3, 4]])
print(df1.combine_first(df2))
