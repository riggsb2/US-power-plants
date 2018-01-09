#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 14:38:12 2018

@author: riggs
"""

import pandas as pd


df = pd.DataFrame([[1,2],[3,4],[4,5],[6,8],[9,10]],columns = ['First','Second'])
print(df)
include = [1,2,3,4,5]

df = df[df['First'].isin(include)]

print(df)
