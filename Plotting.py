#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 15:12:06 2018

@author: riggs
"""

import Analyzer as An
import matplotlib.pyplot as plt
import seaborn as sns

master_df = An.LoadMaster()

Fuel_gen = master_df.groupby(['Year', 'AER Fuel Code']).agg({'Net Generation MWh': 'sum'})
Fuel_gen = Fuel_gen[Fuel_gen['Net Generation MWh']>0]

threshold = 0.05
Fuel_gen['Frac'] = Fuel_gen.groupby(['Year']).apply(lambda x: x / float(x.sum()))
Fuel_gen = Fuel_gen[Fuel_gen['Frac']>threshold]

Fuel_gen.reset_index(inplace = True)

time_df = Fuel_gen.groupby(['Year'], as_index = False)['Net Generation MWh'].sum()



