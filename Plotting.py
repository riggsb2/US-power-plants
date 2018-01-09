#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 15:12:06 2018

@author: riggs
"""

from bokeh.plotting import figure, output_file, show
from bokeh.charts import Bar

import pandas as pd
import os

def LoadResultDF(file):
    os.chdir(os.path.join('Results'))
    df = pd.read_csv(file)
    os.chdir('..')
    return df


df = LoadResultDF('Plant State_AER Fuel Code_Fuel Consumption MWh..csv')
print(df.head())


# output to static HTML file
output_file("lines.html")

# add a line renderer with legend and line thickness

bar = Bar(df, values='timing', label='interpreter', stack='sample', agg='mean',
          title="Python Interpreter Sampling", legend='top_right', plot_width=400)

show(bar)
