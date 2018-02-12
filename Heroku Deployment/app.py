#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 24 09:18:30 2017

@author: riggs
"""

import os

from flask import Flask,render_template,request
import pandas as pd
from bokeh.io import show
#from bokeh.charts import Bar
from bokeh.models.widgets import Dropdown
from bokeh.layouts import widgetbox
from bokeh.plotting import curdoc


def LoadResultDF(file):
    os.chdir(os.path.join('Results'))
    df = pd.read_csv(file)
    os.chdir('..')
    return df

app = Flask(__name__)

@app.route('/',methods =['GET','POST'])
def main():
    if request.method == 'GET':
        df2 = LoadResultDF('Plant State_AER Fuel Code_Fuel Consumption MWh.csv')
        
        menu = [("Entire US", "Entire US"), ("Alaska", "AK"), ("Alabama", "AL"),("Arkansas", "AR"),
                ("Arizona", "AZ"),("California", "CA"),("Colorado", "CO"),("Connecticut", "CT"),("District of Columbia", "DC"),
                ("Deleware", "DE"),("Floride", "FL"),("Georgia", "GA"),("Hawaii", "HI"),("Iowa", "IA"),
                ("Idaho", "ID"),("Illinois", "IL"),("Indiana", "IN"),("Kansas", "KS"),("Kentucky", "KY"),
                ("Louisiana", "LA"),("Massachussettes", "MA"),("Maryland", "MD"),("Maine", "ME"),("Michigan", "MI"),
                ("Minnesota", "MN"),("Missouri", "MO"),("Mississippi", "MS"),("Montana", "MT"),("North Carolina", "NC"),
                ("North Dakota", "ND"),("Nebraska", "NE"),("New Hamsphire", "NH"),("New Jersey", "NJ"),("New Mexico", "NM"),
                ("Nevada", "NV"),("New York", "NY"),("Ohio", "OH"),("Oklahoma", "OK"),("Oregon", "OR"),
                ("Pennsylvania", "PA"),("Rhode Island", "RI"),("South Carolina", "SC"),("South Dakota", "SD"),("Tennessee", "TN"),
                ("Texas", "TX"),("Utah", "UT"),("Virginia", "VA"),("Washington", "WA"),("Wisconsin", "WI"),
                ("West Virgninia", "WV"),("Wyoming", "WY")]
        dropdown = Dropdown(label="States",default_value = 'Entire US', menu=menu)

        def update(att,old,new):
            Region = dropdown.value
            Fuel = df2[df2['Plant State']==Region]
            Fuel = Fuel[Fuel['Fuel Consumption MWh']>=0.01]
            #bar2 = Bar(Fuel, values='Fuel Consumption MWh', label='Year', stack='AER Fuel Code', 
            #  title="State Portfolio of Fuel Type by Fuel Consumption", legend='top_right', plot_width=600, plot_height = 500)  
            
            #show(bar2)
            
        dropdown.on_change('value',update)

        curdoc().add_root(dropdown)
        
        return render_template('index.html')
    else:
        return render_template('index.html')

#@app.route('/StatePortfolios',methods = ['GET'])
#def StatePortfolios():



    #interact(update, Region=states)
    
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)