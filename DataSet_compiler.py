#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 14:33:41 2018

@author: riggs
"""
import numpy as np
import pandas as pd
import os
from scipy import stats
from collections import defaultdict
import matplotlib.pyplot as plt
import time
import math as m

def GenDataset():
    print('Regenerating Master DataSet')
    
    #For master dataframe
    MMBTU_MWH = 0.29307
    IDNames = ['Plant Id', 'Plant Code','Plant ID']
    UtilityNames = ['OPERATOR NAME','UTILITY_NAME','Utility Name','Operator Name']
    YearNames = ['Year','YEAR']
    PlantNames = ['Plant Name']
    StateNames = ['State', 'Plant State']
    MoverNames = ['Reported Prime Mover']
    FuelNames = ['AER Fuel Type Code']
    ElecFuelNames = ['Elec Fuel Consumption MMBtu','ELEC FUEL CONSUMPTION MMBTUS']
    GenNames = ['Net Generation (Megawatthours)','NET GENERATION (megawatthours)']
    CensusNames = ['Census Region']
    NERCNames = ['NERC Region']

    #From 860 Reports
    Plant_Codes = ['PLNTCODE,N,5,0','PLNTCODE','PLANT_CODE','Plant Code']
    County = ['CNTYNAME,C,20','CNTYNAME','COUNTY','County']
    Zipcode = ['PLNTZIP,C,5','PLNTZIP','ZIP5','Zip']
    Sector= ['SECTOR_NAME','Sector Name']
    SectorNum = ['SECTOR_NUMBER','SECTOR','Sector']
    Lat = ['Latitude']
    Long = ['Longitude']
    
    #for cost dataframe
    Fuel_Type = ['FUEL_GROUP','Fuel_Group']
    Source = ['SUPPLIER','Supplier']
    Cost = ['FUEL_COST','Fuel_Cost']
    Month = ['MONTH','Month']
    
    #Monthly Fuel Use and Electricity Gen
    MonthKey = {1:'January',2:'February',3:'March',4:'April',
                5: 'May',6:'June', 7:'July',8:'August',
                9:'September',10:'October',11:'November',12:'December'}
    
    #Fuel Keys
    Fuel_Trans = {'Natural Gas': 'NG', 'Coal': 'COL', 'Petroleum':'DFO',
                  'Petroleum Coke': 'PC','Other Gas': 'OOG'}

    OM_col= ['Total Collection/Abatement O&M Expense','Total Disposal/Abatement O&M Expense',
             'Total Other O&M Expense','Total Revenues','FGD Feed Materials And Chemicals Costs',
             'FGD Labor And Supervision Costs','FGD Waste Disposal Costs',
             'FGD Maintenance Material And Other Costs','FGD Total Costs']

    file_df = pd.read_csv('Files to Read.csv',header=0)
    file_df.sort_values(by='Year', inplace=True, ascending=False)
    
    years = np.arange(min(file_df['Year']),max(file_df['Year']+1),1)
    
    master_df = pd.DataFrame()
    cost_df = pd.DataFrame()
    
    os.chdir(os.path.join('Sources'))

    for year in years:
        start = time.ctime()
        print('Start', start)
        files = file_df.loc[file_df['Year'] == year]

        GFDFile = files.loc[files['Type'] == 'GFD', 'File'].item()
        INFOFile =  files.loc[files['Type'] == 'INFO', 'File'].item()
        #OMFile = files.loc[files['Type'] == 'OM', 'File'].item()
        
        print()
        print('Gen and Fuel File')
        print(GFDFile)

        temp_df = pd.DataFrame()
    
        xls = pd.ExcelFile(GFDFile)
        sheets = xls.sheet_names
      
        df = xls.parse(sheets[0])
        df = Clean(df,IDNames)
        #df = df[df.index!='State-Fuel Level Increment']
        df = df[df['Plant Name']!='State-Fuel Level Increment']

        header = list(df.columns.values)
                    
        YearCol = list(set(YearNames) & set(header))
        UtilityCol = list(set(UtilityNames) & set(header))
        StateCol = list(set(StateNames) & set(header))
        PlCodeCol = list(set(IDNames) & set(header))
        MoverCol = list(set(MoverNames) & set(header))
        FuelCol = list(set(FuelNames) & set(header))
        ElecFuelCol = list(set(ElecFuelNames) & set(header))
        GenCol = list(set(GenNames) & set(header))
        NERCCol = list(set(NERCNames) & set(header))
        CensusCol = list(set(CensusNames) & set(header))

        
        temp_df['Year'] = df[YearCol.pop()]
        temp_df['Utility'] = df[UtilityCol[0]]
        temp_df['Plant State'] = df[StateCol[0]]
        temp_df['Plant Code'] = df[PlCodeCol[0]].astype(int)
        temp_df['NERC Region'] = df[NERCCol[0]]
        temp_df['Census Region'] = df[CensusCol[0]]
        temp_df['Primary Mover'] = df[MoverCol[0]]
        temp_df['AER Fuel Code'] = df[FuelCol[0]]
        temp_df['Fuel Consumption MWh'] = df[ElecFuelCol[0]]*MMBTU_MWH
        temp_df['Net Generation MWh'] = df[GenCol[0]]
        temp_df['Efficiency'] = (df[GenCol[0]]/(df[ElecFuelCol[0]]*MMBTU_MWH))
        
        print()
        print('INFO File')
        print(INFOFile)
        INFO_df = pd.read_csv(INFOFile,header=None)
        INFO_df = Clean(INFO_df,Plant_Codes)
        
        header = list(INFO_df.columns.values)
        
        CountyCol = list(set(County) & set(header))
        ZipCol = list(set(Zipcode) & set(header))
        PlCodeCol = list(set(Plant_Codes) & set(header))

        
        temp_info = pd.DataFrame()
        temp_info['County'] = INFO_df[CountyCol[0]].astype(str)
        temp_info['Zipcode'] = INFO_df[ZipCol[0]].astype(str)
        temp_info['Plant Code'] = INFO_df[PlCodeCol[0]].astype(int)
        
        temp_df = pd.merge(temp_df, temp_info, on='Plant Code', how='outer')
        
        #temp_df = temp_df[temp_df.index!='State-Fuel Level Increment']
        master_df = master_df.append(temp_df)        
        
        '''
        #checks to see if receipts exists
        if len(sheets)>3:
           
            #idx_s = header.index(('Elec_MMBtu January')|('ELEC_MMBTUS_JAN'))
            #idx_f = header.index('Netgen December')
            idx_s = 67 #from 2008-2015 Elec MMBtu Jan =67, NetGen Dec = 90
            idx_f = 90
            month_df = temp_df
            month_df = month_df.join(df.iloc[:,idx_s:idx_f])
            
            tdf = pd.DataFrame()
            if year == 2016:
                df= xls.parse(sheets[5])
            else:
                df= xls.parse(sheets[4])
            df=Clean(df,IDNames)
            
            header = list(df.columns.values)
            YearCol = list(set(YearNames) & set(header))
            MonthCol = list(set(Month) & set(header))
            PlCodeCol = list(set(IDNames) & set(header))
            FuelCol = list(set(Fuel_Type) & set(header))
            SourceCol = list(set(Source) & set(header))
            CostCol = list(set(Cost) & set(header))

            tdf['Year'] = df[YearCol.pop()]
            tdf['Month'] = df[MonthCol[0]]
            tdf['Plant Code'] = df[PlCodeCol[0]]
            tdf['Fuel Group'] = df[FuelCol[0]]
            tdf['Supplier'] = df[SourceCol[0]]
            tdf['Cost'] = df[CostCol[0]] # cents per MMBTU
            
            tdf = tdf.groupby(['Year','Plant Code','Fuel Group','Month'],as_index = False)['Cost'].sum()

            #Fills in cost_df with monthly fuel use, elect gen, and Fuel cost/kWh
            
            plants = list(set(month_df['Plant Code'].unique()) & set(tdf['Plant Code'].unique()))

            for plant in plants:
                sub_cost = tdf[tdf['Plant Code']==plant]
                sub_month = month_df[month_df['Plant Code']==plant]
                fuels = sub_cost['Fuel Group'].unique()
                for fuel in fuels:
                    AER_fuel = Fuel_Trans[fuel]

                    #sets range of months
                    month_s = min(sub_cost['Month'][sub_cost['Fuel Group']==fuel])
                    month_f = max(sub_cost['Month'][sub_cost['Fuel Group']==fuel])

                    #sets fuel type from months dataset
                    sub_fuel = sub_month[sub_month['AER Fuel Code']==AER_fuel]

                    frank = sub_fuel.groupby(['Plant Code','AER Fuel Code'],as_index = False).sum()
                
                    for mo in range(month_s,month_f):
                        #Fuel_con = ' '.join(('Elec_MMBtu',MonthKey[mo])) #MMBTU
                        #Elec_gen = ' '.join(('Netgen',MonthKey[mo])) #MWh
                        Fuel_con = 12+mo #9 = January Elec so 8+1(mo) = 9
                        Elec_gen = 24+mo
                                                
                        #finds index in Cost_df that matches fuel, plant and month
                        idx = tdf.loc[(tdf['Month'] == mo) & (tdf['Plant Code']==plant) & (tdf['Fuel Group']==fuel)].index.tolist()
                        
                        if len(idx)>0 and len(frank)>0: #only calculations for plants that have fuel receipts
                            idx = idx[0]
                            tdf.set_value(idx,'Use MMBtu', float(frank.get_value(0, Fuel_con, takeable=True)))
                            tdf.set_value(idx,'Elec Gen MWh', float(frank.get_value(0, Elec_gen, takeable=True)))

            tdf['Fuel Exp $']=tdf['Cost']*tdf['Use MMBtu']/100 #Cents per MMBtu * MMBtu
            tdf['Elec Pric c/kWh'] = tdf['Fuel Exp $']/tdf['Elec Gen MWh']/10
            cost_df = cost_df.append(tdf)  
        '''
        '''
        if OMFile:
            xls = pd.ExcelFile(OMFile)
            sheets = xls.sheet_names
            print(sheets)
        '''
    #print(master_df.head())
    
    cost_df.to_csv('Cost Dataset.csv')
    master_df.to_csv('Condensed Dataset.csv')

    os.chdir('..')
    end = time.ctime()
    print('End', end)


def Clean(df,keys):
    
    for key in keys:
        head_loc = np.where(df==key)[0]
        if len(head_loc)>0:
            head_loc = head_loc.item()
            break
            #idx = 'Plant Name'
        '''
        else:
            print('PLANT NAME')
            head_loc = np.where(df=='PLANT NAME')[0]
            head_loc= head_loc.item()
            #idx = 'PLANT NAME'
        '''
    df.columns = df.iloc[head_loc]
    todrop = np.arange(0,head_loc+1,1)
    df.drop(todrop,inplace=True)
    
    #df.set_index(idx,inplace=True) 
    df.columns = df.columns.str.replace('\n',' ')
    df.columns = df.columns.str.replace('  ',' ')
    mask = df.isin(['.'])   
    df = df.where(~mask, other=0)
    mask = df.isin([0])   
    df = df.where(~mask, other=np.nan)
    return(df)

def PopulationSet():
    print('Assembling Condensed Population Set')
    

    
    file_df = pd.read_csv('Files to Read.csv',header=0)
    file_df.sort_values(by='Year', inplace=True, ascending=False)
    file_df = file_df[file_df['Type']=='POP']
    years = np.arange(min(file_df['Year']),max(file_df['Year']+1),1)
    POP_df = pd.DataFrame()
    
    os.chdir(os.path.join('Sources'))
    
    ll_file = '2015_Gaz_counties_national.txt'
    ll_df = pd.read_table(ll_file,encoding = 'latin1')
    
    for year in years:
        tdf = pd.DataFrame()
        files = file_df.loc[file_df['Year'] == year]
        POPFile = files.loc[files['Type'] == 'POP', 'File'].item()
        
        tdf = pd.read_csv(POPFile,header=1,encoding = 'latin1')
        tdf['County'],tdf['Plant State'] = tdf['Geography'].str.split(', ').str
        tdf['Year']=year
        
        POP_df = POP_df.append(tdf[['Year','Id','Id2','County','Plant State','Estimate; Total']])
        
    POP_df = POP_df.merge(ll_df, left_on=['Id2'],right_on='GEOID', how='left')
    POP_df.to_csv('Population Set.csv')
    os.chdir('..')

def MetaAnalysis(df):
    #cols = df.columns
    cols = ['Fuel Consumption MWh', 'Net Generation MWh', 'Efficiency']
    for col in cols:
        try:
            print('*****',col,'*****')
            print(df[col].describe())
            df[col].plot.hist(bins = 30)
            plt.xlim(1000)
            plt.show()
            plt.close()

        except:
            pass
    utilities = df['Utility'].value_counts()
    print(utilities.describe())
    utilities = utilities[utilities>30]
    print(utilities.describe())
    utilities.plot.hist(bins = 30)

    
def SecondClean(df):
    eff_upper = 1
    eff_lower = 0
    trouble_df = df[df['Efficiency']>eff_upper]
    trouble_df.append(df[df['Efficiency']<eff_lower])
    df.dropna(axis=0, how = 'any',inplace =True)
    #trouble_df.append(df[df['Fuel Consumption MWh']<0])
    #trouble_df.append(df[df['Net Generation MWh']<0])
    SaveResultDF(trouble_df,'Trouble utilities')
    print(len(trouble_df), ' entries have been removed')
    
    df = df[df['Efficiency']<eff_upper]
    df = df[df['Efficiency']>eff_lower]
    df = df[df['Fuel Consumption MWh']>0]
    df = df[df['Net Generation MWh']>0]
    
    df['Zipcode'] = (df['Zipcode'].astype(str).str.zfill(5)).astype(str)
    return df

def FeatureEng(df):

    def Neighbors(dist):
        def GeoMile(lat):
            km_mile = 0.621   
            rad_lat = m.radians(lat)
            lat_mi = km_mile*(111132.954-559.822*m.cos(2*rad_lat) +1.175*m.cos(4*rad_lat))/1000
            long_mi = km_mile*(m.pi*6378137.0*m.cos(rad_lat)/(180*(1-0.00669437999014*m.sin(rad_lat)**2)**0.5))/1000
            return(lat_mi,long_mi)
            
        def FindNeighbors(x):    
            clat = x['Latitude']
            clong = x['Longitude']
            dlat = x['dlat']
            dlong = x['dlong']
            neighbors = tdf['Net Generation MWh'][(tdf['Latitude']<=clat+dlat) & (tdf['Latitude']>=clat-dlat) &
                         (tdf['Longitude']<=clong+dlong) & (tdf['Longitude']>=clong-dlong)]
            return neighbors.sum(),len(neighbors)
    
        #def PopDensity(x):
            

        tdf[['dlat','dlong']] = dist/tdf['Latitude'].apply(GeoMile).apply(pd.Series)
        years = df['Year'].unique()
        
        power_df = pd.DataFrame()
        for year in years:
            power_df=power_df.append(tdf[tdf['Year']==year].apply(FindNeighbors,axis=1).apply(pd.Series))
        power_df.columns =['{0:.0f} power'.format(dist),'{0:.0f} neighbors'.format(dist)]
        power_df[['Year','Plant Code']] = tdf[['Year','Plant Code']]
        
        
        
        return(power_df)
            
    tdf = df.groupby(['Year','Plant Code','Latitude','Longitude'],as_index=False).sum()

    for n in [5,10,25,50,100,150,200]:
        df = pd.merge(df, Neighbors(n), on=['Plant Code','Year'], how='left')
        break
        
    return df
        
def LoadSource(filename):
    os.chdir(os.path.join('Sources'))
    df = pd.read_csv('.'.join((filename,'csv')))
    os.chdir('..')
    return(df)
      
def SaveResultDF(df,file):
    os.chdir(os.path.join('Results'))
    df.to_csv('.'.join((file,'csv')),index=False)
    os.chdir('..')
    return

def YearlyDataSet(df):
    years = df['Year'].unique()
    
    for year in years:
        year_df = df[df['Year']== year]
        SaveResultDF(year_df, ('_'.join((str(year),'sub_master'))))
        

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
        'Zipcode'
        'County'
        'Latitude'
        'Longitude'
        'N power'
        'N neighbors'
        'Sector'
'''

#GenDataset()
#print(SecondClean(LoadSource('Condensed Dataset')))
#PopulationSet()
#print(LoadSource('Population Set'))

