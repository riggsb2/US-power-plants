#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 14:33:41 2018

@author: riggs
"""
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import time
import math as m
import calendar as cal
import sys

def GenDataset():
    print('Generating Master DataSet')
    
    #For master dataframe
    MMBTU_MWH = 0.29307
    IDNames = ['Plant Id', 'Plant Code','Plant ID', 'Plant_Id']
    UtilityNames = ['OPERATOR NAME','UTILITY_NAME','Utility Name','Operator Name','Operator_Name']
    YearNames = ['Year','YEAR','Year']
    PlantNames = ['Plant Name','Plant_Name']
    StateNames = ['State', 'Plant State','Plant_State']
    MoverNames = ['Reported Prime Mover','Reported_Prime_Mover']
    FuelNames = ['AER Fuel Type Code','Aer_Fuel_Type_Code']
    ElecFuelNames = ['Elec Fuel Consumption MMBtu','ELEC FUEL CONSUMPTION MMBTUS','Elec_Fuel_Consumption_MMBtus','Elec_Fuel_Consumption_MMBtu']
    GenNames = ['Net Generation (Megawatthours)','NET GENERATION (megawatthours)','Net_Generation_(MWH)']
    CensusNames = ['Census Region','Census_Region']
    NERCNames = ['NERC Region','Nerc_Region']
    
    #From 860 Reports
    Plant_Codes = ['PLNTCODE,N,5,0','PLNTCODE','PLANT_CODE','Plant Code',
                   'Plntcode_N_5_0','Plntcode','Plant_Code']
    County = ['Cntyname_C_20','Cntyname','County']
    Zipcode = ['Plntzip_C_5','Plntzip','Zips','Zip','Zip5']
    Sector= ['SECTOR_NAME','Sector Name','sector_name','Sector_Name']
    SectorNum = ['SECTOR_NUMBER','SECTOR','Sector']
    Lat = ['Latitude']
    Long = ['Longitude']
        
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
    loc_df = pd.DataFrame()

    os.chdir(os.path.join('Sources/In Use'))
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
        df = df[df['Plant_Name']!='State-Fuel Level Increment']
        
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
        temp_df['Plant_State'] = df[StateCol[0]]
        temp_df['Plant_Code'] = df[PlCodeCol[0]].astype(int)
        temp_df['NERC_Region'] = df[NERCCol[0]]
        temp_df['Census_Region'] = df[CensusCol[0]]
        temp_df['Primary_Mover'] = df[MoverCol[0]]
        temp_df['AER_Fuel_Code'] = df[FuelCol[0]]
        temp_df['Fuel_Consumption_MWh'] = df[ElecFuelCol[0]]*MMBTU_MWH
        temp_df['Net_Generation_MWh'] = df[GenCol[0]]
        #temp_df['Efficiency'] = (df[GenCol[0]]/(df[ElecFuelCol[0]]*MMBTU_MWH))

        idx_s = header.index('Elec_MMBtu_1')
        idx_f = header.index('Netgen_12')
        temp_df = temp_df.join(df.iloc[:,idx_s:idx_f])
        
        print()
        print('INFO File')
        print(INFOFile)
        INFO_df = pd.read_csv(INFOFile,header=None)
        INFO_df = Clean(INFO_df,Plant_Codes)
        
        header = list(INFO_df.columns.values)
        
        CountyCol = list(set(County) & set(header))
        ZipCol = list(set(Zipcode) & set(header))
        PlCodeCol = list(set(Plant_Codes) & set(header))
        LatCol = list(set(Lat) & set(header))
        LongCol = list(set(Long) & set(header))
        
        temp_info = pd.DataFrame()
        temp_info['County'] = INFO_df[CountyCol[0]].astype(str)
        temp_info['Zipcode'] = INFO_df[ZipCol[0]].astype(str)
        temp_info['Plant_Code'] = INFO_df[PlCodeCol[0]].astype(int)
        
        temp_df = pd.merge(temp_df, temp_info, on='Plant_Code', how='outer')

        if LatCol:
            t_loc_df = pd.DataFrame()
            t_loc_df['Plant_Code'] = INFO_df[PlCodeCol[0]].astype(int)
            t_loc_df['Latitude'] = INFO_df[LatCol[0]]
            t_loc_df['Longitude'] = INFO_df[LongCol[0]]
            loc_df = loc_df.append(t_loc_df)        

        
        for i in sheets:
            if 'Receipts' in i:
               
                cdf = Clean(xls.parse(i), IDNames)
                Cost_df = cdf[['Month','Plant_Id','Fuel_Group','Fuel_Cost']]
                Cost_df['Fuel_Cost'] = Cost_df['Fuel_Cost'].astype(float)
                Cost_df = Cost_df.groupby(['Plant_Id', 'Fuel_Group', 'Month'])['Fuel_Cost'].mean().unstack('Month')
                
                Cost_df.reset_index(level=['Plant_Id', 'Fuel_Group'],inplace=True)
                old_col = list(Cost_df.columns.values)
                col = old_col[:2]+['_'.join(('Fuel_Cost',str(x))) for x in old_col[2:]]
                Cost_df.rename(columns= dict(zip(old_col,col)), inplace = True)
                Cost_df['Fuel_Group'].replace(Fuel_Trans,inplace = True)
                Cost_df['Fuel_Cost_Mean']=Cost_df.iloc[:,2:].mean(axis=1)
                Cost_df['Fuel_Cost_StDev']=Cost_df.iloc[:,2:].std(axis=1)  
                Cost_df['Fuel_Cost_Var']=Cost_df.iloc[:,2:].var(axis=1)                

                #Cost values are in cents per million Btu (MMBtu)
                temp_df = temp_df.merge(Cost_df, left_on=['Plant_Code','AER_Fuel_Code'],
                                        right_on=['Plant_Id','Fuel_Group'], how = 'left')
                
                break
        
        master_df = master_df.append(temp_df)        
        
        '''
        if OMFile:
            xls = pd.ExcelFile(OMFile)
            sheets = xls.sheet_names
            print(sheets)
        '''
    loc_df.drop_duplicates(inplace = True)
    master_df = pd.merge(master_df, loc_df, on='Plant_Code', how='outer')

    os.chdir('..')
    master_df.to_csv('Compiled Dataset.csv', index=False)

    os.chdir('..')
    end = time.ctime()
    print('End', end)


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
    df.columns = df.columns.str.replace(',',' ')
    df.columns = df.columns.str.replace('  ',' ')
    df.columns = df.columns.str.title()
    df.columns = df.columns.str.replace(' ','_')
    df.columns = df.columns.str.replace('Megawatthours','MWH')
    df.columns = df.columns.str.replace('Mmbtu','MMBtu')
    df.columns = df.columns.str.replace('MMBtus','MMBtu')    
    
    MoAbb = dict((v,k) for k,v in enumerate(cal.month_abbr))
    MoName = dict((v,k) for k,v in enumerate(cal.month_name))
    Months = {**MoName, **MoAbb}
    
    replacement = dict()
    for col in df.columns.values:
        if col !='nan':
            try:
                s = col.split('_')  
            except:
                print('ERROR')
                print(col)
                print(df.columns.values)
                sys.exit()
            if s[-1] in Months.keys():
                s[-1] = str(Months[s[-1]])
                replacement[col] = '_'.join(s)
            
    df.rename(columns= replacement, inplace = True)
    
    mask = df.isin(['.'])   
    df = df.where(~mask, other=np.nan)
    mask = df.isin([' '])   
    df = df.where(~mask, other=np.nan)
    #mask = df.isin([0])   
    #df = df.where(~mask, other=np.nan)
    return(df)

def PopulationSet():
    print('Assembling Condensed Population Set')
        
    file_df = pd.read_csv('Files to Read.csv',header=0)
    file_df.sort_values(by='Year', inplace=True, ascending=False)
    file_df = file_df[file_df['Type']=='POP']
    years = np.arange(min(file_df['Year']),max(file_df['Year']+1),1)
    POP_df = pd.DataFrame()
    
    os.chdir(os.path.join('Sources/In Use'))
    
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
    POP_df.rename(columns={POP_df.columns.values[-1]:'INTPTLONG'},inplace=True)
    os.chdir('..')
    POP_df.to_csv('Population Set.csv', index = False)
    os.chdir('..')

def MetaAnalysis(df):
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
    
def FeatureEng(df):

    def Neighbors(dists):
        print('Calculating Neighbor Stats', time.ctime())
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
                neighbors = tdf['Net_Generation_MWh'][(tdf['Latitude']<=clat+dlat) & (tdf['Latitude']>=clat-dlat) &
                             (tdf['Longitude']<=clong+dlong) & (tdf['Longitude']>=clong-dlong)]
                population = py_df['Estimate; Total'][(py_df['INTPTLAT']<=clat+dlat) & (py_df['INTPTLAT']>=clat-dlat) &
                             (py_df['INTPTLONG']<=clong+dlong) & (py_df['INTPTLONG']>=clong-dlong)]
                results.extend([neighbors.sum(),len(neighbors), population.sum(),len(population)])
            return results
                
        
        #tdf[['dlat','dlong']] = dist/tdf['Latitude'].apply(GeoMile).apply(pd.Series)
        years = df['Year'].unique()
        power_df = pd.DataFrame()
        for year in years:
            print('Year: ',year, " start time ", time.ctime())
            py_df = pop_df[pop_df['Year']==year]
            power_df=power_df.append(tdf[tdf['Year']==year].apply(FindNeighbors,axis=1).apply(pd.Series)) 
        power_df.dropna(axis = 1, how = 'all', inplace = True) #for some reason power_df has Year, Utility, etc. all Nan so they need to be dropped
        
        col = []
        for dist in dists:
            col.extend(['{0:.0f} power'.format(dist),'{0:.0f} neighbors'.format(dist),
                        '{0:.0f} pop'.format(dist),'{0:.0f} counties'.format(dist)])
        power_df.columns = col
        power_df[['Year','Plant_Code']] = tdf[['Year','Plant_Code']]
        
        print('Neighbors Compiled: Resulting DF.head()')

        print(power_df.head())

        return(power_df)
        
    tdf = df.groupby(['Year','Plant_Code','Latitude','Longitude'],as_index=False).sum()
    pop_df = LoadSource('Population Set').groupby(['Year','INTPTLAT','INTPTLONG'],as_index=False).sum()
    
    dists = [5,10,25,50,100,150,200,300]
    df = pd.merge(df, Neighbors(dists), on=['Plant_Code','Year'], how='left')
    
    os.chdir(os.path.join('Sources'))
    df.to_csv('Engineered Features.csv', index=False)
    os.chdir('..')
    return()
        
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
#PopulationSet()
#print(LoadSource('Population Set').head())
#df = LoadSource('Compiled Dataset')
#FeatureEng(df)
df = LoadSource('Engineered Features')
for col in df.columns.values:
    print(col, df[col].isnull().sum()/len(df[col]))
