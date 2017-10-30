import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy import stats
import seaborn as sns

def Clean(df):

    head_loc = np.where(df=='Plant Name')[0]
    if head_loc:
        head_loc = head_loc.item()
        idx = 'Plant Name'
    else:
        head_loc = np.where(df=='PLANT NAME')[0]
        head_loc= head_loc.item()
        idx = 'PLANT NAME'
        
    df.columns = df.iloc[head_loc]
    todrop = np.arange(0,head_loc+1,1)
    df.drop(todrop,inplace=True)
    
    df.set_index(idx,inplace=True) 
    df.columns = df.columns.str.replace('\n',' ')
    df.columns = df.columns.str.replace('  ',' ')
    mask = df.isin(['.'])   
    df = df.where(~mask, other=0)
    mask = df.isin([0])   
    df = df.where(~mask, other=np.nan)
    return(df)

def StatePortfolio(df):
    years = df['Year'].unique()    

    portfolio = pd.DataFrame(columns = ['Year','Plant State'])
    for year in years:
        year_df = df[df['Year']==year]
        
        #Entire US per year        
        US_net = year_df['Net Generation MWh'].sum()
        tdf = pd.DataFrame(columns = ['Year','Plant State'])
        tdf.Year = pd.Series(year)
        tdf['Plant State'] = pd.Series('Entire US')
        
        fuel_types = year_df['AER Fuel Code'].unique()
        for fuel in fuel_types:
            if pd.isnull(fuel)==False:
                net_gen = year_df.loc[year_df['AER Fuel Code'] == fuel, 'Net Generation MWh'].sum()
                frac = net_gen/US_net
                tdf[fuel] = pd.Series(frac)
        portfolio = portfolio.append(tdf,ignore_index=True)        

    col = list(portfolio.columns.values)
    PS = col.index('Plant State')
    col.insert(0, col.pop(PS))
    y = col.index('Year')
    col.insert(0, col.pop(y))

    portfolio = portfolio[col]
    return(portfolio)

   
def PortfolioPlot(df):
    if not os.path.isdir(os.path.join('State Portfolios')):
        os.makedirs(os.path.join('State Portfolios'))
    os.chdir(os.path.join('State Portfolios'))
    
    states = df['Plant State'].unique()
    for state in states:
        if pd.isnull(state)==False:
            state_df = df[df['Plant State']==state]
            state_df = state_df.loc[pd.notnull(state_df['Year'])]
            state_df= state_df.loc[:, (state_df > 0.01).any()]
            fuels = list(state_df.columns.values)[2:]
            state_df['Year'] = state_df['Year'].astype(int)
            plt.clf()
            ax1 = state_df.plot.bar('Year',fuels,stacked = True)
            ax1.set_title(' '.join(('Generation Distribution in',state)))
            ax1.set_xlabel('Year')
            ax1.set_ylabel('Fraction Generated Energy')
            ax1.set_ylim(0,1)
            box = ax1.get_position()
            ax1.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            handles, labels = ax1.get_legend_handles_labels()
            ax1.legend(handles[::-1], labels[::-1],loc='center left', bbox_to_anchor=(1, 0.5), title = 'AER Fuel Type Code' )
            ax1.plot()
            fig1 = plt.gcf()
            fig1.savefig(' '.join((state,'portfolio.png')), facecolor = 'white',dpi=90, bbox_inches='tight')
            plt.close('all')

    os.chdir('..')
    return()
        
def MoverTrends(df,stat,min_pts=2):
    cols = ['Plant Name','Plant State','Primary Mover','AER Fuel Code','m','b','Rsq','Std Error']
    trend_df = pd.DataFrame()
    states = df['Plant State'].unique()
    for state in states:
        state_df = df[df['Plant State']==state]
        movers = state_df['Primary Mover'].unique()
        for mover in movers:
            move_df = state_df[state_df['Primary Mover']==mover]
            fuels = move_df['AER Fuel Code'].unique()
            for fuel in fuels:            
                plants = move_df.index.unique()
                for plant in plants:
                    plant_df = move_df[move_df.index==plant]
                    x = plant_df['Year'].values
                    y = plant_df[stat].values
                    mask = ~pd.isnull(x) & ~pd.isnull(y)
                    x = x[mask]
                    y = y[mask]
                    if len(plant_df['Year'].unique())>=min_pts and len(plant_df[stat].unique())>=min_pts:
                        m, b, r_value, p_value, std_err = stats.linregress(x,y)
                        tdf = pd.DataFrame([[plant,state,mover,fuel,m,b,r_value**2,std_err]],columns = cols)
                        trend_df=trend_df.append(tdf,ignore_index=True)
    return(trend_df)

def TrendBoxplot(df,name, Rsq_lim = 0,lower=-100,upper=100):
    df = df[df['Rsq']>=Rsq_lim]
    df = df[df['m']<upper]
    df = df[df['m']>lower]

    col = df['Primary Mover'].unique()
    max_idx = len(df)
    idx = np.arange(1,max_idx,1)
    mover_df = pd.DataFrame(index = idx)

    stat_df = pd.DataFrame(index = col,columns = ['Median','StDev'])
   
    if not os.path.isdir(os.path.join(' '.join((name, 'Trends')))):
        os.makedirs(os.path.join(' '.join((name, 'Trends'))))
    os.chdir(os.path.join(' '.join((name, 'Trends'))))
    
    for mover in col:
        if type(mover)==str:
            tdf = df.loc[df['Primary Mover']==mover]
            fuels = tdf['AER Fuel Code'].unique()  
            moverfuel_df = pd.DataFrame(index = idx)
            if len(fuels) >1:
                for fuel in fuels:
                    if type(fuel) ==str:
                        moverfuel = fuel
                        moverfuel_df[moverfuel] = tdf.loc[tdf['AER Fuel Code']==fuel,'m']
                        median = moverfuel_df[moverfuel].median(skipna=True)
                        stdev = moverfuel_df[moverfuel].std(skipna=True)
                        stat_df.ix[moverfuel]=[median,stdev]
                moverfuel_df.dropna(1,how='all',inplace=True)
                
                if len(list(moverfuel_df.columns.values))>0:
                    sns.boxplot(moverfuel_df)
                    plt.title(' '.join((mover,'trend Box Plot')))
                    plt.xlabel('Fuel Type')
                    plt.ylabel('fractional change/year')
                    plt.savefig(''.join((mover,'.png')), facecolor = 'white',dpi=90, bbox_inches='tight')
                    plt.close()
                mover_df[mover] = df.loc[df['Primary Mover']==mover,'m']
                median = mover_df[mover].median(skipna=True)
                stdev = mover_df[mover].std(skipna=True)
                stat_df.ix[mover]=[median,stdev]
            else:
                mover_df[mover] = df.loc[df['Primary Mover']==mover,'m']
                median = mover_df[mover].median(skipna=True)
                stdev = mover_df[mover].std(skipna=True)
                stat_df.ix[mover]=[median,stdev]

    mover_df.sort_index(inplace=True)
    mover_df.dropna(1,how='all',inplace=True)
    sns.boxplot(mover_df)
    plt.title('Mover trends boxplot')
    plt.xlabel('Mover Type')
    plt.ylabel('fractional change/year')
    stat_df.to_csv(' '.join((name, ' summary statistics.csv')),delimiter = ',')
    plt.savefig(' '.join((name, ' Trend Statistics.png')), facecolor = 'white',dpi=90, bbox_inches='tight')
    plt.close()
        
    os.chdir('..')
    return()
    
MMBTU_MWH = 0.29307
IDNames = ['Plant Id', 'Plant Code','Plant ID']
YearNames = ['Year','YEAR']
PlantNames = ['Plant Name']
StateNames = ['State', 'Plant State']
MoverNames = ['Reported Prime Mover']
FuelNames = ['AER Fuel Type Code']
ElecFuelNames = ['Elec Fuel Consumption MMBtu','ELEC FUEL CONSUMPTION MMBTUS']
GenNames = ['Net Generation (Megawatthours)','NET GENERATION (megawatthours)']
xls = pd.ExcelFile('Files to Read.xlsx')
file_df = xls.parse(header =0)
file_df.sort_values(by='Year', inplace=True, ascending=False)

years = np.arange(min(file_df['Year']),max(file_df['Year']+1),1)

master_df = pd.DataFrame()

for year in years:
    files = file_df.loc[file_df['Year'] == year] 
    GFDFile = files.loc[files['Type'] == 'GFD', 'File'].item()

    temp_df = pd.DataFrame()

    xls = pd.ExcelFile(GFDFile)
    sheets = xls.sheet_names
    
    df = xls.parse(sheets[0])
    df = Clean(df)    
    header = list(df.columns.values)
    YearCol = list(set(YearNames) & set(header))
    StateCol = list(set(StateNames) & set(header))
    PlCodeCol = list(set(IDNames) & set(header))
    MoverCol = list(set(MoverNames) & set(header))
    FuelCol = list(set(FuelNames) & set(header))
    ElecFuelCol = list(set(ElecFuelNames) & set(header))
    GenCol = list(set(GenNames) & set(header))
    temp_df['Year'] = df[YearCol.pop()]
    temp_df['Plant State'] = df[StateCol[0]]
    temp_df['Plant Code'] = df[PlCodeCol[0]]
    temp_df['Primary Mover'] = df[MoverCol[0]]
    temp_df['AER Fuel Code'] = df[FuelCol[0]]
    temp_df['Net Generation MWh'] = df[GenCol[0]]
    temp_df['Efficiency'] = (df[GenCol[0]]/(df[ElecFuelCol[0]]*MMBTU_MWH))
    
    temp_df = temp_df[temp_df.index!='State-Fuel Level Increment']
    master_df = master_df.append(temp_df)

if not os.path.isdir(os.path.join('Slimmed Results')):
    os.makedirs(os.path.join('Slimmed Results'))
os.chdir(os.path.join('Slimmed Results'))

print(master_df.head())
Efftrends = MoverTrends(master_df,'Efficiency',5)
Efftrends.to_csv('Efficiency Trend df.csv',delimiter = ',')
TrendBoxplot(Efftrends,'Efficiency',0.7,-0.25,0.25)

EnergyPort = StatePortfolio(master_df)
EnergyPort.to_csv('State distribution df.csv',delimiter = ',')
PortfolioPlot(EnergyPort)

