#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 10:54:40 2018

@author: riggs
"""

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
