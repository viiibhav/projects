# -*- coding: utf-8 -*-
"""
Created on Sat May  2 19:00:21 2020

@author: vdabadgh
"""

import numpy as np
import pandas as pd

filename = 'nba_seasons\leagues_NBA_{}_games_games.csv'

cols = [0, 3, 4, 5, 6]
names = ['Date', 'Visitor/Neutral', 'V/N points', 'Home/Neutral', 'H/N points']
years = np.arange(2005, 2019)

df = {year: pd.DataFrame({name: [] for name in names}) for year in years}
for year in years[:-3]:
    df[year] = pd.read_csv(filename.format(year), header=0, usecols=cols,
                           names=names, keep_default_na=False)
    df[year]['Date'] = pd.to_datetime(df[year]['Date']).dt.date

# Seasons 2016 -- 2018 have different formats
cols = [0, 1, 2, 3, 4]
for year in years[-3:]:
    df[year] = pd.read_csv(filename.format(year), header=0, nrows=1230,
                           usecols=cols, names=names, keep_default_na=False)
    df[year]['Date'] = pd.to_datetime(df[year]['Date']).dt.date

# List of teams
teams = pd.read_csv(filename.format(2018), header=None, skiprows=1, nrows=35,
                    usecols=[8, 9, 10], names=['Team', 'Team Code', 'TeamID'],
                    index_col='Team')

# Team Divisions
divisions = {'Atlantic': ['Boston Celtics', 'Brooklyn Nets', 'New Jersey Nets',
                          'New York Knicks', 'Philadelphia 76ers', 'Toronto Raptors'],
             'Central': ['Chicago Bulls', 'Cleveland Cavaliers', 'Detroit Pistons',
                         'Indiana Pacers', 'Milwaukee Bucks'],
             'Southeast': ['Atlanta Hawks', 'Charlotte Hornets', 'Charlotte Bobcats',
                           'Miami Heat', 'Orlando Magic', 'Washington Wizards'],
             'Northwest': ['Denver Nuggets', 'Minnesota Timberwolves', 'Oklahoma City Thunder',
                           'Portland Trail Blazers', 'Utah Jazz', 'Seattle SuperSonics'],
             'Pacific': ['Golden State Warriors', 'Los Angeles Clippers', 'Los Angeles Lakers',
                         'Phoenix Suns', 'Sacramento Kings'],
             'Southwest': ['Dallas Mavericks', 'Houston Rockets', 'Memphis Grizzlies',
                           'New Orleans Pelicans', 'New Orleans Hornets',
                           'New Orleans/Oklahoma City Hornets', 'San Antonio Spurs']}

teams['Conference'] = ''
teams['Division'] = ''
for team in teams.index:
    div, = [key for key, value in divisions.items() if team in value]
    teams.at[team, 'Division'] = div
    eastern = ['Atlantic', 'Central', 'Southeast']
    teams.at[team, 'Conference'] = 'Eastern' if div in eastern else 'Western'


# team_div = teams[['Team Code', 'Division', 'TeamID', 'Conference']]

# Add team divisions in the main dataframe
for year in years[:-3]:
    # Add Visitor/Neutral Division
    df[year] = df[year].merge(teams, left_on='Visitor/Neutral', right_on='Team')
    df[year].rename(columns={'Division': 'V/N Division', 'Team Code': 'V/N Code',
                             'TeamID': 'V/N ID', 'Conference': 'V/N Conference'}, inplace=True)

    # Add Home/Visitor Division
    df[year] = df[year].merge(teams, left_on='Home/Neutral', right_on='Team')
    df[year].rename(columns={'Division': 'H/N Division', 'Team Code': 'H/N Code',
                             'TeamID': 'H/N ID', 'Conference': 'H/N Conference'}, inplace=True)
    
    # Sort by Date
    df[year].sort_values(by=['Date'], inplace=True)
    
    # Reset index that is now jumbled during the merging
    df[year].reset_index(drop=True, inplace=True)


teams_by_code = teams.copy()
teams_by_code.reset_index(inplace=True)
teams_by_code.set_index('Team Code', drop=True, inplace=True)

for year in years[-3:]:
    # Add Visitor/Neutral Division
    df[year] = df[year].merge(teams_by_code, left_on='Visitor/Neutral', right_on='Team Code')
    df[year].rename(columns={'Division': 'V/N Division', 'Visitor/Neutral': 'V/N Code',
                              'Team': 'Visitor/Neutral', 'TeamID': 'V/N ID',
                              'Conference': 'V/N Conference'}, inplace=True)

    # Add Home/Visitor Division
    df[year] = df[year].merge(teams_by_code, left_on='Home/Neutral', right_on='Team Code')
    df[year].rename(columns={'Division': 'H/N Division', 'Home/Neutral': 'H/N Code',
                              'Team': 'Home/Neutral', 'TeamID': 'H/N ID',
                              'Conference': 'H/N Conference'}, inplace=True)
    
    # Sort by Date
    df[year].sort_values(by=['Date'], inplace=True)
    
    # Reset index that is now jumbled during the merging
    df[year].reset_index(drop=True, inplace=True)


# Label games
for year in years:
    if year != 2012:
        df[year]['label'] = ''
        df[year]['label'] = df[year].apply(lambda x: 'OOC' if x['V/N Conference'] != x['H/N Conference']
                                    else 'DIV' if x['V/N Division'] == x['H/N Division']
                                    else 'ICOD3', axis=1)
    
    
        g = df[year].groupby(by=['Visitor/Neutral', 'Home/Neutral'])
        for t1 in df[year]['Visitor/Neutral'].unique():
            for t2 in df[year]['Visitor/Neutral'].unique():
                if t2 != t1:
                    g1 = g.get_group((t1, t2))
                    g2 = g.get_group((t2, t1))
                    icod, = 'ICOD3' == g1['label'].unique()
                    if icod:
                        ngames = len(g1) + len(g2)
                        if ngames == 4:
                            idxs = [idx for idx in g1.index] + [idx for idx in g2.index]
                            for idx in idxs:
                                df[year].at[idx, 'label'] = 'ICOD4'
                            

# Playoffs
wl_table = {}
for year in years:
    wl_table[year] = pd.DataFrame({'Team': []})
    wl_table[year]['Team'] = df[year]['Visitor/Neutral'].unique()
    wl_table[year] = wl_table[year].merge(teams[['Conference']], right_index=True, left_on='Team')
    wl_table[year]['Wins'] = 0
    wl_table[year].set_index('Team', drop=True, inplace=True)
    
    # Get win-loss
    for index, row in df[year].iterrows():
        if row['V/N points'] > row['H/N points']:
            wl_table[year].at[row['Visitor/Neutral'], 'Wins'] += 1
        else:
            wl_table[year].at[row['Home/Neutral'], 'Wins'] += 1

playoffs = {year: {conf: [] for conf in teams['Conference'].unique()} for year in years}
for year in years:
    g = wl_table[year].groupby(by='Conference')
    for conf in teams['Conference'].unique():
        h = g.get_group(conf)
        h = h.sort_values(by='Wins', ascending=False)
        playoffs[year][conf] = list(h.head(8).index)