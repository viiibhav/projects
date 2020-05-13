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
    if div in ['Atlantic', 'Central', 'Southeast']:
        teams.at[team, 'Conference'] = 'Eastern'
    else:
        teams.at[team, 'Conference'] = 'Western'


team_div = teams[['Team Code', 'Division', 'TeamID']]

# Add team divisions in the main dataframe
for year in years[:-3]:
    # Add Visitor/Neutral Division
    df[year] = df[year].merge(team_div, left_on='Visitor/Neutral', right_on='Team')
    df[year].rename(columns={'Division': 'V/N Division', 'Team Code': 'V/N Code',
                             'TeamID': 'V/N ID'}, inplace=True)

    # Add Home/Visitor Division
    df[year] = df[year].merge(team_div, left_on='Home/Neutral', right_on='Team')
    df[year].rename(columns={'Division': 'H/N Division', 'Team Code': 'H/N Code',
                             'TeamID': 'H/N ID'}, inplace=True)
    
    # Sort by Date
    df[year].sort_values(by=['Date'], inplace=True)
    
    # Reset index that is now jumbled during the merging
    df[year].reset_index(drop=True, inplace=True)


team_div_code = team_div
team_div_code.reset_index(inplace=True)
team_div_code.set_index('Team Code', drop=True, inplace=True)

for year in years[-3:]:
    # Add Visitor/Neutral Division
    df[year] = df[year].merge(team_div_code, left_on='Visitor/Neutral', right_on='Team Code')
    df[year].rename(columns={'Division': 'V/N Division', 'Visitor/Neutral': 'V/N Code',
                              'Team': 'Visitor/Neutral', 'TeamID': 'V/N ID'}, inplace=True)

    # Add Home/Visitor Division
    df[year] = df[year].merge(team_div_code, left_on='Home/Neutral', right_on='Team Code')
    df[year].rename(columns={'Division': 'H/N Division', 'Home/Neutral': 'H/N Code',
                              'Team': 'Home/Neutral', 'TeamID': 'H/N ID'}, inplace=True)
    
    # Sort by Date
    df[year].sort_values(by=['Date'], inplace=True)
    
    # Reset index that is now jumbled during the merging
    df[year].reset_index(drop=True, inplace=True)
