# -*- coding: utf-8 -*-
"""
Created on Sat May  2 19:00:21 2020

@author: vdabadgh
"""

import numpy as np
import pandas as pd

filename = 'nba_seasons\leagues_NBA_{}_games_games.csv'
cols = [0, 3, 4, 5, 6, 15, 17]
names = ['Date', 'Visitor/Neutral', 'V/N points', 'Home/Neutral', 'H/N points', 'V/N ID', 'H/N ID']
years = np.arange(2001, 2019)
df = {year: pd.DataFrame({name: [] for name in names}) for year in years}
for year in years[:-3]:
    df[year] = pd.read_csv(filename.format(year), header=0, usecols=cols,
                           names=names, keep_default_na=False)

cols = [0, 1, 2, 3, 4, 15, 17]
teams = {}
for year in years[-3:]:
    df[year] = pd.read_csv(filename.format(year), header=0, nrows=1230,
                           usecols=cols, names=names, keep_default_na=False)
    teams[year] = pd.read_csv(filename.format(year), header=None, skiprows=1, nrows=30,
                              usecols=[8, 9, 10],
                              names=['Team', 'Team Code', 'TeamID'])