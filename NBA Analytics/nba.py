# -*- coding: utf-8 -*-
"""
Created on Sat May  2 19:00:21 2020

@author: vdabadgh
"""

import numpy as np
import pandas as pd

filename = 'nba_seasons\leagues_NBA_{}_games_games.csv'
cols = [0, 1, 3, 4, 5, 6, 7, 15, 17]
names = ['Date', 'Start (ET)', 'Visitor/Neutral', 'V/N points',
         'Home/Neutral', 'H/N points', 'OT', 'V/N ID', 'H/N ID']
years = np.arange(2001, 2019)
df = pd.DataFrame({name: [] for name in names})
for year in years:
    df1 = pd.read_csv(filename.format(year), usecols=cols, names=names)
    df = pd.concat([df, df1])