# -*- coding: utf-8 -*-
"""
Created on Thu Aug  9 11:31:11 2018

@author: sdabadghao
"""

import pandas
#import numpy as np

nhl = pandas.read_excel('Playoff_results_NHL_NBA_NFL.xlsx',sheet_name='NHL_w13', usecols='N:S', names=["season", "match_ID", "winner", "loser", "winner_score", "loser_score"], nrows=255)
nba = pandas.read_excel('Playoff_results_NHL_NBA_NFL.xlsx',sheet_name='NBA', usecols='O:T', names=["season", "match_ID", "winner", "loser", "winner_score", "loser_score"], nrows=270)
nfl = pandas.read_excel('Playoff_results_NHL_NBA_NFL.xlsx',sheet_name='NFL_w01', usecols='O:R', names=["season", "match_ID", "winner", "loser"], nrows=187)


#####
#  NHL
#####

ratings_wins = pandas.read_excel('nhl_allratings.xlsx','wins',index_col='team')
ratings_massey = pandas.read_excel('nhl_allratings.xlsx','massey',index_col='team')
ratings_colley = pandas.read_excel('nhl_allratings.xlsx','colley',index_col='team')
ratings_elo = pandas.read_excel('nhl_allratings.xlsx','elo',index_col='team')
ratings_zero1 = pandas.read_excel('nhl_allratings.xlsx','zero1',index_col='team')
ratings_alpha2 = pandas.read_excel('nhl_allratings.xlsx','alpha2',index_col='team')
ratings_alpha3 = pandas.read_excel('nhl_allratings.xlsx','alpha3',index_col='team')
ratings_alpha4 = pandas.read_excel('nhl_allratings.xlsx','alpha4',index_col='team')
ratings_alpha5 = pandas.read_excel('nhl_allratings.xlsx','alpha5',index_col='team')
ratings_alpha7 = pandas.read_excel('nhl_allratings.xlsx','alpha7',index_col='team')
ratings_alpha10 = pandas.read_excel('nhl_allratings.xlsx','alpha10',index_col='team')
ratings_alpha20 = pandas.read_excel('nhl_allratings.xlsx','alpha20',index_col='team')
ratings_alpha100 = pandas.read_excel('nhl_allratings.xlsx','alpha100',index_col='team')
ratings_alpha1000 = pandas.read_excel('nhl_allratings.xlsx','alpha1000',index_col='team')

nhl['wins']=0
nhl['massey']=0
nhl['colley']=0
nhl['elo']=0
nhl['zero1']=0
nhl['alpha2']=0
nhl['alpha3']=0
nhl['alpha4']=0
nhl['alpha5']=0
nhl['alpha7']=0
nhl['alpha10']=0
nhl['alpha20']=0
nhl['alpha100']=0
nhl['alpha1000']=0
nhl['correct']=1

for idx in range(len(nhl)):
    nhl.loc[idx,'wins'] = 1 if (ratings_wins.loc[nhl.loc[idx,'winner'],nhl.loc[idx,'season']] > ratings_wins.loc[nhl.loc[idx,'loser'],nhl.loc[idx,'season']]) else 0
    nhl.loc[idx,'massey'] = 1 if (ratings_massey.loc[nhl.loc[idx,'winner'],nhl.loc[idx,'season']] > ratings_massey.loc[nhl.loc[idx,'loser'],nhl.loc[idx,'season']]) else 0
    nhl.loc[idx,'colley'] = 1 if (ratings_colley.loc[nhl.loc[idx,'winner'],nhl.loc[idx,'season']] > ratings_colley.loc[nhl.loc[idx,'loser'],nhl.loc[idx,'season']]) else 0
    nhl.loc[idx,'elo'] = 1 if (ratings_elo.loc[nhl.loc[idx,'winner'],nhl.loc[idx,'season']] > ratings_elo.loc[nhl.loc[idx,'loser'],nhl.loc[idx,'season']]) else 0
    nhl.loc[idx,'zero1'] = 1 if (ratings_zero1.loc[nhl.loc[idx,'winner'],nhl.loc[idx,'season']] > ratings_zero1.loc[nhl.loc[idx,'loser'],nhl.loc[idx,'season']]) else 0
    nhl.loc[idx,'alpha2'] = 1 if (ratings_alpha2.loc[nhl.loc[idx,'winner'],nhl.loc[idx,'season']] > ratings_alpha2.loc[nhl.loc[idx,'loser'],nhl.loc[idx,'season']]) else 0
    nhl.loc[idx,'alpha3'] = 1 if (ratings_alpha3.loc[nhl.loc[idx,'winner'],nhl.loc[idx,'season']] > ratings_alpha3.loc[nhl.loc[idx,'loser'],nhl.loc[idx,'season']]) else 0
    nhl.loc[idx,'alpha4'] = 1 if (ratings_alpha4.loc[nhl.loc[idx,'winner'],nhl.loc[idx,'season']] > ratings_alpha4.loc[nhl.loc[idx,'loser'],nhl.loc[idx,'season']]) else 0
    nhl.loc[idx,'alpha5'] = 1 if (ratings_alpha5.loc[nhl.loc[idx,'winner'],nhl.loc[idx,'season']] > ratings_alpha5.loc[nhl.loc[idx,'loser'],nhl.loc[idx,'season']]) else 0
    nhl.loc[idx,'alpha7'] = 1 if (ratings_alpha7.loc[nhl.loc[idx,'winner'],nhl.loc[idx,'season']] > ratings_alpha7.loc[nhl.loc[idx,'loser'],nhl.loc[idx,'season']]) else 0
    nhl.loc[idx,'alpha10'] = 1 if (ratings_alpha10.loc[nhl.loc[idx,'winner'],nhl.loc[idx,'season']] > ratings_alpha10.loc[nhl.loc[idx,'loser'],nhl.loc[idx,'season']]) else 0
    nhl.loc[idx,'alpha20'] = 1 if (ratings_alpha20.loc[nhl.loc[idx,'winner'],nhl.loc[idx,'season']] > ratings_alpha20.loc[nhl.loc[idx,'loser'],nhl.loc[idx,'season']]) else 0
    nhl.loc[idx,'alpha100'] = 1 if (ratings_alpha100.loc[nhl.loc[idx,'winner'],nhl.loc[idx,'season']] > ratings_alpha100.loc[nhl.loc[idx,'loser'],nhl.loc[idx,'season']]) else 0
    nhl.loc[idx,'alpha1000'] = 1 if (ratings_alpha1000.loc[nhl.loc[idx,'winner'],nhl.loc[idx,'season']] > ratings_alpha1000.loc[nhl.loc[idx,'loser'],nhl.loc[idx,'season']]) else 0

score_nhl = nhl.filter(['season','wins','massey','colley','elo','zero1','alpha2','alpha3','alpha4','alpha5','alpha7','alpha10','alpha20','alpha100','alpha1000'])

pivotscore_nhl = score_nhl.pivot_table(index=['season'], aggfunc=sum)
#
##from mlxtend.evaluate import mcnemar,mcnemar_table
###import numpy as np
##
##
##
##test = mcnemar_table(y_target=nhl.correct,y_model1=nhl.wins,y_model2=nhl.zero1)
##chi2, p = mcnemar(ary=test, corrected=True)



######
#  NBA
######

ratings_wins = pandas.read_excel('nba_allratings.xlsx','wins',index_col='team')
ratings_massey = pandas.read_excel('nba_allratings.xlsx','massey',index_col='team')
ratings_colley = pandas.read_excel('nba_allratings.xlsx','colley',index_col='team')
ratings_elo = pandas.read_excel('nba_allratings.xlsx','elo',index_col='team')
ratings_zero1 = pandas.read_excel('nba_allratings.xlsx','zero1',index_col='team')
ratings_alpha2 = pandas.read_excel('nba_allratings.xlsx','alpha2',index_col='team')
ratings_alpha3 = pandas.read_excel('nba_allratings.xlsx','alpha3',index_col='team')
ratings_alpha4 = pandas.read_excel('nba_allratings.xlsx','alpha4',index_col='team')
ratings_alpha5 = pandas.read_excel('nba_allratings.xlsx','alpha5',index_col='team')
ratings_alpha7 = pandas.read_excel('nba_allratings.xlsx','alpha7',index_col='team')
ratings_alpha10 = pandas.read_excel('nba_allratings.xlsx','alpha10',index_col='team')
ratings_alpha20 = pandas.read_excel('nba_allratings.xlsx','alpha20',index_col='team')
ratings_alpha100 = pandas.read_excel('nba_allratings.xlsx','alpha100',index_col='team')
ratings_alpha1000 = pandas.read_excel('nba_allratings.xlsx','alpha1000',index_col='team')

nba['wins']=0
nba['massey']=0
nba['colley']=0
nba['elo']=0
nba['zero1']=0
nba['alpha2']=0
nba['alpha3']=0
nba['alpha4']=0
nba['alpha5']=0
nba['alpha7']=0
nba['alpha10']=0
nba['alpha20']=0
nba['alpha100']=0
nba['alpha1000']=0
nba['correct']=1

for idx in range(len(nba)):
    nba.loc[idx,'wins'] = 1 if (ratings_wins.loc[nba.loc[idx,'winner'],nba.loc[idx,'season']] > ratings_wins.loc[nba.loc[idx,'loser'],nba.loc[idx,'season']]) else 0
    nba.loc[idx,'massey'] = 1 if (ratings_massey.loc[nba.loc[idx,'winner'],nba.loc[idx,'season']] > ratings_massey.loc[nba.loc[idx,'loser'],nba.loc[idx,'season']]) else 0
    nba.loc[idx,'colley'] = 1 if (ratings_colley.loc[nba.loc[idx,'winner'],nba.loc[idx,'season']] > ratings_colley.loc[nba.loc[idx,'loser'],nba.loc[idx,'season']]) else 0
    nba.loc[idx,'elo'] = 1 if (ratings_elo.loc[nba.loc[idx,'winner'],nba.loc[idx,'season']] > ratings_elo.loc[nba.loc[idx,'loser'],nba.loc[idx,'season']]) else 0
    nba.loc[idx,'zero1'] = 1 if (ratings_zero1.loc[nba.loc[idx,'winner'],nba.loc[idx,'season']] > ratings_zero1.loc[nba.loc[idx,'loser'],nba.loc[idx,'season']]) else 0
    nba.loc[idx,'alpha2'] = 1 if (ratings_alpha2.loc[nba.loc[idx,'winner'],nba.loc[idx,'season']] > ratings_alpha2.loc[nba.loc[idx,'loser'],nba.loc[idx,'season']]) else 0
    nba.loc[idx,'alpha3'] = 1 if (ratings_alpha3.loc[nba.loc[idx,'winner'],nba.loc[idx,'season']] > ratings_alpha3.loc[nba.loc[idx,'loser'],nba.loc[idx,'season']]) else 0
    nba.loc[idx,'alpha4'] = 1 if (ratings_alpha4.loc[nba.loc[idx,'winner'],nba.loc[idx,'season']] > ratings_alpha4.loc[nba.loc[idx,'loser'],nba.loc[idx,'season']]) else 0
    nba.loc[idx,'alpha5'] = 1 if (ratings_alpha5.loc[nba.loc[idx,'winner'],nba.loc[idx,'season']] > ratings_alpha5.loc[nba.loc[idx,'loser'],nba.loc[idx,'season']]) else 0
    nba.loc[idx,'alpha7'] = 1 if (ratings_alpha7.loc[nba.loc[idx,'winner'],nba.loc[idx,'season']] > ratings_alpha7.loc[nba.loc[idx,'loser'],nba.loc[idx,'season']]) else 0
    nba.loc[idx,'alpha10'] = 1 if (ratings_alpha10.loc[nba.loc[idx,'winner'],nba.loc[idx,'season']] > ratings_alpha10.loc[nba.loc[idx,'loser'],nba.loc[idx,'season']]) else 0
    nba.loc[idx,'alpha20'] = 1 if (ratings_alpha20.loc[nba.loc[idx,'winner'],nba.loc[idx,'season']] > ratings_alpha20.loc[nba.loc[idx,'loser'],nba.loc[idx,'season']]) else 0
    nba.loc[idx,'alpha100'] = 1 if (ratings_alpha100.loc[nba.loc[idx,'winner'],nba.loc[idx,'season']] > ratings_alpha100.loc[nba.loc[idx,'loser'],nba.loc[idx,'season']]) else 0
    nba.loc[idx,'alpha1000'] = 1 if (ratings_alpha1000.loc[nba.loc[idx,'winner'],nba.loc[idx,'season']] > ratings_alpha1000.loc[nba.loc[idx,'loser'],nba.loc[idx,'season']]) else 0

score_nba = nba.filter(['season','wins','massey','colley','elo','zero1','alpha2','alpha3','alpha4','alpha5','alpha7','alpha10','alpha20','alpha100','alpha1000'])

pivotscore_nba = score_nba.pivot_table(index=['season'], aggfunc=sum)




######
#  NFL
######

ratings_wins = pandas.read_excel('nfl_allratings.xlsx','wins',index_col='team')
ratings_massey = pandas.read_excel('nfl_allratings.xlsx','massey',index_col='team')
ratings_colley = pandas.read_excel('nfl_allratings.xlsx','colley',index_col='team')
ratings_elo = pandas.read_excel('nfl_allratings.xlsx','elo',index_col='team')
ratings_zero1 = pandas.read_excel('nfl_allratings.xlsx','zero1',index_col='team')
ratings_alpha2 = pandas.read_excel('nfl_allratings.xlsx','alpha2',index_col='team')
ratings_alpha3 = pandas.read_excel('nfl_allratings.xlsx','alpha3',index_col='team')
ratings_alpha4 = pandas.read_excel('nfl_allratings.xlsx','alpha4',index_col='team')
ratings_alpha5 = pandas.read_excel('nfl_allratings.xlsx','alpha5',index_col='team')
ratings_alpha7 = pandas.read_excel('nfl_allratings.xlsx','alpha7',index_col='team')
ratings_alpha10 = pandas.read_excel('nfl_allratings.xlsx','alpha10',index_col='team')
ratings_alpha20 = pandas.read_excel('nfl_allratings.xlsx','alpha20',index_col='team')
ratings_alpha100 = pandas.read_excel('nfl_allratings.xlsx','alpha100',index_col='team')
ratings_alpha1000 = pandas.read_excel('nfl_allratings.xlsx','alpha1000',index_col='team')

nfl['wins']=0
nfl['massey']=0
nfl['colley']=0
nfl['elo']=0
nfl['zero1']=0
nfl['alpha2']=0
nfl['alpha3']=0
nfl['alpha4']=0
nfl['alpha5']=0
nfl['alpha7']=0
nfl['alpha10']=0
nfl['alpha20']=0
nfl['alpha100']=0
nfl['alpha1000']=0
nfl['correct']=1

for idx in range(len(nfl)):
    nfl.loc[idx,'wins'] = 1 if (ratings_wins.loc[nfl.loc[idx,'winner'],nfl.loc[idx,'season']] > ratings_wins.loc[nfl.loc[idx,'loser'],nfl.loc[idx,'season']]) else 0
    nfl.loc[idx,'massey'] = 1 if (ratings_massey.loc[nfl.loc[idx,'winner'],nfl.loc[idx,'season']] > ratings_massey.loc[nfl.loc[idx,'loser'],nfl.loc[idx,'season']]) else 0
    nfl.loc[idx,'colley'] = 1 if (ratings_colley.loc[nfl.loc[idx,'winner'],nfl.loc[idx,'season']] > ratings_colley.loc[nfl.loc[idx,'loser'],nfl.loc[idx,'season']]) else 0
    nfl.loc[idx,'elo'] = 1 if (ratings_elo.loc[nfl.loc[idx,'winner'],nfl.loc[idx,'season']] > ratings_elo.loc[nfl.loc[idx,'loser'],nfl.loc[idx,'season']]) else 0
    nfl.loc[idx,'zero1'] = 1 if (ratings_zero1.loc[nfl.loc[idx,'winner'],nfl.loc[idx,'season']] > ratings_zero1.loc[nfl.loc[idx,'loser'],nfl.loc[idx,'season']]) else 0
    nfl.loc[idx,'alpha2'] = 1 if (ratings_alpha2.loc[nfl.loc[idx,'winner'],nfl.loc[idx,'season']] > ratings_alpha2.loc[nfl.loc[idx,'loser'],nfl.loc[idx,'season']]) else 0
    nfl.loc[idx,'alpha3'] = 1 if (ratings_alpha3.loc[nfl.loc[idx,'winner'],nfl.loc[idx,'season']] > ratings_alpha3.loc[nfl.loc[idx,'loser'],nfl.loc[idx,'season']]) else 0
    nfl.loc[idx,'alpha4'] = 1 if (ratings_alpha4.loc[nfl.loc[idx,'winner'],nfl.loc[idx,'season']] > ratings_alpha4.loc[nfl.loc[idx,'loser'],nfl.loc[idx,'season']]) else 0
    nfl.loc[idx,'alpha5'] = 1 if (ratings_alpha5.loc[nfl.loc[idx,'winner'],nfl.loc[idx,'season']] > ratings_alpha5.loc[nfl.loc[idx,'loser'],nfl.loc[idx,'season']]) else 0
    nfl.loc[idx,'alpha7'] = 1 if (ratings_alpha7.loc[nfl.loc[idx,'winner'],nfl.loc[idx,'season']] > ratings_alpha7.loc[nfl.loc[idx,'loser'],nfl.loc[idx,'season']]) else 0
    nfl.loc[idx,'alpha10'] = 1 if (ratings_alpha10.loc[nfl.loc[idx,'winner'],nfl.loc[idx,'season']] > ratings_alpha10.loc[nfl.loc[idx,'loser'],nfl.loc[idx,'season']]) else 0
    nfl.loc[idx,'alpha20'] = 1 if (ratings_alpha20.loc[nfl.loc[idx,'winner'],nfl.loc[idx,'season']] > ratings_alpha20.loc[nfl.loc[idx,'loser'],nfl.loc[idx,'season']]) else 0
    nfl.loc[idx,'alpha100'] = 1 if (ratings_alpha100.loc[nfl.loc[idx,'winner'],nfl.loc[idx,'season']] > ratings_alpha100.loc[nfl.loc[idx,'loser'],nfl.loc[idx,'season']]) else 0
    nfl.loc[idx,'alpha1000'] = 1 if (ratings_alpha1000.loc[nfl.loc[idx,'winner'],nfl.loc[idx,'season']] > ratings_alpha1000.loc[nfl.loc[idx,'loser'],nfl.loc[idx,'season']]) else 0

score_nfl = nfl.filter(['season','wins','massey','colley','elo','zero1','alpha2','alpha3','alpha4','alpha5','alpha7','alpha10','alpha20','alpha100','alpha1000'])

pivotscore_nfl = score_nfl.pivot_table(index=['season'], aggfunc=sum)

#
output = pandas.ExcelWriter('results18.xlsx', engine='xlsxwriter')
pivotscore_nhl.to_excel(output, sheet_name='NHL')
pivotscore_nba.to_excel(output, sheet_name='NBA')
pivotscore_nfl.to_excel(output, sheet_name='NFL')
output.save()

output = pandas.ExcelWriter('formcnemar18.xlsx', engine='xlsxwriter')
nhl.to_excel(output, sheet_name='NHL')
nba.to_excel(output, sheet_name='NBA')
nfl.to_excel(output, sheet_name='NFL')
output.save()