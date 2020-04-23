# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 13:36:56 2020

@author: vdabadgh
"""

import pandas as pd
import researchpy as rp
import numpy as np


"""
McNemar's Test
"""
xls = pd.ExcelFile("formcnemar18.xlsx")
sports = xls.sheet_names
df = {sport: xls.parse(sport) for sport in sports}
rankings = df["NHL"].columns.tolist()[-15:-1]

# mcnemar by league
mcnemar_league = {}
for sport in sports:
    mcnemar_league[sport] = np.ones([len(rankings), len(rankings), 2])
    for i, r1 in enumerate(rankings):
        for j, r2 in enumerate(rankings):
            if r1 != r2:
                _, res = rp.crosstab(df[sport][r1], df[sport][r2], test="mcnemar")
                chisq, p, _ = res.results.values
                mcnemar_league[sport][i, j] = np.array([chisq, p])

# write to excel
writer = pd.ExcelWriter('mcnemar_league.xlsx', engine='xlsxwriter')
# define sheet names
chisq_sheets = [sport + "_chisq" for sport in sports]
p_sheets = [sport + "_p-value" for sport in sports]

# loop for each sport
for sport, chisq_sheet, p_sheet in zip(sports, chisq_sheets, p_sheets):
    # chi_sq
    chisq_df = pd.DataFrame(mcnemar_league[sport][:, :, 0], index=rankings, columns=rankings)
    chisq_df.to_excel(writer, sheet_name=chisq_sheet)
    # p-value
    p_df = pd.DataFrame(mcnemar_league[sport][:, :, 1], index=rankings, columns=rankings)
    p_df.to_excel(writer, sheet_name=p_sheet)
writer.save()

# mcnemar by season
# seasons = {sport: set(df[sport]["season"]) for sport in sports}
# mcnemar_season = {sport: {} for sport in sports}
# for sport in sports:
#     for season in seasons[sport]:
#         df_season = df[sport].loc[df[sport]["season"] == season]
#         mcnemar_season[sport][season] = np.ones([len(rankings), len(rankings), 2])
#         for i, r1 in enumerate(rankings):
#             for j, r2 in enumerate(rankings):
#                 if r1 != r2:
#                     _, res = rp.crosstab(df_season[r1], df[sport][r2], test="mcnemar")
#                     chisq, p, _ = res.results.values
#                     mcnemar_season[sport][season][i, j] = np.array([chisq, p])

# # write to excel
# writer = pd.ExcelWriter('mcnemar_season.xlsx', engine='xlsxwriter')
# # define sheet names
# chisq_sheets = {sport: {season: sport + "_" + str(season) + "_chisq"
#                         for season in seasons[sport]} for sport in sports}
# p_sheets = {sport: {season: sport + "_" + str(season) + "_p-value"
#                     for season in seasons[sport]} for sport in sports}

# for sport in sports:
#     for season in seasons[sport]:
#         # chi_sq
#         chisq_df = pd.DataFrame(mcnemar_season[sport][season][:, :, 0], index=rankings, columns=rankings)
#         chisq_df.to_excel(writer, sheet_name=chisq_sheets[sport][season])
#         # p-value
#         p_df = pd.DataFrame(mcnemar_season[sport][season][:, :, 1], index=rankings, columns=rankings)
#         p_df.to_excel(writer, sheet_name=p_sheets[sport][season])
# writer.save()

# a[:, None] - a[None, :]