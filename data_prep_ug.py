# -*- coding: utf-8 -*-
"""
Created on Sun Jun 22 19:56:45 2025

@author: ssj34
"""

import pandas as pd
import numpy as np

"""
df = pd.read_csv("taxcalc/cit_data_training_ug.csv")

# to generate the weights file

df_weight = df[['weight']].rename(columns={'weight':'WT2023'})
for year in range(2024,2031):
    df_weight['WT'+str(year)] = df_weight['WT2023']
df_weight.to_csv("taxcalc/cit_weights_training_ug.csv", index=False)
"""

# Run separately after generating the file
df_weight = pd.read_csv("taxcalc/cit_weights_training_ug.csv")
# Calibration
# Collection 2022-23  
coll_cit_2022_23_actual = 2077*1e9
# Collection in the model = 1087 bill
coll_cit_2022_23_model = 2392*1e9
# multiplicative factor
multiplicative_factor = coll_cit_2022_23_actual/coll_cit_2022_23_model
#print(multiplicative_factor)
df_weight['WT2023'] = multiplicative_factor*df_weight['WT2023'] 
for year in range(2024,2031):
    df_weight['WT'+str(year)] = df_weight['WT2023']
df_weight.to_csv("taxcalc/cit_weights_training_ug.csv", index=False)




