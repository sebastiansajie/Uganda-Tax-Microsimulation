# -*- coding: utf-8 -*-
"""
Created on Sat Apr 26 16:39:02 2025

@author: ssj34
"""

import pandas as pd
import numpy as np

df = pd.read_csv("taxcalc/vat_botswana_monthly.csv")

fields = ["CONS_Alcohol_Tobacco", "CONS_Clothing_Footwear", "CONS_Housing",	
          "CONS_Hhold_Goods_Services", "CONS_Health", "CONS_Transport", "CONS_Communication", 
          "CONS_Recreation_Culture",	"CONS_Education", "CONS_Restaurants_Hotels", 
          "CONS_Miscellaneous"]

df["'CONS_Food"] = df["CONS_Food"]*12

df['GROSS_INCOME'] = df['GROSS_INCOME']*12

cons = 0
for field in fields:
    df[field] = df[field]*12
    cons = cons+df[field]

df['CONS_Other'] = cons
df['CONS_Total'] = cons + df["CONS_Food"]
df.to_csv("taxcalc/vat_botswana.csv", index=False)

df_weight = df[['weight']].rename(columns={'weight':'WT2015'})
for year in range(2016,2031):
    df_weight['WT'+str(year)] = df_weight['WT2015']
df_weight.to_csv("taxcalc/vat_weights_botswana.csv", index=False)

df_weight = pd.read_csv("taxcalc/vat_weights_botswana.csv")
# Calibration
vat_collection_2015 = 5.5476
vat_model_base = 3.43
multiplicative_factor = vat_collection_2015/vat_model_base
print(multiplicative_factor)
df_weight['WT2015'] = multiplicative_factor*df_weight['WT2015'] 
for year in range(2016,2031):
    df_weight['WT'+str(year)] = df_weight['WT2015']
df_weight.to_csv("taxcalc/vat_weights_botswana.csv", index=False)