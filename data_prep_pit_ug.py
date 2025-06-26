# -*- coding: utf-8 -*-
"""
Created on Sun Jun 22 19:56:45 2025

@author: ssj34
"""
# Sampling

import sys
sys.path.insert(0, 'C:/Users/wb305167/OneDrive - WBG/python_latest/Tax-Revenue-Analysis')
from stata_python import *
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt

pit_df=pd.read_csv('PAYE_2022_23_Annual_Individual_anon.csv', index_col=0)

pit_df['id_n'] = pit_df.index
pit_df['Year'] = 2023
#df_1 = centile_plot(pit_df, "sch1_gross_income", 10)

df_1, bin_list, row_bin = my_qcut(pit_df, "sch1_gross_income", 10)

tabulate(df_1, "mean", "sch1_gross_income", by="centile")

tabulate(df_1, "sum", "sch1_tax_on_total_income", by="centile")

pit_df["sch1_tax_on_total_income"].sum()/1e9

pit_df=pit_df.sort_values(by=['sch1_gross_income'])
pit_df=pit_df.reset_index(drop=True)
# allocate the data into bins
pit_df['bin'] = pd.qcut(pit_df['sch1_gross_income'], 10, labels=False)
pit_df['weight']=1
# bin_ratio is the fraction of the number of records selected in each bin
# 1/10,...1/5, 1/1
#bin_ratio=[20,20,20,20,20,10,10,5,2,1]
bin_ratio=[500,500,500,500,500,100,100,20,10,5]
frames=[]
df={}
for i in range(len(bin_ratio)):
    # find out the size of each bin
    bin_size=len(pit_df[pit_df['bin']==i])//bin_ratio[i]
    # draw a random sample from each bin
    df[i]=pit_df[pit_df['bin']==i].sample(n=bin_size)
    df[i]['weight'] = bin_ratio[i]
    frames=frames+[df[i]]

pit_sample= pd.concat(frames)
#pit_sample.to_csv('taxcalc/pit_sample_ug.csv')
pit_sample.to_csv('taxcalc/pit_sample_small_ug.csv')

varlist = ['sch1_gross_income', 'sch1_allowable_deductions','sch1_total_taxable_income',
           'sch1_tax_on_total_income']
total_weight_sample = pit_sample['weight'].sum()
total_weight_population = pit_df['weight'].sum()
#comparing the statistic of the population and sample
for var in varlist:
    pit_sample['weighted_'+var] = pit_sample[var]*pit_sample['weight']
    sample_sum = pit_sample['weighted_'+var].sum()
    population_sum = pit_df[var].sum()
    print("            Sample Sum for ", var, " = ", sample_sum)
    print("        Population Sum for ", var, " = ", population_sum)
    print(" Sampling Error for Sum(%) ", var, " = ", "{:.2%}".format((population_sum-sample_sum)/population_sum))
    sample_mean = sample_sum/total_weight_sample
    population_mean = population_sum/total_weight_population
    print("           Sample Mean for ", var, " = ", sample_mean)
    print("       Population Mean for ", var, " = ", population_mean)
    print("Sampling Error for Mean(%) ", var, " = ", "{:.2%}".format((population_mean-sample_mean)/population_mean))    

#df = pd.read_csv("taxcalc/pit_sample_ug.csv")
df = pd.read_csv("taxcalc/pit_sample_small_ug.csv")

# to generate the weights file

df_weight = df[['weight']].rename(columns={'weight':'WT2023'})
#df_weight = df[['Taxpayer_ID']].rename(columns={'Taxpayer_ID':'WT2023'})
#df_weight['WT2023'] = 1.0
for year in range(2024,2031):
    df_weight['WT'+str(year)] = df_weight['WT2023']
#df_weight.to_csv("taxcalc/pit_sample_weights_ug.csv", index=False)
df_weight.to_csv("taxcalc/pit_sample_small_weights_ug.csv", index=False)

# Run separately after generating the file to calibrate
#df_weight = pd.read_csv("taxcalc/pit_sample_weights_ug.csv")
df_weight = pd.read_csv("taxcalc/pit_sample_small_weights_ug.csv")

# Calibration
# Collection 2022-23  
coll_cit_2022_23_actual = 4454*1e9
# Collection in the model = 1087 bill
coll_cit_2022_23_model = 4937*1e9
# multiplicative factor
multiplicative_factor = coll_cit_2022_23_actual/coll_cit_2022_23_model
#print(multiplicative_factor)
df_weight['WT2023'] = multiplicative_factor*df_weight['WT2023'] 
for year in range(2024,2031):
    df_weight['WT'+str(year)] = df_weight['WT2023']

#df_weight.to_csv("taxcalc/pit_sample_weights_ug.csv", index=False)
df_weight.to_csv("taxcalc/pit_sample_small_weights_ug.csv", index=False)




