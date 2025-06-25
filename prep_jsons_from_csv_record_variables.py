# -*- coding: utf-8 -*-
"""
Created on Sat Apr 26 16:39:02 2025

@author: ssj34
"""

import pandas as pd
import json


df = pd.read_csv("record_variables_input_pit.csv")

final_json = {'read': {}, 'calc': {}}
cols = list(df.columns)
for idx, row in df.iterrows():
    print(row['field_category'])
    field_name=row['field_name']
    print(field_name)
    item = {}
    if (row['field_category']=='read'):
        item['required'] = row['required']
    item['type'] = str(row['type'])
    item['desc'] = str(row['desc'])
    #print(row['2015'])
    form = {}
    form[str('2023')] = str(row['2023'])
    item['form']=form
    if (row['field_category']=='read'):
        item['cross_year'] = str(row['cross_year'])
        item['attribute'] = str(row['attribute'])
    final_json[row['field_category']][field_name]=item

with open('taxcalc/records_variables_pit_ug.json', 'w') as f:
    json.dump(final_json, f, indent=4)

print(f"JSON successfully written {final_json}")

