
"""
pitaxcalc-demo functions that calculate personal income tax liability.
"""
# CODING-STYLE CHECKS:
# pycodestyle functions.py
# pylint --disable=locally-disabled functions.py

import math
import copy
import numpy as np
from taxcalc.decorators import iterate_jit



@iterate_jit(nopython=True)
def cal_CONS_Food(CONS_Food_Crops_behavior, rate_Food_Crops, vat_Food_Crops):
    """
    """
    vat_Food_Crops = rate_Food_Crops * CONS_Food_Crops_behavior
    return vat_Food_Crops



@iterate_jit(nopython=True)
def cal_CONS_behavior(CONS_Food_Crops, rate_Food_Crops, rate_Food_Crops_curr_law, elasticity_consumption_threshold, elasticity_consumption_value, CONS_Food_Crops_behavior):
    """
    Compute consumption after adjusting for behavior.
    """
    elasticity_consumption_threshold0 = elasticity_consumption_threshold[0]
    elasticity_consumption_threshold1 = elasticity_consumption_threshold[1]
    #elasticity_consumption_threshold2 = elasticity_pit_taxable_income_threshold[2]
    elasticity_consumption_value0=elasticity_consumption_value[0]
    elasticity_consumption_value1=elasticity_consumption_value[1]
    elasticity_consumption_value2=elasticity_consumption_value[2]
    if CONS_Food_Crops<=0:
        elasticity=0
    elif CONS_Food_Crops<elasticity_consumption_threshold0:
        elasticity=elasticity_consumption_value0
    elif CONS_Food_Crops<elasticity_consumption_threshold1:
        elasticity=elasticity_consumption_value1
    else:
        elasticity=elasticity_consumption_value2
    rate=rate_Food_Crops
    rate_curr_law=rate_Food_Crops_curr_law
    # Compute the fractional change of VAT rate safely
    if rate == 0 and rate_curr_law == 0:
        # If both rates are zero, we set the fraction change to 0 (i.e. no effect)
        frac_change_of_vat_rate = 0
    elif rate_curr_law == 0:
        # If current law rate is zero but the other rate is non-zero, this would normally cause a division by zero.
        # You can choose to raise an error, or decide on another fallback behavior.
        frac_change_of_vat_rate = 0
    else:
        frac_change_of_vat_rate = (rate - rate_curr_law) / rate_curr_law
    frac_change_of_consumption = elasticity*frac_change_of_vat_rate
    CONS_Food_Crops_behavior = CONS_Food_Crops*(1+frac_change_of_consumption)
    return CONS_Food_Crops_behavior

@iterate_jit(nopython=True)
def cal_vat(vat_Food_Crops, vat_Processed_Food, vat_Fruits_Vegetables_Spices, vat_Fish, vat_Meat, vat_Poultry, vat_Dairy, vat_Beverages,
            vat_Alcohol, vat_Tobacco, vat_Other_Non_Consumption, vat_Rent, vat_Electricity, vat_Water, vat_Fuel, vat_Other_Services,
            vat_Telecom, vat_Soap_Cosmetics, vat_Other_Home_Consumption, vat_Books_Newspapers, vat_Health, vat_Education, 
            vat_Transport_Services, vat_Air_Transport, vat_Accomodation, vat_Entertainment, vat_Finance, vat_Clothing_Footwear, 
            vat_Furniture, vat_Durable_Goods, vat_Vehicle_Purchase, vat_Other_Consumption, vat_Religious_Consumption, vatax):
    # Sum the VAT values for the following products:
    # vat_Food_Crops, vat_Processed_Food, vat_Fruits_Vegetables_Spices, vat_Fish, vat_Meat, vat_Poultry, vat_Dairy, vat_Beverages, vat_Alcohol, vat_Tobacco, vat_Other_Non_Consumption, vat_Rent, vat_Electricity, vat_Water, vat_Fuel, vat_Other_Services, vat_Telecom, vat_Soap_Cosmetics, vat_Other_Home_Consumption, vat_Books_Newspapers, vat_Health, vat_Education, vat_Transport_Services, vat_Air_Transport, vat_Accomodation, vat_Entertainment, vat_Finance, vat_Clothing_Footwear, vat_Furniture, vat_Durable_Goods, vat_Vehicle_Purchase, vat_Other_Consumption, vat_Religious_Consumption
    vat = vat_Food_Crops + vat_Processed_Food + vat_Fruits_Vegetables_Spices + vat_Fish + vat_Meat + vat_Poultry + vat_Dairy + vat_Beverages + vat_Alcohol + vat_Tobacco + vat_Other_Non_Consumption + vat_Rent + vat_Electricity + vat_Water + vat_Fuel + vat_Other_Services + vat_Telecom + vat_Soap_Cosmetics + vat_Other_Home_Consumption + vat_Books_Newspapers + vat_Health + vat_Education + vat_Transport_Services + vat_Air_Transport + vat_Accomodation + vat_Entertainment + vat_Finance + vat_Clothing_Footwear + vat_Furniture + vat_Durable_Goods + vat_Vehicle_Purchase + vat_Other_Consumption + vat_Religious_Consumption
    vatax = vat
    return vatax
