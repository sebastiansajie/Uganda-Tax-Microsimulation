
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
def cal_CONS_Food(CONS_Food):
    """
    """  
    return CONS_Food

@iterate_jit(nopython=True)
def cal_CONS_Non_Food(CONS_Alcohol_Tobacco, CONS_Clothing_Footwear,	CONS_Housing, 
                      CONS_Hhold_Goods_Services, CONS_Health, CONS_Transport, 
                      CONS_Communication, CONS_Recreation_Culture, CONS_Education, 
                      CONS_Restaurants_Hotels,	 CONS_Miscellaneous, CONS_Non_Food):
    """
    """
    CONS_Non_Food = (CONS_Alcohol_Tobacco + CONS_Clothing_Footwear + CONS_Housing + 
                  CONS_Hhold_Goods_Services + CONS_Health + CONS_Transport + 
                  CONS_Communication + CONS_Recreation_Culture + CONS_Education + 
                  CONS_Restaurants_Hotels + CONS_Miscellaneous)    
    return CONS_Non_Food

@iterate_jit(nopython=True)
def cal_CONS_Other(CONS_Alcohol_Tobacco, CONS_Clothing_Footwear,	CONS_Housing, 
                   CONS_Hhold_Goods_Services, CONS_Health, CONS_Transport, 
                   CONS_Communication, CONS_Recreation_Culture, CONS_Education, 
                   CONS_Restaurants_Hotels,	 CONS_Miscellaneous, CONS_Other):
    """
    """
    CONS_Other = (CONS_Alcohol_Tobacco + CONS_Clothing_Footwear + CONS_Housing + 
                  CONS_Hhold_Goods_Services + CONS_Health + CONS_Transport + 
                  CONS_Communication + CONS_Recreation_Culture + CONS_Education + 
                  CONS_Restaurants_Hotels + CONS_Miscellaneous)
    return CONS_Other

@iterate_jit(nopython=True)
def cal_CONS_Total(CONS_Food, CONS_Other, CONS_Total):
    """
    """
    CONS_Total = CONS_Food + CONS_Other
    return CONS_Total

@iterate_jit(nopython=True)
def cal_etr_Food(rate_Food, etr_Food):
    """
    """
    etr_Food = rate_Food

    return etr_Food

@iterate_jit(nopython=True)
def cal_etr_Food_curr_law(rate_Food_curr_law, CONS_Food, etr_Food_curr_law):
    """
    """

    etr_Food_curr_law = rate_Food_curr_law
    
    return etr_Food_curr_law

@iterate_jit(nopython=True)
def cal_etr_Non_Food(rate_Alcohol_Tobacco, rate_Clothing_Footwear, 
                     rate_Housing, rate_Household_Goods_Services, 
                     rate_Health, rate_Transport, rate_Communication, 
                     rate_Recreation_Culture, rate_Education, 
                     rate_Restaurants_Hotels, rate_Miscellaneous,
                     CONS_Alcohol_Tobacco, CONS_Clothing_Footwear, CONS_Housing, 
                     CONS_Hhold_Goods_Services, CONS_Health, CONS_Transport, CONS_Communication,
                     CONS_Recreation_Culture, CONS_Education, CONS_Restaurants_Hotels,
                     CONS_Miscellaneous, CONS_Non_Food, etr_Non_Food):
    """
    """
    if CONS_Non_Food != 0:
        etr_Non_Food = (rate_Alcohol_Tobacco*CONS_Alcohol_Tobacco+rate_Clothing_Footwear*CONS_Clothing_Footwear+
                    rate_Housing*CONS_Housing+rate_Household_Goods_Services*CONS_Hhold_Goods_Services+
                    rate_Health*CONS_Health+rate_Transport*CONS_Transport+
                    rate_Communication*CONS_Communication+rate_Recreation_Culture*CONS_Recreation_Culture+
                    rate_Education*CONS_Education+rate_Restaurants_Hotels*CONS_Restaurants_Hotels+
                    rate_Miscellaneous*CONS_Miscellaneous)/CONS_Non_Food
    else:
        etr_Non_Food = 0.0       
    return etr_Non_Food

@iterate_jit(nopython=True)
def cal_etr_Non_Food_curr_law(rate_Alcohol_Tobacco_curr_law, rate_Clothing_Footwear_curr_law, rate_Housing_curr_law, rate_Household_Goods_Services_curr_law, rate_Health_curr_law, rate_Transport_curr_law, rate_Communication_curr_law, rate_Recreation_Culture_curr_law, rate_Education_curr_law, rate_Restaurants_Hotels_curr_law, rate_Miscellaneous_curr_law,
                     CONS_Alcohol_Tobacco, CONS_Clothing_Footwear, CONS_Housing, 
                     CONS_Hhold_Goods_Services, CONS_Health, CONS_Transport, CONS_Communication,
                     CONS_Recreation_Culture, CONS_Education, CONS_Restaurants_Hotels,
                     CONS_Miscellaneous, CONS_Non_Food, etr_Non_Food_curr_law):
    """
    """
    if CONS_Non_Food != 0:
        etr_Non_Food_curr_law = (rate_Alcohol_Tobacco_curr_law*CONS_Alcohol_Tobacco+rate_Clothing_Footwear_curr_law*CONS_Clothing_Footwear+
                    rate_Housing_curr_law*CONS_Housing+rate_Household_Goods_Services_curr_law*CONS_Hhold_Goods_Services+
                    rate_Health_curr_law*CONS_Health+rate_Transport_curr_law*CONS_Transport+
                    rate_Communication_curr_law*CONS_Communication+rate_Recreation_Culture_curr_law*CONS_Recreation_Culture+
                    rate_Education_curr_law*CONS_Education+rate_Restaurants_Hotels_curr_law*CONS_Restaurants_Hotels+
                    rate_Miscellaneous_curr_law*CONS_Miscellaneous)/CONS_Non_Food
    else:
        etr_Non_Food_curr_law = 0.0      
    return etr_Non_Food_curr_law

@iterate_jit(nopython=True)
def cal_CONS_Food_behavior(elasticity_consumption_food_threshold, elasticity_consumption_food_value, etr_Food, etr_Food_curr_law, CONS_Total, CONS_Food, CONS_Food_behavior):
    """
    Compute consumption after adjusting for behavior.
    """
    elasticity_consumption_threshold0 = elasticity_consumption_food_threshold[0]
    elasticity_consumption_threshold1 = elasticity_consumption_food_threshold[1]
    #elasticity_consumption_threshold2 = elasticity_pit_taxable_income_threshold[2]
    elasticity_consumption_value0=elasticity_consumption_food_value[0]
    elasticity_consumption_value1=elasticity_consumption_food_value[1]
    elasticity_consumption_value2=elasticity_consumption_food_value[2]
    if CONS_Total<=0:
        elasticity=0
    elif CONS_Total<elasticity_consumption_threshold0:
        elasticity=elasticity_consumption_value0
    elif CONS_Total<elasticity_consumption_threshold1:
        elasticity=elasticity_consumption_value1
    else:
        elasticity=elasticity_consumption_value2
    rate=etr_Food
    rate_curr_law=etr_Food_curr_law
    # Compute the fractional change of VAT rate safely
    frac_change_of_vat_rate = (rate - rate_curr_law) / (1+rate_curr_law)
    frac_change_of_consumption = elasticity*frac_change_of_vat_rate
    CONS_Food_behavior = CONS_Food*(1+frac_change_of_consumption)
    return CONS_Food_behavior

@iterate_jit(nopython=True)
def cal_CONS_Non_Food_behavior(elasticity_consumption_non_food_threshold, elasticity_consumption_non_food_value, etr_Non_Food, etr_Non_Food_curr_law, CONS_Total, CONS_Non_Food, CONS_Non_Food_behavior):
    """
    Compute consumption after adjusting for behavior.
    """
    elasticity_consumption_threshold0 = elasticity_consumption_non_food_threshold[0]
    elasticity_consumption_threshold1 = elasticity_consumption_non_food_threshold[1]
    #elasticity_consumption_threshold2 = elasticity_pit_taxable_income_threshold[2]
    elasticity_consumption_value0=elasticity_consumption_non_food_value[0]
    elasticity_consumption_value1=elasticity_consumption_non_food_value[1]
    elasticity_consumption_value2=elasticity_consumption_non_food_value[2]
    if CONS_Total<=0:
        elasticity=0
    elif CONS_Total<elasticity_consumption_threshold0:
        elasticity=elasticity_consumption_value0
    elif CONS_Total<elasticity_consumption_threshold1:
        elasticity=elasticity_consumption_value1
    else:
        elasticity=elasticity_consumption_value2
    rate=etr_Non_Food
    rate_curr_law=etr_Non_Food_curr_law
    # Compute the fractional change of VAT rate safely
    frac_change_of_vat_rate = (rate - rate_curr_law) / (1+rate_curr_law)
    frac_change_of_consumption = elasticity*frac_change_of_vat_rate
    CONS_Non_Food_behavior = CONS_Non_Food*(1+frac_change_of_consumption)
    return CONS_Non_Food_behavior

@iterate_jit(nopython=True)
def cal_vat_food(etr_Food, CONS_Food_behavior, vat_Food):
    # Sum the VAT values for the following products:
    # vat_Food_Crops, vat_Processed_Food, vat_Fruits_Vegetables_Spices, vat_Fish, vat_Meat, vat_Poultry, vat_Dairy, vat_Beverages, vat_Alcohol, vat_Tobacco, vat_Other_Non_Consumption, vat_Rent, vat_Electricity, vat_Water, vat_Fuel, vat_Other_Services, vat_Telecom, vat_Soap_Cosmetics, vat_Other_Home_Consumption, vat_Books_Newspapers, vat_Health, vat_Education, vat_Transport_Services, vat_Air_Transport, vat_Accomodation, vat_Entertainment, vat_Finance, vat_Clothing_Footwear, vat_Furniture, vat_Durable_Goods, vat_Vehicle_Purchase, vat_Other_Consumption, vat_Religious_Consumption
    vat_Food = etr_Food*CONS_Food_behavior 
    return vat_Food

@iterate_jit(nopython=True)
def cal_vat_non_food(etr_Non_Food, CONS_Non_Food_behavior, vat_Non_Food):
    # Sum the VAT values for the following products:
    # vat_Food_Crops, vat_Processed_Food, vat_Fruits_Vegetables_Spices, vat_Fish, vat_Meat, vat_Poultry, vat_Dairy, vat_Beverages, vat_Alcohol, vat_Tobacco, vat_Other_Non_Consumption, vat_Rent, vat_Electricity, vat_Water, vat_Fuel, vat_Other_Services, vat_Telecom, vat_Soap_Cosmetics, vat_Other_Home_Consumption, vat_Books_Newspapers, vat_Health, vat_Education, vat_Transport_Services, vat_Air_Transport, vat_Accomodation, vat_Entertainment, vat_Finance, vat_Clothing_Footwear, vat_Furniture, vat_Durable_Goods, vat_Vehicle_Purchase, vat_Other_Consumption, vat_Religious_Consumption
    vat_Non_Food = etr_Non_Food*CONS_Non_Food_behavior 
    return vat_Non_Food

@iterate_jit(nopython=True)
def cal_vat(vat_Food, vat_Non_Food, vatax):
    # Sum the VAT values for the following products:
    # vat_Food_Crops, vat_Processed_Food, vat_Fruits_Vegetables_Spices, vat_Fish, vat_Meat, vat_Poultry, vat_Dairy, vat_Beverages, vat_Alcohol, vat_Tobacco, vat_Other_Non_Consumption, vat_Rent, vat_Electricity, vat_Water, vat_Fuel, vat_Other_Services, vat_Telecom, vat_Soap_Cosmetics, vat_Other_Home_Consumption, vat_Books_Newspapers, vat_Health, vat_Education, vat_Transport_Services, vat_Air_Transport, vat_Accomodation, vat_Entertainment, vat_Finance, vat_Clothing_Footwear, vat_Furniture, vat_Durable_Goods, vat_Vehicle_Purchase, vat_Other_Consumption, vat_Religious_Consumption
    vatax = vat_Food + vat_Non_Food 
    #vatax=1.0
    return vatax
