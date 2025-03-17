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
def cal_gross_wages (wage_w_pr, piecework_w_pr, comissions_w_pr, overtime_w_pr, 
                    rewards_w_pr, bonus_w_pr, vacation_w_pr, profit_sharing_w_pr, 
                    profit_sharing_w_se, christmas_bonus_w_pr, christmas_bonus_w_se, 
                    wage_bu_t, layoff_w_t, transfers_pension_mx, transfers_pension_abroad, 
                    other_other_lastmonth, other_other_prev5, wage_w_se, gross_wage):
    """
    Compute total gross income from wages.
    """
    gross_wage = (wage_w_pr + piecework_w_pr + comissions_w_pr + overtime_w_pr + 
                  rewards_w_pr + bonus_w_pr + vacation_w_pr + profit_sharing_w_pr + 
                  profit_sharing_w_se + christmas_bonus_w_pr + christmas_bonus_w_se + 
                  wage_bu_t + layoff_w_t + transfers_pension_mx + transfers_pension_abroad + 
                  other_other_lastmonth + other_other_prev5 + wage_w_se)
    return gross_wage

@iterate_jit(nopython=True)
def cal_gross_income_ex (gross_wage, income_dividends_t, income_interest_t, income_capital_t, 
                                     income_other_t, rent_building_abroad, rent_building_mx, 
                                     transfers_donation_nongov, transfers_donation_otherHH, 
                                     gross_income_ex):
    """
    Compute total gross income (including exempt incomes).
    """
    gross_income_ex = (gross_wage + income_dividends_t + income_interest_t + income_capital_t + income_other_t +     
                              rent_building_abroad + rent_building_mx +
                              transfers_donation_nongov + transfers_donation_otherHH)
    return gross_income_ex

@iterate_jit(nopython=True)
def cal_t_cum_income(income_wages_t, income_dividends_t, income_interest_t, income_capital_t, income_other_t,     
                          income_rent_t, income_donations_t, tax_c_income):
    """
    Compute total cumulable incomes.
    """
    tax_c_income = (income_wages_t + income_dividends_t + income_interest_t + income_capital_t + income_other_t +     
                              income_rent_t + income_donations_t)
    return tax_c_income

@iterate_jit(nopython=True)
def cal_deductions(deductible_expenses, gross_total_income, authorized_deduction):
    """
    Compute authorized deductions.
    """
    authorized_deduction = min(deductible_expenses, gross_total_income * 0.15)  
    return authorized_deduction

@iterate_jit(nopython=True)
def cal_taxable_income(authorized_deduction, tax_c_income):
    """
    Compute taxable income.
    """    
    taxable_income = tax_c_income - authorized_deduction
    return taxable_income

@iterate_jit(nopython=True)
def cal_pit_w(rate1, rate2, rate3, rate4, rate5, rate6, rate7, rate8, rate9, rate10, rate11, 
              tbrk1, tbrk2, tbrk3, tbrk4, tbrk5, tbrk6, tbrk7, tbrk8, tbrk9, tbrk10, 
              taxable_income, pitax_w, highest_rate_applied):
    """
    Compute PIT and store the highest marginal rate applied.
    """
    inc = taxable_income

    if inc <= tbrk1:
        pitax_w = (inc - 0.01) * rate1
        highest_rate_applied = rate1
    elif inc > tbrk1 and inc <= tbrk2:
        pitax_w = (inc - tbrk1) * rate2 + (tbrk1 - 0.01) * rate1
        highest_rate_applied = rate2
    elif inc > tbrk2 and inc <= tbrk3:
        pitax_w = (inc - tbrk2) * rate3 + (tbrk2 - tbrk1) * rate2 + (tbrk1 - 0.01) * rate1
        highest_rate_applied = rate3
    elif inc > tbrk3 and inc <= tbrk4:
        pitax_w = (inc - tbrk3) * rate4 + (tbrk3 - tbrk2) * rate3 + (tbrk2 - tbrk1) * rate2 + (tbrk1 - 0.01) * rate1
        highest_rate_applied = rate4
    elif inc > tbrk4 and inc <= tbrk5:
        pitax_w = (inc - tbrk4) * rate5 + (tbrk4 - tbrk3) * rate4 + (tbrk3 - tbrk2) * rate3 + (tbrk2 - tbrk1) * rate2 + (tbrk1 - 0.01) * rate1
        highest_rate_applied = rate5
    elif inc > tbrk5 and inc <= tbrk6:
        pitax_w = (inc - tbrk5) * rate6 + (tbrk5 - tbrk4) * rate5 + (tbrk4 - tbrk3) * rate4 + (tbrk3 - tbrk2) * rate3 + (tbrk2 - tbrk1) * rate2 + (tbrk1 - 0.01) * rate1
        highest_rate_applied = rate6
    elif inc > tbrk6 and inc <= tbrk7:
        pitax_w = (inc - tbrk6) * rate7 + (tbrk6 - tbrk5) * rate6 + (tbrk5 - tbrk4) * rate5 + (tbrk4 - tbrk3) * rate4 + (tbrk3 - tbrk2) * rate3 + (tbrk2 - tbrk1) * rate2 + (tbrk1 - 0.01) * rate1
        highest_rate_applied = rate7
    elif inc > tbrk7 and inc <= tbrk8:
        pitax_w = (inc - tbrk7) * rate8 + (tbrk7 - tbrk6) * rate7 + (tbrk6 - tbrk5) * rate6 + (tbrk5 - tbrk4) * rate5 + (tbrk4 - tbrk3) * rate4 + (tbrk3 - tbrk2) * rate3 + (tbrk2 - tbrk1) * rate2 + (tbrk1 - 0.01) * rate1
        highest_rate_applied = rate8
    elif inc > tbrk8 and inc <= tbrk9:
        pitax_w = (inc - tbrk8) * rate9 + (tbrk8 - tbrk7) * rate8 + (tbrk7 - tbrk6) * rate7 + (tbrk6 - tbrk5) * rate6 + (tbrk5 - tbrk4) * rate5 + (tbrk4 - tbrk3) * rate4 + (tbrk3 - tbrk2) * rate3 + (tbrk2 - tbrk1) * rate2 + (tbrk1 - 0.01) * rate1
        highest_rate_applied = rate9
    elif inc > tbrk9 and inc <= tbrk10:
        pitax_w = (inc - tbrk9) * rate10 + (tbrk9 - tbrk8) * rate9 + (tbrk8 - tbrk7) * rate8 + (tbrk7 - tbrk6) * rate7 + (tbrk6 - tbrk5) * rate6 + (tbrk5 - tbrk4) * rate5 + (tbrk4 - tbrk3) * rate4 + (tbrk3 - tbrk2) * rate3 + (tbrk2 - tbrk1) * rate2 + (tbrk1 - 0.01) * rate1
        highest_rate_applied = rate10
    elif inc > tbrk10:
        pitax_w = (inc - tbrk10) * rate11 + (tbrk10 - tbrk9) * rate10 + (tbrk9 - tbrk8) * rate9 + (tbrk8 - tbrk7) * rate8 + (tbrk7 - tbrk6) * rate7 + (tbrk6 - tbrk5) * rate6 + (tbrk5 - tbrk4) * rate5 + (tbrk4 - tbrk3) * rate4 + (tbrk3 - tbrk2) * rate3 + (tbrk2 - tbrk1) * rate2 + (tbrk1 - 0.01) * rate1
        highest_rate_applied = rate11

    return pitax_w, highest_rate_applied

@iterate_jit(nopython=True)
def cal_additional_dividends(additional_dividend_rate, income_dividends_t, additional_dividend_pitax):
    
    """
    Compute additional 10% tax for dividends.
    """
   
    additional_dividend_pitax =(income_dividends_t*additional_dividend_rate)
    return additional_dividend_pitax

@iterate_jit(nopython=True)
def cal_gambling_pitax(gambling_rate, lotteries, gambling_pitax):
    """
    Compute tax for gambling incomes.
    """
    gambling_pitax =  (lotteries*gambling_rate)
    return gambling_pitax

@iterate_jit(nopython=True)
def cal_capitalgains_pitax(capitalgains_bonds_rate, capital_bonds, capitalgains_pitax):
    
    """
    Compute additional 10% tax for dividends.
    """
   
    capitalgains_pitax =(capital_bonds*capitalgains_bonds_rate)
    return capitalgains_pitax

@iterate_jit(nopython=True)
def cal_pendingconcepts_pitax(highest_rate_applied, layoff_pending, pending_capital_pitax, pending_pitax):
    
    """
    Compute tax for pending amounts (layoff compensations, sales of land and housing, etc.).
    """
    pending_pitax = (layoff_pending + pending_capital_pitax)*highest_rate_applied,
    return pending_pitax

@iterate_jit(nopython=True)
def cal_total_pit(pitax_w, additional_dividend_pitax, gambling_pitax, capitalgains_pitax, pending_pitax, pitax):
    """
    Compute Total PIT.
    """
    pitax = pitax_w + additional_dividend_pitax + gambling_pitax + capitalgains_pitax + pending_pitax
    return pitax