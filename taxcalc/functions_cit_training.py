"""
Functions that calculate personal income tax liability.
"""
# CODING-STYLE CHECKS:
# pycodestyle functions.py
# pylint --disable=locally-disabled functions.py

import math
import copy
import numpy as np
from taxcalc.decorators import iterate_jit


@iterate_jit(nopython=True)
def Net_accounting_profit(Revenues, Other_revenues, Expenses, Net_accounting_profit):
    """
    Compute accounting profit from business
    """
    Net_accounting_profit = Revenues + Other_revenues - Expenses
    return Net_accounting_profit

@iterate_jit(nopython=True)
def Total_additions_to_profits(Total_additions_to_GP):
    """
    Compute accounting profit from business
    """
    Total_additions_to_GP = Total_additions_to_GP
    return Total_additions_to_GP

@iterate_jit(nopython=True)
def Total_taxable_profit(Net_accounting_profit, Total_additions_to_GP, 
                         Gross_taxable_profit):
    """
    Compute total taxable profits afer adding back non-allowable deductions.
    """
    Gross_taxable_profit = Net_accounting_profit + Total_additions_to_GP
    return Gross_taxable_profit

@iterate_jit(nopython=True)
def Op_WDV_depr(Op_WDV_Bld, Op_WDV_Mach, Op_WDV_Others, sch2_wdv_beg_year_class_20, Op_WDV_Comp):
    """
    Return the opening WDV of each asset class.
    """
    Op_WDV_Bld, Op_WDV_Mach, Op_WDV_Others, Op_WDV_Comp = (Op_WDV_Bld, 
    Op_WDV_Mach, Op_WDV_Others, Op_WDV_Comp)
    return (Op_WDV_Bld, Op_WDV_Mach, Op_WDV_Others, Op_WDV_Comp)

@iterate_jit(nopython=True)
def Tax_depr_Bld(rate_init_allow_bld, Op_WDV_Bld, Add_Bld, rate_depr_bld, Tax_depr_Bld):
    """
    Compute tax depreciation of building asset class.
    """
    initial_allowance = rate_init_allow_bld*Add_Bld
    Add_Bld_remaining = max(Add_Bld - initial_allowance,0)
    Tax_depr_Bld = initial_allowance + max(rate_depr_bld*(Op_WDV_Bld-Add_Bld_remaining),0)
    return Tax_depr_Bld

@iterate_jit(nopython=True)
def Tax_depr_class_20(rate_depr_20, sch2_wdv_beg_year_class_20, sch2_addinit_during_yr_class_20, 
                      sch2_dispsl_class_20, Tax_depr_class_20):
    """
    Compute tax depreciation of intangibles asset class
    """
    Tax_depr_class_20 = max(rate_depr_20*(sch2_wdv_beg_year_class_20 + sch2_addinit_during_yr_class_20 - sch2_dispsl_class_20),0)
    return Tax_depr_class_20

@iterate_jit(nopython=True)
def Tax_depr_Mach(rate_init_allow_mach, Op_WDV_Mach, Add_Mach, Excl_Mach, rate_depr_mach, Tax_depr_Mach):
    """
    Compute tax depreciation of Machinary asset class
    """
    initial_allowance = rate_init_allow_mach*Add_Mach
    Add_Mach_remaining = max(Add_Mach - initial_allowance,0)
    Tax_depr_Mach = initial_allowance + max(rate_depr_mach*(Op_WDV_Mach + Add_Mach_remaining - Excl_Mach),0)
    return Tax_depr_Mach

@iterate_jit(nopython=True)
def Tax_depr_Others(Op_WDV_Others, Add_Others, Excl_Others, rate_depr_others, Tax_depr_Others):
    """
    Compute tax depreciation of Other asset class
    """
    Tax_depr_Others = max(rate_depr_others*(Op_WDV_Others + Add_Others - Excl_Others),0)
    return Tax_depr_Others

@iterate_jit(nopython=True)
def Tax_depr_Comp(Op_WDV_Comp, Add_Comp, Excl_Comp, rate_depr_comp, Tax_depr_Comp):
    """
    Compute tax depreciation of Computer asset class
    """
    Tax_depr_Comp = max(rate_depr_comp*(Op_WDV_Comp + Add_Comp - Excl_Comp),0)
    return Tax_depr_Comp

@iterate_jit(nopython=True)
def Tax_depreciation(Tax_depr_Bld, Tax_depr_class_20, Tax_depr_Mach, Tax_depr_Others, Tax_depr_Comp, Tax_depr):
    """
    Compute total depreciation of all asset classes.
    """
    Tax_depr = Tax_depr_Bld + Tax_depr_class_20 + Tax_depr_Mach + Tax_depr_Others + Tax_depr_Comp
    return Tax_depr

@iterate_jit(nopython=True)
def Cl_WDV_depr(Op_WDV_Bld, Add_Bld, Excl_Bld, Tax_depr_Bld, 
                sch2_wdv_beg_year_class_20, sch2_addinit_during_yr_class_20, 
                sch2_dispsl_class_20, Tax_depr_class_20,
                Op_WDV_Mach, Add_Mach, Excl_Mach, Tax_depr_Mach,
                Op_WDV_Others, Add_Others, Excl_Others, Tax_depr_Others,
                Op_WDV_Comp, Add_Comp, Excl_Comp, Tax_depr_Comp,
                Cl_WDV_Bld, Cl_WDV_Intang, Cl_WDV_Mach, Cl_WDV_Others, Cl_WDV_Comp):
    """
    Compute Closing WDV of each block of asset.
    """
    Cl_WDV_Bld = max((Op_WDV_Bld + Add_Bld - Excl_Bld),0) - Tax_depr_Bld
    Cl_WDV_Class_20 = max((sch2_wdv_beg_year_class_20 + sch2_addinit_during_yr_class_20 - sch2_dispsl_class_20),0) - Tax_depr_class_20
    Cl_WDV_Mach = max((Op_WDV_Mach + Add_Mach - Excl_Mach),0) - Tax_depr_Mach
    Cl_WDV_Others = max((Op_WDV_Others + Add_Others - Excl_Others),0) - Tax_depr_Others
    Cl_WDV_Comp= max((Op_WDV_Comp + Add_Comp - Excl_Comp),0) - Tax_depr_Comp
    return (Cl_WDV_Bld, Cl_WDV_Class_20, Cl_WDV_Mach, Cl_WDV_Others, Cl_WDV_Comp)

@iterate_jit(nopython=True)
def Total_deductions(tax_holiday_agriculture, tax_holiday_manufacturing, 
                     tax_holiday_education, tax_holiday_ICT, tax_holiday_Bujagali,
                     tax_holiday_SACO_FIN, tax_holiday_other, 
                     Tax_Holiday_Agriculture, Tax_Holiday_Manufacturing, 
                     Tax_Holiday_Education, Tax_Holiday_ICT, Tax_Holiday_Bujagali,
                     Tax_Holiday_SACO_FIN, Tax_Holiday_Other,
                     Tax_depr, Other_deductions, Investment_incentive, Total_deductions):
    """
    Compute net taxable profits afer allowing deductions.
    """

    if (tax_holiday_agriculture==1)&(Tax_Holiday_Agriculture==1):
        incentive_claimed = max(Investment_incentive,0)
    if (tax_holiday_manufacturing==1)&(Tax_Holiday_Manufacturing==1):
        incentive_claimed = max(Investment_incentive,0)
    if (tax_holiday_education==1)&(Tax_Holiday_Education==1):
        incentive_claimed = max(Investment_incentive,0)
    if (tax_holiday_ICT==1)&(Tax_Holiday_ICT==1):
        incentive_claimed = max(Investment_incentive,0)
    if (tax_holiday_Bujagali==1)&(Tax_Holiday_Bujagali==1):
        incentive_claimed = max(Investment_incentive,0)
    if (tax_holiday_SACO_FIN==1)&(Tax_Holiday_SACO_FIN==1):
        incentive_claimed = max(Investment_incentive,0)
    if (tax_holiday_other==1)&(Tax_Holiday_Other==1):
        incentive_claimed = max(Investment_incentive,0)
              
    Total_deductions = Tax_depr + max(Other_deductions,0) + incentive_claimed
    return Total_deductions

@iterate_jit(nopython=True)
def Net_taxable_profit(Gross_taxable_profit, Total_deductions, Net_taxable_profit):
    """
    Compute net taxable profits afer allowing deductions.
    """
    Net_taxable_profit = Gross_taxable_profit - Total_deductions
    #print(Net_taxable_profit)
    return Net_taxable_profit

@iterate_jit(nopython=True)
def Carried_forward_losses(Carried_forward_losses, CF_losses):
    """
    Compute net taxable profits afer allowing deductions.
    """
    CF_losses = Carried_forward_losses
    return CF_losses

@iterate_jit(nopython=True)
def Tax_base_CF_losses(Loss_CFLimit, CF_losses, Net_taxable_profit,
    Loss_lag1, Loss_lag2, Loss_lag3, Loss_lag4, Loss_lag5, Loss_lag6, Loss_lag7, Loss_lag8,
    newloss1, newloss2, newloss3, newloss4, newloss5, newloss6, newloss7, newloss8, Used_loss_total, Tax_base):
    
    """
    Compute net tax base afer allowing donations and losses.
    """
    BF_loss = np.array([Loss_lag1, Loss_lag2, Loss_lag3, Loss_lag4, Loss_lag5, Loss_lag6, Loss_lag7, Loss_lag8])
    #print(BF_loss)
    Gross_Tax_base = Net_taxable_profit
    #print(Gross_Tax_base)
    if BF_loss.sum() == 0:
        BF_loss[0] = CF_losses
    #print(BF_loss)
    N = int(Loss_CFLimit)
    if N == 0:
        (newloss1, newloss2, newloss3, newloss4, newloss5, newloss6, newloss7, newloss8) = np.zeros(8)
        Used_loss_total = 0
        Tax_base = Gross_Tax_base
        
    else:
        BF_loss = BF_loss[:N]
                
        if Gross_Tax_base < 0:
            CYL = abs(Gross_Tax_base)
            Used_loss = np.zeros(N)
        elif Gross_Tax_base >0:
            CYL = 0
            Cum_used_loss = 0
            Used_loss = np.zeros(N)
            for i in range(N, 0, -1):
                GTI = Gross_Tax_base - Cum_used_loss
                Used_loss[i-1] = min(BF_loss[i-1], GTI)
                Cum_used_loss += Used_loss[i-1]
        elif Gross_Tax_base == 0:
            CYL=0
            Used_loss = np.zeros(N)
    
        New_loss = BF_loss - Used_loss
        Tax_base = Gross_Tax_base - Used_loss.sum()
        newloss1 = CYL
        Used_loss_total = Used_loss.sum()
        (newloss2, newloss3, newloss4, newloss5, newloss6, newloss7, newloss8) = np.append(New_loss[:-1], np.zeros(8-N))

    return (Tax_base, newloss1, newloss2, newloss3, newloss4, newloss5, newloss6, newloss7, newloss8, Used_loss_total)

@iterate_jit(nopython=True)
def Net_tax_base_behavior(cit_rate_genbus, cit_rate_genbus_curr_law, elasticity_cit_taxable_income_threshold,
                          elasticity_cit_taxable_income_value, Tax_base, 
                          Net_tax_base_behavior):
    """
    Compute net taxable profits afer allowing deductions.
    """
    NP = Tax_base   
    elasticity_taxable_income_threshold0 = elasticity_cit_taxable_income_threshold[0]
    elasticity_taxable_income_threshold1 = elasticity_cit_taxable_income_threshold[1]
    elasticity_taxable_income_threshold2 = elasticity_cit_taxable_income_threshold[2]
    elasticity_taxable_income_value0=elasticity_cit_taxable_income_value[0]
    elasticity_taxable_income_value1=elasticity_cit_taxable_income_value[1]
    elasticity_taxable_income_value2=elasticity_cit_taxable_income_value[2]
    if NP<=0:
        elasticity=0
    elif NP<elasticity_taxable_income_threshold0:
        elasticity=elasticity_taxable_income_value0
    elif NP<elasticity_taxable_income_threshold1:
        elasticity=elasticity_taxable_income_value1
    else:
        elasticity=elasticity_taxable_income_value2

    frac_change_net_of_cit_rate_genbus = ((1-cit_rate_genbus)-(1-cit_rate_genbus_curr_law))/(1-cit_rate_genbus_curr_law)
    frac_change_Net_tax_base_genbus = elasticity*(frac_change_net_of_cit_rate_genbus)    
    Net_tax_base_behavior = Tax_base*(1+frac_change_Net_tax_base_genbus)
    return Net_tax_base_behavior

DEBUG = False
DEBUG_IDX = 0

@iterate_jit(nopython=True)
def mat_liability_book(mat_rate_book_profit, Net_accounting_profit, MAT_book):
    """
    Compute the minimum tax liability given the turnover
    """
    # subtract TI_special_rates from TTI to get Aggregate_Income, which is
    # the portion of TTI that is taxed at normal rates
    MAT_book = mat_rate_book_profit*Net_accounting_profit       
    return MAT_book

@iterate_jit(nopython=True)
def mat_liability_turnover(mat_rate_turnover, Revenues, 
                           Other_revenues, MAT_turnover):
    """
    Compute the minimum tax liability given the turnover
    """
    # subtract TI_special_rates from TTI to get Aggregate_Income, which is
    # the portion of TTI that is taxed at normal rates
    Total_Turnover = Revenues+Other_revenues
    MAT_turnover = mat_rate_turnover*Total_Turnover       
    return MAT_turnover

@iterate_jit(nopython=True)
def cit_liability(cit_rate_genbus, turnover_tax_threshold, turnover_tax_rate, 
                  mat_applied_on_book_profit, 
                  MAT_book, MAT_turnover, Taxpayer_Type_Code, Revenues, 
                  Other_revenues, Net_tax_base_behavior, Chargeable_Income_Return,
                  MAT, citax):
    """
    Compute tax liability given the corporate rate
    """
    # subtract TI_special_rates from TTI to get Aggregate_Income, which is
    # the portion of TTI that is taxed at normal rates
    taxinc = max(Net_tax_base_behavior, 0)
    if (taxinc == 0) & (Chargeable_Income_Return > 0):
        taxinc=Chargeable_Income_Return
    #taxinc=Chargeable_Income_Return
    Total_Turnover = Revenues+Other_revenues
    if (Taxpayer_Type_Code == 1):
        if (Total_Turnover >= turnover_tax_threshold):
            citax = cit_rate_genbus * taxinc
        else:
            citax = turnover_tax_rate*Total_Turnover  
        if (mat_applied_on_book_profit==1):
            MAT = MAT_book
        else:
            MAT = MAT_turnover
        
        if MAT>citax:
            citax=MAT
    else:
        citax=0    
    return citax




