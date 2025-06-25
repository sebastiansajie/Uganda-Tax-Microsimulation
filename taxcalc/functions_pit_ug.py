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
def cal_gross_income(housing_allowance_exempted, transport_allowance_exempted, 
                     medical_allowance_exempted, leave_allowance_exempted, 
                     over_time_allowance_exempted, other_taxable_allowance_exempted, 
                     housing_fringe_exempted, motor_vehicle_allowance_exempted,
                     domestic_servants_allowance_exempted,other_taxable_fringe_benefits_exempted,
                     sch1_basic_salary, sch1_housing_allowance,sch1_transport,
                     sch1_medical,sch1_leave,sch1_over_time,sch1_other_taxable_allow,
                     sch1_housing,sch1_motor_vehicle,sch1_domestic_servants,
                     sch1_other_taxable_benf,GTI):
    """
    Compute gross income from wages.
    """
    GTI = (sch1_basic_salary+sch1_housing_allowance*(1-housing_allowance_exempted)+
                  sch1_transport*(1-transport_allowance_exempted)+
                  sch1_medical*(1-medical_allowance_exempted)+
                  sch1_leave*(1-leave_allowance_exempted)+
                  sch1_over_time*(1-over_time_allowance_exempted)+
                  sch1_other_taxable_allow*(1-other_taxable_allowance_exempted)+
                  sch1_housing*(1-housing_fringe_exempted)+
                  sch1_motor_vehicle*(1-motor_vehicle_allowance_exempted)+
                  sch1_domestic_servants*(1-domestic_servants_allowance_exempted)+
                  sch1_other_taxable_benf*(1-other_taxable_fringe_benefits_exempted))

    return GTI

@iterate_jit(nopython=True)
def calc_taxable_income(standard_deduction, deductions_allowed, GTI, sch1_allowable_deductions, taxable_income):
    """
    This function calculates the taxable income.
    """
    taxable_income = GTI - standard_deduction - sch1_allowable_deductions*deductions_allowed
    return taxable_income

@iterate_jit(nopython=True)
def cal_pit(rate1, rate2, rate3, rate4, rate5, rate6, rate7, 
              tbrk1, tbrk2, tbrk3, tbrk4, tbrk5, tbrk6, 
              taxable_income, pitax):
    """
    Compute PIT.
    """
    taxinc = taxable_income

    pitax = (rate1 * min(taxinc, tbrk1) +
             rate2 * min(tbrk2 - tbrk1, max(0., taxinc - tbrk1)) +
             rate3 * min(tbrk3 - tbrk2, max(0., taxinc - tbrk2)) +
             rate4 * min(tbrk4 - tbrk3, max(0., taxinc - tbrk3)) +
             rate5 * min(tbrk5 - tbrk4, max(0., taxinc - tbrk4)) +
             rate6 * min(tbrk6 - tbrk5, max(0., taxinc - tbrk5)) +             
             rate7 * max(0., taxinc - tbrk6))

    return pitax
