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
def cal_total_gross_income(income_wage_l, income_dividends_c, income_interest_c,     
                           total_gross_income):
    """
    Compute total gross income.
    """
    total_gross_income = (income_wage_l + income_dividends_c + income_interest_c)
    return total_gross_income

@iterate_jit(nopython=True)
def cal_total_pit(pitax1, pitax):
    """
    Compute PIT.
    """
    pitax = pitax1
    return pitax
