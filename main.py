import pandas as pd
import logging
import os
import numpy as np
import sys
import params
import simm_lib
import re
import ir_risk

##############################
# Setup Logging Configuration
##############################
logger = logging.getLogger(os.path.basename(__file__))
if not len(logger.handlers):
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s|%(name)s === %(message)s ===', datefmt='%Y-%m-%d %I:%M:%S')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
###############################


def main():
    logger.info('Main program')

    input_file = 'simm_input.csv'
    trades_pos = pd.read_csv(input_file, dtype = {'Bucket': str, 'Label1': str, 'Label2': str, 'Amount': np.float64, 'AmountUSD': np.float64})

    trades_pos = simm_lib.risk_classification(trades_pos, params)
    trades_pos_no_classification = trades_pos[trades_pos.reason != 'Good'].copy()
    trades_pos = trades_pos[trades_pos.reason == 'Good'].copy()

    trades_pos_IR = trades_pos[trades_pos.RiskClass == 'IR'].copy()
    ir = ir_risk.InterestRate()
    trades_pos_IR = ir.prep_data(trades_pos_IR, params)

    trades_pos_all = pd.concat([trades_pos_IR, trades_pos_no_classification])
    trades_pos_all.to_csv('all_trades_pos.csv', index=False)

    trades_simm = trades_pos_all[trades_pos_all.reason == 'Good'].copy()


    return

if __name__ == '__main__':
    main()