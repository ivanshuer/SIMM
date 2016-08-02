import pandas as pd
import logging
import os
import numpy as np
import sys
import params
import simm_lib
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

    simm_lib.prep_output_directory(params)

    input_file = 'simm_input_1.csv'
    trades_pos = pd.read_csv(input_file, dtype = {'Bucket': str, 'Label1': str, 'Label2': str, 'Amount': np.float64, 'AmountUSD': np.float64})

    trades_pos = simm_lib.risk_classification(trades_pos, params)
    trades_pos_no_classification = trades_pos[trades_pos.reason != 'Good'].copy()
    trades_pos = trades_pos[trades_pos.reason == 'Good'].copy()

    trades_pos_all = simm_lib.prep_data(trades_pos, params)
    trades_pos_all = pd.concat([trades_pos_all, trades_pos_no_classification])
    trades_pos_all.to_csv('all_trades_pos.csv', index=False)

    trades_simm = trades_pos_all[trades_pos_all.reason == 'Good'].copy()
    trades_simm = trades_simm[['ProductClass', 'RiskType', 'Qualifier', 'Bucket', 'Label1', 'Label2', 'AmountUSD', 'RiskClass']].copy()
    trades_simm.AmountUSD.fillna(0, inplace=True)

    #pos = trades_simm[trades_simm.RiskClass == 'CreditQ'].copy()
    t1 = simm_lib.calculate_simm(trades_simm, params)
    t1.to_csv('1.csv', index=False)

    return

if __name__ == '__main__':
    main()