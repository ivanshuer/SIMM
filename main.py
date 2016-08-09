import pandas as pd
import logging
import os
import numpy as np
import params
import simm_lib
import argparse

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
    # Setup input argument
    parser = argparse.ArgumentParser(description='SIMM Calculation.')
    parser.add_argument('-f', dest='input_file', type=str, required=True, help='simm input csv file')
    args = parser.parse_args()

    # Create output directory for product and risk class
    simm_lib.prep_output_directory(params)

    # Read input file with specified data type
    #input_file = 'simm_input_1.csv'
    input_file = args.input_file
    trades_pos = pd.read_csv(input_file, dtype = {'Bucket': str, 'Label1': str, 'Label2': str, 'Amount': np.float64, 'AmountUSD': np.float64})

    # Calculate risk classification
    trades_pos = simm_lib.risk_classification(trades_pos, params)
    trades_pos_no_classification = trades_pos[trades_pos.reason != 'Good'].copy()
    trades_pos = trades_pos[trades_pos.reason == 'Good'].copy()

    # Check input data quality
    trades_pos_all = simm_lib.prep_data(trades_pos, params)
    trades_pos_all = pd.concat([trades_pos_all, trades_pos_no_classification])
    trades_pos_all.to_csv('all_trades_pos.csv', index=False)

    # Prepare input data
    trades_simm = trades_pos_all[trades_pos_all.reason == 'Good'].copy()
    trades_simm = trades_simm[['ProductClass', 'RiskType', 'Qualifier', 'Bucket', 'Label1', 'Label2', 'AmountUSD', 'RiskClass']].copy()
    trades_simm.AmountUSD.fillna(0, inplace=True)

    # Calculate SIMM and dump output
    #pos = trades_simm[trades_simm.RiskClass == 'CreditQ'].copy()
    simm = simm_lib.calculate_simm(trades_simm, params)
    simm = pd.DataFrame([simm], columns=['SIMM'])
    simm.to_csv('simm_output.csv', index=False)

    return

if __name__ == '__main__':
    main()