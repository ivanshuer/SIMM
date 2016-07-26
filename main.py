import pandas as pd
import logging
import os
import numpy as np
import sys
import simm_params
import re

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

    print 'Hello World'

    input_file = 'simm_input.csv'
    trades_pos = pd.read_csv(input_file, dtype = {'Bucket': str, 'Label1': str, 'Label2': str, 'Amount': np.float64, 'AmountUSD': np.float64})

    with open('classification.txt', 'r') as f:
        data = f.readlines()

    data_clean = []
    for line in data:
        data_clean.append(re.sub('^#.*', '', line))

    return

if __name__ == '__main__':
    main()