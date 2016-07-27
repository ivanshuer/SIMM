import numpy as np
import pandas as pd
import os
import logging

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

def risk_classification(trades_pos, params):
    """ Risk class classification in terms of RiskType
    """
    trades_pos_no_product = trades_pos[~trades_pos.ProductClass.isin(params.Product)].copy()
    if len(trades_pos_no_product) > 0:
        logger.info('{0} trades missed Product Class'.format(len(trades_pos_no_product)))
        trades_pos_no_product['reason'] = 'Product'

    trades_pos = trades_pos[trades_pos.ProductClass.isin(params.Product)].copy()

    trades_pos['RiskClass'] = np.NaN
    trades_pos.ix[trades_pos.RiskType.isin(params.IR), 'RiskClass'] = 'IR'
    trades_pos.ix[trades_pos.RiskType.isin(params.CreditQ), 'RiskClass'] = 'CreditQ'
    trades_pos.ix[trades_pos.RiskType.isin(params.CreditNonQ), 'RiskClass'] = 'CreditNonQ'
    trades_pos.ix[trades_pos.RiskType.isin(params.Equity), 'RiskClass'] = 'Equity'
    trades_pos.ix[trades_pos.RiskType.isin(params.FX), 'RiskClass'] = 'FX'
    trades_pos.ix[trades_pos.RiskType.isin(params.Commodity), 'RiskClass'] = 'Commodity'

    trades_pos_no_risk_class = trades_pos[~trades_pos.RiskClass.isin(params.RiskType)].copy()
    if len(trades_pos_no_risk_class) > 0:
        logger.info('{0} trades can not be classified for risk class'.format(len(trades_pos_no_risk_class)))
        trades_pos_no_risk_class['reason'] = 'RiskType'

    trades_pos = trades_pos[trades_pos.RiskClass.isin(params.RiskType)].copy()

    trades_pos_no_qualifier = trades_pos[trades_pos.Qualifier.isnull()].copy()
    if len(trades_pos_no_qualifier) > 0:
        logger.info('{0} trades missed Qualifiers'.format(len(trades_pos_no_qualifier)))
        trades_pos_no_risk_class['reason'] = 'Qualifiers'

    trades_pos = trades_pos[trades_pos.Qualifier.notnull()].copy()
    trades_pos['reason'] = 'Good'

    trades_pos = pd.concat([trades_pos, trades_pos_no_product, trades_pos_no_risk_class, trades_pos_no_qualifier])

    return trades_pos
