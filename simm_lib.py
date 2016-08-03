import numpy as np
import pandas as pd
import os
import logging
import math
import shutil
import delta_margin
import vega_margin

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

def prep_output_directory(params):
    for prod in params.Product:
        output_path = '{0}\{1}'.format(os.getcwd(), prod)
        if os.path.exists(output_path):
            shutil.rmtree(output_path)

        os.mkdir(output_path)

    for prod in params.Product:
        for risk in params.RiskType:
            output_path = '{0}\{1}\{2}'.format(os.getcwd(), prod, risk)
            if not os.path.exists(output_path):
                os.mkdir(output_path)

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

def prep_data_IRCurve(pos, params):
    """Check data quality for IR Curve factor"""

    # Check Bucket
    pos_no_bucket = pos[~pos.Bucket.isin(params.IR_Bucket)].copy()
    if len(pos_no_bucket) > 0:
        logger.info('{0} IR Curve trades have wrong Bucket'.format(len(pos_no_bucket)))
        pos_no_bucket['reason'] = 'Bucket'

    pos = pos[pos.Bucket.isin(params.IR_Bucket)].copy()

    # Check Label1
    pos_no_label1 = pos[~pos.Label1.isin(params.IR_Tenor)].copy()
    if len(pos_no_label1) > 0:
        logger.info('{0} IR Curve trades have wrong Label 1'.format(len(pos_no_label1)))
        pos_no_label1['reason'] = 'Label1'

    pos = pos[pos.Label1.isin(params.IR_Tenor)].copy()

    # Check Label2
    pos_no_label2 = pos[~(((pos.Qualifier == 'USD') & pos.Label2.isin(params.IR_USD_Sub_Curve)) |
                          ((pos.Qualifier != 'USD') & pos.Label2.isin(params.IR_Sub_Curve)))].copy()

    if len(pos_no_label2) > 0:
        logger.info('{0} IR Curve trades have wrong Label 2'.format(len(pos_no_label2)))
        pos_no_label2['reason'] = 'Label2'

    pos = pos[((pos.Qualifier == 'USD') & pos.Label2.isin(params.IR_USD_Sub_Curve)) |
              ((pos.Qualifier != 'USD') & pos.Label2.isin(params.IR_Sub_Curve))].copy()

    pos = pd.concat([pos, pos_no_bucket, pos_no_label1, pos_no_label2])

    return pos

def prep_data_IRVol(pos, params):
    """Check data quality for IR Vol factor"""

    # Check Label1
    pos_no_label1 = pos[~pos.Label1.isin(params.IR_Tenor)].copy()
    if len(pos_no_label1) > 0:
        logger.info('{0} IR Vol trades have wrong Label 1'.format(len(pos_no_label1)))
        pos_no_label1['reason'] = 'Label1'

    pos = pos[pos.Label1.isin(params.IR_Tenor)].copy()

    pos = pd.concat([pos, pos_no_label1])

    return pos

def prep_data_IR(pos, params):
    pos_IRCurve = pos[pos.RiskType == 'Risk_IRCurve'].copy()
    pos_IRCurve = prep_data_IRCurve(pos_IRCurve, params)

    pos_IRVol = pos[pos.RiskType == 'Risk_IRVol'].copy()
    pos_IRVol = prep_data_IRVol(pos_IRVol, params)

    pos_Inflation = pos[pos.RiskType == 'Risk_Inflation'].copy()

    pos = pd.concat([pos_IRCurve, pos_IRVol, pos_Inflation])

    return pos

def prep_data_CreditQ(pos, params):
    """Check data quality for CreditQ factor"""

    # Check Bucket
    pos_no_bucket = pos[~pos.Bucket.isin(params.CreditQ_Bucket)].copy()
    if len(pos_no_bucket) > 0:
        logger.info('{0} CreditQ trades have wrong Bucket'.format(len(pos_no_bucket)))
        pos_no_bucket['reason'] = 'Bucket'

    pos = pos[pos.Bucket.isin(params.CreditQ_Bucket)].copy()

    # Check Label1
    pos_no_label1 = pos[~pos.Label1.isin(params.CreditQ_Tenor)].copy()
    if len(pos_no_label1) > 0:
        logger.info('{0} CreditQ trades have wrong Label 1'.format(len(pos_no_label1)))
        pos_no_label1['reason'] = 'Label1'

    pos = pos[pos.Label1.isin(params.CreditQ_Tenor)].copy()

    pos = pd.concat([pos, pos_no_bucket, pos_no_label1])

    return pos

def prep_data_CreditNonQ(pos, params):
    """Check data quality for CreditNonQ factor"""

    # Check Bucket
    pos_no_bucket = pos[~pos.Bucket.isin(params.CreditNonQ_Bucket)].copy()
    if len(pos_no_bucket) > 0:
        logger.info('{0} CreditNonQ trades have wrong Bucket'.format(len(pos_no_bucket)))
        pos_no_bucket['reason'] = 'Bucket'

    pos = pos[pos.Bucket.isin(params.CreditNonQ_Bucket)].copy()

    # Check Label1
    pos_no_label1 = pos[~pos.Label1.isin(params.CreditNonQ_Tenor)].copy()
    if len(pos_no_label1) > 0:
        logger.info('{0} CreditNonQ trades have wrong Label 1'.format(len(pos_no_label1)))
        pos_no_label1['reason'] = 'Label1'

    pos = pos[pos.Label1.isin(params.CreditNonQ_Tenor)].copy()

    pos = pd.concat([pos, pos_no_bucket, pos_no_label1])

    return pos

def prep_data_Equity(pos, params):
    """Check data quality for Equity factor"""

    # Check Bucket
    pos_no_bucket = pos[~pos.Bucket.isin(params.Equity_Bucket)].copy()
    if len(pos_no_bucket) > 0:
        logger.info('{0} Equity trades have wrong Bucket'.format(len(pos_no_bucket)))
        pos_no_bucket['reason'] = 'Bucket'

    pos = pos[pos.Bucket.isin(params.Equity_Bucket)].copy()

    pos = pd.concat([pos, pos_no_bucket])

    return pos

def prep_data_Commodity(pos, params):
    """Check data quality for Commodity factor"""

    # Check Bucket
    pos_no_bucket = pos[~pos.Bucket.isin(params.Commodity_Bucket)].copy()
    if len(pos_no_bucket) > 0:
        logger.info('{0} Commodity trades have wrong Bucket'.format(len(pos_no_bucket)))
        pos_no_bucket['reason'] = 'Bucket'

    pos = pos[pos.Bucket.isin(params.Commodity_Bucket)].copy()

    pos = pd.concat([pos, pos_no_bucket])

    return pos

def prep_data(pos, params):
    """Check data quality for all risk factors"""

    pos_IR = pos[pos.RiskClass == 'IR'].copy()
    pos_IR = prep_data_IR(pos_IR, params)

    pos_CreditQ = pos[pos.RiskClass == 'CreditQ'].copy()
    pos_CreditQ = prep_data_CreditQ(pos_CreditQ, params)

    pos_CreditNonQ = pos[pos.RiskClass == 'CreditNonQ'].copy()
    pos_CreditNonQ = prep_data_CreditNonQ(pos_CreditNonQ, params)

    pos_Equity = pos[pos.RiskClass == 'Equity'].copy()
    pos_Equity = prep_data_Equity(pos_Equity, params)

    pos_Commodity = pos[pos.RiskClass == 'Commodity'].copy()
    pos_Commodity = prep_data_Commodity(pos_Commodity, params)

    pos_FX = pos[pos.RiskClass == 'FX'].copy()

    return pd.concat([pos_IR, pos_CreditQ, pos_CreditNonQ, pos_Equity, pos_Commodity, pos_FX])

def calc_delta_margin(pos, params):
    pos_delta = pos[pos.RiskType.isin(params.Delta_Factor)].copy()

    pos_delta_margin_gp = []
    pos_delta_margin = []

    if len(pos_delta) > 0:
        delta_margin_loader = delta_margin.DeltaMargin()
        pos_delta_margin = delta_margin_loader.margin_risk_factor(pos_delta, params)

    if len(pos_delta_margin) > 0:
        pos_delta_margin_gp = pos_delta_margin.groupby(['ProductClass', 'RiskClass'])
        pos_delta_margin_gp = pos_delta_margin_gp.agg({'Margin': np.sum})
        pos_delta_margin_gp.reset_index(inplace=True)
        pos_delta_margin_gp['MarginType'] = 'Delta'

    return pos_delta_margin_gp

def calc_vega_margin(pos, params):
    pos_vega = pos[pos.RiskType.isin(params.Vega_Factor)].copy()

    pos_vega_margin_gp = []
    pos_vega_margin = []

    if len(pos_vega) > 0:
        vega_margin_loader = vega_margin.VegaMargin()
        pos_vega_margin = vega_margin_loader.margin_risk_factor(pos_vega, params)

    if len(pos_vega_margin) > 0:
        pos_vega_margin_gp = pos_vega_margin.groupby(['ProductClass', 'RiskClass'])
        pos_vega_margin_gp = pos_vega_margin_gp.agg({'Margin': np.sum})
        pos_vega_margin_gp.reset_index(inplace=True)
        pos_vega_margin_gp['MarginType'] = 'Vega'

    return pos_vega_margin_gp

def calculate_simm(pos, params):

    product_margin = []

    for product in pos.ProductClass.unique():
        for risk in pos.RiskClass.unique():
            logger.info('Calcualte SIMM for {0} and {1}'.format(product, risk))
            pos_product = pos[(pos.ProductClass == product) & (pos.RiskClass == risk)].copy()

            pos_gp_delta_margin = calc_delta_margin(pos_product, params)
            if len(pos_gp_delta_margin) > 0:
                product_margin.append(pos_gp_delta_margin)

            pos_gp_vega_margin = calc_vega_margin(pos_product, params)
            if len(pos_gp_vega_margin) > 0:
                product_margin.append(pos_gp_vega_margin)

    product_margin = pd.concat(product_margin)


    return product_margin










