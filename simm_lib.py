import numpy as np
import pandas as pd
import os
import logging
import math

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
        if not os.path.exists(output_path):
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
    pos_CreditQ = prep_data_IR(pos_CreditQ, params)

    pos_CreditNonQ = pos[pos.RiskClass == 'CreditNonQ'].copy()
    pos_CreditNonQ = prep_data_IR(pos_CreditNonQ, params)

    pos_Equity = pos[pos.RiskClass == 'Equity'].copy()
    pos_Equity = prep_data_IR(pos_Equity, params)

    pos_Commodity = pos[pos.RiskClass == 'Commodity'].copy()
    pos_Commodity = prep_data_IR(pos_Commodity, params)

    pos_FX = pos[pos.RiskClass == 'FX'].copy()

    return pd.concat([pos_IR, pos_CreditQ, pos_CreditNonQ, pos_Equity, pos_Commodity, pos_FX])

def net_sensitivities(pos, params):
    risk_class = pos.RiskClass.unique()[0]

    if risk_class == 'IR':
        factor_group = ['ProductClass', 'RiskType', 'Qualifier', 'Bucket', 'Label1', 'Label2', 'RiskClass']
    elif risk_class in ['CreditQ', 'CreditNonQ']:
        factor_group = ['ProductClass', 'RiskType', 'Qualifier', 'Bucket', 'Label1', 'RiskClass']
    else:
        factor_group = ['ProductClass', 'RiskType', 'Qualifier', 'Bucket', 'RiskClass']

    pos_gp = pos.groupby(factor_group)
    pos_delta = pos_gp.agg({'AmountUSD': np.sum})
    pos_delta.reset_index(inplace=True)

    return pos_delta

def find_factor_idx(tenor_factor, curve_factor, tenors, curves, risk_class):
    idx = 0

    if risk_class == 'IR':
        for tenor in tenors:
            for curve in curves:
                if tenor_factor == tenor and curve_factor == curve:
                    return idx
                else:
                    idx = idx + 1
    elif risk_class in ['CreditQ', 'CreditNonQ']:
        for tenor in tenors:
            if tenor_factor == tenor:
                return idx
            else:
                idx = idx + 1

    return -1

def build_risk_factors(pos_gp, params):

    risk_class = pos_gp.RiskClass.unique()[0]

    if risk_class == 'IR':
        gp_curr = pos_gp.Qualifier.unique()[0]

        curve = params.IR_Sub_Curve
        if gp_curr == 'USD':
            curve = params.IR_USD_Sub_Curve

        s = np.zeros(len(params.IR_Tenor) * len(curve))

        for i, row in pos_gp.iterrows():
            idx = find_factor_idx(row['Label1'], row['Label2'], params.IR_Tenor, curve, risk_class)
            if idx >= 0:
                s[idx] = row['AmountUSD']

    elif risk_class == 'CreditQ':
        s = np.zeros(len(pos_gp.Qualifier.unique()) * len(params.CreditQ_Tenor))

        for i, row in pos_gp.iterrows():
            idx = find_factor_idx(row['Label1'], [], params.CreditQ_Tenor, [], risk_class)
            if idx >= 0:
                s[idx] = row['AmountUSD']

    elif risk_class == 'CreditNonQ':
        s = np.zeros(len(pos_gp.Qualifier.unique()) * len(params.CreditNonQ_Tenor))

        for i, row in pos_gp.iterrows():
            idx = find_factor_idx(row['Label1'], [], params.CreditNonQ_Tenor, [], risk_class)
            if idx >= 0:
                s[idx] = row['AmountUSD']

    else:
        s = np.zeros(len(pos_gp.Qualifier.unique()))

        for i, row in pos_gp.iterrows():
            s[i] = row['AmountUSD']

    return s

def build_concentration_risk(pos_gp, params):
    risk_class = pos_gp.RiskClass.unique()
    bucket = pos_gp.Bucket.unique()

    f = lambda x: max(1, math.sqrt(abs(x) / Tb))

    if risk_class == 'IR':
        Tb = params.IR_G10_DKK_Threshold

        gp_curr = pos_gp.Qualifier.unique()[0]

        if not gp_curr in params.G10_Curr:
            Tb = params.IR_Other_Threshold

        CR = max(1, math.sqrt(abs(pos_gp.AmountUSD.sum() / Tb)))

    elif risk_class in ['CreditQ', 'CreditNonQ']:
        if risk_class == 'CreditQ':
            Tb = params.CreditQ_Threshold
            curves = params.CreditQ_Tenor
        elif risk_class == 'CreditNonQ':
            Tb = params.CreditNonQ_Threshold
            curves = params.CreditNonQ_Tenor

        pos_qualifier_gp = pos_gp.groupby(['Qualifier'])
        pos_qualifier_gp = pos_qualifier_gp.agg({'AmountUSD': np.sum})
        pos_qualifier_gp.reset_index(inplace=True)

        CR = pos_qualifier_gp.AmountUSD.apply(f)
        CR = np.repeat(CR.values, len(curves))

    else:
        if risk_class == 'Equity':
            if bucket in params.Equity_EM:
                Tb = params.Equity_EM_Threshold
            elif bucket in params.Equity_DEVELOPED:
                Tb = params.Equity_DEVELOPED_Threshold
            elif bucket in params.Equity_INDEX:
                Tb = params.Equity_INDEX_Threshold
        elif risk_class == 'Commodity':
            if bucket in params.Commodity_FUEL:
                Tb = params.Commodity_FUEL_Threshold
            elif bucket in params.Commodity_POWER:
                Tb = params.Commodity_POWER_Threshold
            elif bucket in params.Commodity_OTHER:
                Tb = params.Commodity_OTHER_Threshold
        elif risk_class == 'FX':
            Tb = params.FX_Threshold

        CR = pos_gp.AmountUSD.apply(f)
        CR = CR.values

    return CR

def build_risk_weights(pos_gp, params):
    risk_class = pos_gp.RiskClass.unique()

    if risk_class == 'IR':
        bucket = pd.DataFrame(pos_gp.Bucket.unique(), columns=['curr_type'])
        RW = pd.merge(bucket, params.IR_Weights, left_on=['curr_type'], right_on=['curr'], how='inner')
        RW = RW.drop(['curr_type', 'curr'], axis=1)
        RW = RW.as_matrix()

        gp_curr = pos_gp.Qualifier.unique()[0]

        curve = params.IR_Sub_Curve
        if gp_curr == 'USD':
            curve = params.IR_USD_Sub_Curve

        RW = np.repeat(RW, len(curve))
    else:
        if risk_class == 'CreditQ':
            weights = params.CreditQ_Weights
            num_factors = len(pos_gp.Qualifier) * len(params.CreditQ_Tenor)
        elif risk_class == 'CreditNonQ':
            weights = params.CreditNonQ_Weights
            num_factors = len(pos_gp.Qualifier) * len(params.CreditNonQ_Tenor)
        elif risk_class == 'Equity':
            weights = params.Equity_Weights
            num_factors = len(pos_gp.Qualifier)
        elif risk_class == 'Commodity':
            weights = params.Commodity_Weights
            num_factors = len(pos_gp.Qualifier)
        elif risk_class == 'FX':
            weights = params.FX_Weights
            num_factors = len(pos_gp.Qualifier)

        if risk_class != 'FX':
            bucket = pd.DataFrame(pos_gp.Bucket.unique(), columns=['bucket'])
            RW = pd.merge(bucket, weights, left_on=['bucket'], right_on=['bucket'], how='inner')
            RW = np.array(RW.weight.values[0])
        else:
            RW = np.array([weights])

        RW = np.repeat(RW, num_factors)

    return RW

def delta_margin_IR_risk_factor(pos, params):
    """Calculate Delta Margin for IR Class"""

    pos_delta = net_sensitivities(pos, params)

    product_class = pos_delta.ProductClass.unique()[0]
    risk_class = pos_delta.RiskClass.unique()[0]
    risk_type = pos_delta.RiskType.unique()[0]

    def find_factor_idx(tenor_factor, curve_factor, tenors, curves, risk_class):
        idx = 0

        for tenor in tenors:
            for curve in curves:
                if tenor_factor == tenor and curve_factor == curve:
                    return idx
                else:
                    idx = idx + 1

        return -1

    def delta_margin_currency(gp):
        logger.info('Calculate Delta Margin for {0}'.format(gp.Qualifier.unique()))

        Tb = params.IR_G10_DKK_Threshold

        gp_curr = gp.Qualifier.unique()[0]

        if not gp_curr in params.G10_Curr:
            Tb = params.IR_Other_Threshold

        CR = max(1, math.sqrt(abs(gp.AmountUSD.sum() / Tb)))

        curve = params.IR_Sub_Curve
        if gp_curr == 'USD':
            curve = params.IR_USD_Sub_Curve

        s = np.zeros(len(params.IR_Tenor) * len(curve))
        bucket = pd.DataFrame(gp.Bucket.unique(), columns=['curr_type'])
        RW = pd.merge(bucket, params.IR_Weights, left_on=['curr_type'], right_on=['curr'], how='inner')
        RW = RW.drop(['curr_type', 'curr'], axis=1)
        RW = RW.as_matrix()
        RW = np.repeat(RW, len(curve))

        if gp.RiskType.unique()[0] == 'Risk_Inflation':
            RW.fill(params.IR_Inflation_Weights)

        for i, row in gp.iterrows():
            idx = find_factor_idx(row['Label1'], row['Label2'], params.IR_Tenor, curve, risk_class)
            if idx >= 0:
                s[idx] = row['AmountUSD']

        WS = RW * s * CR

        fai = np.zeros((len(curve), len(curve)))
        fai.fill(params.IR_Fai)
        np.fill_diagonal(fai, 1)

        rho = params.IR_Corr
        if gp.RiskType.unique()[0] == 'Risk_Inflation':
            rho = np.zeros(params.IR_Corr.shape)
            rho.fill(params.IR_Inflation_Rho)
            np.fill_diagonal(rho, 1)

        Corr = np.kron(rho, fai)

        K = np.mat(WS) * np.mat(Corr) * np.mat(np.reshape(WS, (len(WS), 1)))
        K = math.sqrt(K.item(0))

        ret = gp[['ProductClass', 'RiskType', 'Qualifier', 'RiskClass']].copy()
        ret['S'] = max(min(WS.sum(), K), -K)
        ret['CR'] = CR
        ret['WeightDelta'] = K

        return ret

    pos_delta = pos_delta.groupby(['Qualifier']).apply(delta_margin_currency)
    pos_delta.reset_index(inplace=True, drop=True)

    intermediate_path = '{0}\{1}\{2}'.format(os.getcwd(), product_class, risk_class)
    pos_delta.to_csv('{0}\{1}_delta_margin_all_curs.csv'.format(intermediate_path, risk_type), index=False)

    all_curr = pos_delta.Qualifier.unique()
    g = np.zeros((len(all_curr), len(all_curr)))
    for i in range(len(all_curr)):
        for j in range(len(all_curr)):
            CRi = pos_delta[pos_delta.Qualifier == all_curr[i]].CR[0]
            CRj = pos_delta[pos_delta.Qualifier == all_curr[j]].CR[0]

            g[i][j] = min(CRi, CRj) / max(CRi, CRj)

    np.fill_diagonal(g, 0)
    S = pos_delta.S

    SS = np.mat(S) * np.mat(g) * np.mat(np.reshape(S, (len(S), 1))) * params.IR_Gamma
    SS = SS.item(0)
    delta_margin = math.sqrt(np.dot(pos_delta.WeightDelta, pos_delta.WeightDelta) + SS)

    ret_mm = pos_delta[['ProductClass', 'RiskClass']].copy()
    ret_mm['DeltaMargin'] = delta_margin

    return ret_mm

def delta_margin_IR(pos, params):
    pos_curve = pos[pos.RiskType == 'Risk_IRCurve'].copy()

    pos_delta_margin = []

    if len(pos_curve) > 0:
        curve_delta_margin = delta_margin_IR_risk_factor(pos_curve, params)
        pos_delta_margin = pd.concat([curve_delta_margin])

    pos_inflation = pos[pos.RiskType == 'Risk_Inflation'].copy()
    if len(pos_inflation) > 0:
        inflation_delta_margin = delta_margin_IR_risk_factor(pos_inflation, params)
        pos_delta_margin = pd.concat([pos_delta_margin, inflation_delta_margin])

    if len(pos_delta_margin):
        pos_delta_margin_gp = pos_delta_margin.groupby(['ProductClass', 'RiskClass'])
        pos_delta_margin_gp = pos_delta_margin_gp.agg({'DeltaMargin': np.sum})
        pos_delta_margin_gp.reset_index(inplace=True)

    return pos_delta_margin_gp










