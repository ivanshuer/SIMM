import numpy as np
import pandas as pd
import os
import logging
import math
import shutil

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

def net_sensitivities(pos, params):
    risk_class = pos.RiskClass.unique()[0]

    if risk_class == 'IR':
        factor_group = ['ProductClass', 'RiskType', 'Qualifier', 'Bucket', 'Label1', 'Label2', 'RiskClass']
    elif risk_class in ['CreditQ', 'CreditNonQ']:
        factor_group = ['ProductClass', 'RiskType', 'Qualifier', 'Bucket', 'Label1', 'RiskClass']
    elif risk_class in ['Equity', 'Commodity']:
        factor_group = ['ProductClass', 'RiskType', 'Qualifier', 'Bucket', 'RiskClass']
    elif risk_class == 'FX':
        factor_group = ['ProductClass', 'RiskType', 'Qualifier', 'RiskClass']

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
                s[idx + i*len(params.CreditQ_Tenor)] = row['AmountUSD']

    elif risk_class == 'CreditNonQ':
        s = np.zeros(len(pos_gp.Qualifier.unique()) * len(params.CreditNonQ_Tenor))

        for i, row in pos_gp.iterrows():
            idx = find_factor_idx(row['Label1'], [], params.CreditNonQ_Tenor, [], risk_class)
            if idx >= 0:
                s[idx+ i*len(params.CreditNonQ_Tenor)] = row['AmountUSD']

    else:
        s = np.zeros(len(pos_gp.Qualifier.unique()))

        for i, row in pos_gp.iterrows():
            s[i] = row['AmountUSD']

    return s

def build_concentration_risk(pos_gp, params):
    risk_class = pos_gp.RiskClass.unique()[0]
    bucket = pos_gp.Bucket.unique()[0]

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
    risk_class = pos_gp.RiskClass.unique()[0]

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

def build_in_bucket_correlation(pos_gp, params):
    risk_class = pos_gp.RiskClass.unique()[0]
    bucket = pos_gp.Bucket.unique()[0]

    if risk_class == 'IR':
        gp_curr = pos_gp.Qualifier.unique()[0]
        curve = params.IR_Sub_Curve
        if gp_curr == 'USD':
            curve = params.IR_USD_Sub_Curve

        fai = np.zeros((len(curve), len(curve)))
        fai.fill(params.IR_Fai)
        np.fill_diagonal(fai, 1)

        rho = params.IR_Corr

        Corr = np.kron(rho, fai)
    else:
        num_qualifiers = len(pos_gp.Qualifier.unique())

        CR = build_concentration_risk(pos_gp, params)

        F = np.zeros((len(CR), len(CR)))

        for i in range(len(CR)):
            for j in range(len(CR)):
                CRi = CR[i]
                CRj = CR[j]

                F[i][j] = min(CRi, CRj) / max(CRi, CRj)

        if risk_class in ['CreditQ', 'CreditNonQ']:
            if risk_class == 'CreditQ':
                same_is_rho = params.CreditQ_Rho_Agg_Same_IS
                diff_is_rho = params.CreditQ_Rho_Agg_Diff_IS
                if bucket == 'Residual':
                    same_is_rho = params.CreditQ_Rho_Res_Same_IS
                    diff_is_rho = params.CreditQ_Rho_Res_Diff_IS
            else:
                same_is_rho = params.CreditNonQ_Rho_Agg_Same_IS
                diff_is_rho = params.CreditNonQ_Rho_Agg_Diff_IS
                if bucket == 'Residual':
                    same_is_rho = params.CreditNonQ_Rho_Res_Same_IS
                    diff_is_rho = params.CreditNonQ_Rho_Res_Diff_IS

            rho = np.ones((num_qualifiers, num_qualifiers)) * diff_is_rho
            np.fill_diagonal(rho, same_is_rho)

            one_mat = np.ones((len(params.CreditQ_Tenor), len(params.CreditQ_Tenor)))
            rho = np.kron(rho, one_mat)

        elif risk_class in ['Equity', 'Commodity']:
            bucket_df = pd.DataFrame(pos_gp.Bucket.unique(), columns=['bucket'])

            if risk_class == 'Equity':
                bucket_params = params.Equity_Rho
            elif risk_class == 'Commodity':
                bucket_params = params.Commodity_Rho

            rho = pd.merge(bucket_df, bucket_params, left_on=['bucket'], right_on=['bucket'], how='inner')
            rho = rho['corr'][0]

        elif risk_class == 'FX':
            rho = params.FX_Rho

        Corr = rho * F
        np.fill_diagonal(Corr, 1)

    return Corr

def build_bucket_correlation(pos_delta, params):
    risk_class = pos_delta.RiskClass.unique()[0]

    g = 0

    if risk_class == 'IR':
        all_curr = pos_delta.Group.unique()
        g = np.zeros((len(all_curr), len(all_curr)))
        for i in range(len(all_curr)):
            for j in range(len(all_curr)):
                CRi = pos_delta[pos_delta.Group == all_curr[i]].CR[0]
                CRj = pos_delta[pos_delta.Group == all_curr[j]].CR[0]

                g[i][j] = min(CRi, CRj) / max(CRi, CRj)

        g = g * params.IR_Gamma
    elif risk_class == 'CreditQ':
        g = params.CreditQ_Corr
    elif risk_class == 'CreditNonQ':
        g = params.CreditNonQ_Corr
    elif risk_class == 'Equity':
        g = params.Equity_Corr
    elif risk_class == 'Commodity':
        g = params.Commodity_Corr

    g = np.mat(g)
    np.fill_diagonal(g, 0)

    return g

def build_non_residual_S(pos_gp, params):
    risk_class = pos_gp.RiskClass.unique()[0]

    if risk_class == 'IR':
        S = pos_gp.S
    elif risk_class in ['CreditQ', 'CreditNonQ', 'Equity', 'Commodity']:
        if risk_class == 'CreditQ':
            S = np.zeros(len(params.CreditQ_Bucket) - 1)

            for i in range(len(params.CreditQ_Bucket) - 1):
                for j in range(len(pos_gp.Group)):
                    if pos_gp.Group[j] == params.CreditQ_Bucket[i]:
                        S[i] = pos_gp.S[j]
                        break

        elif risk_class == 'CreditNonQ':
            S = np.zeros(len(params.CreditNonQ_Bucket) - 1)

            for i in range(len(params.CreditNonQ_Bucket) - 1):
                for j in range(len(pos_gp.Group)):
                    if pos_gp.Group[j] == params.CreditNonQ_Bucket[i]:
                        S[i] = pos_gp.S[j]
                        break

        elif risk_class == 'Equity':
            S = np.zeros(len(params.Equity_Bucket) - 1)

            for i in range(len(params.Equity_Bucket) - 1):
                for j in range(len(pos_gp.Group)):
                    if pos_gp.Group[j] == params.Equity_Bucket[i]:
                        S[i] = pos_gp.S[j]
                        break

        elif risk_class == 'Commodity':
            S = np.zeros(len(params.Commodity_Bucket))

            for i in range(len(params.Commodity_Bucket)):
                for j in range(len(pos_gp.Group)):
                    if pos_gp.Group[j] == params.Commodity_Bucket[i]:
                        S[i] = pos_gp.S[j]
                        break

    elif risk_class == 'FX':
        S = 0

    return S

def delta_margin_group(gp, params):
    logger.info('Calculate Delta Margin for {0}'.format(gp.Qualifier.unique()))

    risk_class = gp.RiskClass.unique()[0]

    s = build_risk_factors(gp, params)
    RW = build_risk_weights(gp, params)
    CR = build_concentration_risk(gp, params)

    WS = RW * s * CR

    Corr = build_in_bucket_correlation(gp, params)

    K = np.mat(WS) * np.mat(Corr) * np.mat(np.reshape(WS, (len(WS), 1)))
    K = math.sqrt(K.item(0))

    ret = gp[['ProductClass', 'RiskType', 'RiskClass']].copy()
    ret.drop_duplicates(inplace=True)
    ret['S'] = max(min(WS.sum(), K), -K)

    if risk_class == 'IR':
        ret['CR'] = CR
    else:
        ret['CR'] = CR[0]

    ret['WeightDelta'] = K

    if risk_class == 'IR':
        ret['Group'] = gp['Qualifier'].unique()[0]
    elif risk_class == 'FX':
        ret['Group'] = gp['RiskType'].unique()[0]
    else:
        ret['Group'] = gp['Bucket'].unique()[0]

    return ret

def delta_margin_risk_factor(pos, params):
    """Calculate Delta Margin for IR Class"""

    pos_delta = net_sensitivities(pos, params)

    product_class = pos_delta.ProductClass.unique()[0]
    risk_class = pos_delta.RiskClass.unique()[0]
    risk_type = pos_delta.RiskType.unique()[0]

    if risk_class == 'IR':
        group = 'Qualifier'
    elif risk_class == 'FX':
        group = 'RiskType'
    else:
        group = 'Bucket'

    pos_delta = pos_delta.groupby([group]).apply(delta_margin_group, params)
    pos_delta.reset_index(inplace=True, drop=True)

    intermediate_path = '{0}\{1}\{2}'.format(os.getcwd(), product_class, risk_class)
    pos_delta.to_csv('{0}\{1}_delta_margin_group.csv'.format(intermediate_path, risk_type), index=False)

    g = build_bucket_correlation(pos_delta, params)

    pos_delta_non_residual = pos_delta[pos_delta.Group != 'Residual'].copy()
    pos_delta_residual = pos_delta[pos_delta.Group == 'Residual'].copy()

    S = build_non_residual_S(pos_delta_non_residual, params)
    SS = np.mat(S) * np.mat(g) * np.mat(np.reshape(S, (len(S), 1)))
    SS = SS.item(0)

    delta_margin = math.sqrt(np.dot(pos_delta_non_residual.WeightDelta, pos_delta_non_residual.WeightDelta) + SS)

    if len(pos_delta_residual) > 0:
        delta_margin = delta_margin + pos_delta_residual.WeightDelta

    ret_mm = pos_delta[['ProductClass', 'RiskClass']].copy()
    ret_mm['DeltaMargin'] = delta_margin

    return ret_mm

def delta_margin(pos, params):
    pos_delta = pos[pos.RiskType.isin(params.Delta_Factor)].copy()

    if len(pos_delta) > 0:
        pos_delta_margin = delta_margin_risk_factor(pos_delta, params)

    if len(pos_delta_margin):
        pos_delta_margin_gp = pos_delta_margin.groupby(['ProductClass', 'RiskClass'])
        pos_delta_margin_gp = pos_delta_margin_gp.agg({'DeltaMargin': np.sum})
        pos_delta_margin_gp.reset_index(inplace=True)

    return pos_delta_margin_gp

def calculate_simm(pos, params):
    pos_gp_delta_margin = pos.groupby(['ProductClass']).apply(delta_margin, params)
    pos_gp_delta_margin.reset_index(inplace=True, drop=True)

    return pos_gp_delta_margin










