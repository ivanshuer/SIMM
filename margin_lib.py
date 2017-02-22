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

    file_handler = logging.FileHandler('log.txt')
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
###############################

def calculate_CR(gp, params, margin):

    if gp['RiskClass'] == 'IR':
        if gp['Qualifier'] in params.IR_Low_Vol_Curr:
            if margin == 'Delta':
                Tb = params.IR_Delta_Low_Vol_Threshold
            elif margin in ['Vega', 'Curvature']:
                Tb = params.IR_Vega_Low_Vol_Threshold
        elif gp['Qualifier'] in params.IR_Reg_Vol_Less_Well_Traded_Curr:
            if margin == 'Delta':
                Tb = params.IR_Delta_Reg_Vol_Less_Well_Traded_Threshold
            elif margin in ['Vega', 'Curvature']:
                Tb = params.IR_Vega_Reg_Vol_Less_Well_Traded_Threshold
        elif gp['Qualifier'] in params.IR_Reg_Vol_Well_Traded_Curr:
            if margin == 'Delta':
                Tb = params.IR_Delta_Reg_Vol_Well_Traded_Threshold
            elif margin in ['Vega', 'Curvature']:
                Tb = params.IR_Vega_Reg_Vol_Well_Traded_Threshold
        else:
            if margin == 'Delta':
                Tb = params.IR_Delta_High_Vol_Threshold
            elif margin in ['Vega', 'Curvature']:
                Tb = params.IR_Vega_High_Vol_Threshold
    elif gp['RiskClass'] == 'FX':
        if margin == 'Delta':
            if gp['Qualifier'] in params.FX_Significantly_Material:
                Tb = params.FX_Significantly_Material_FX_Threshold
            elif gp['Qualifier'] in params.FX_Frequently_Traded:
                Tb = params.FX_Frequently_Traded_Threshold
            else:
                Tb = params.FX_Others_Threshold
        elif margin in ['Vega', 'Curvature']:
            curr1 = gp['Qualifier'][0:3]
            curr2 = gp['Qualifier'][3:6]

            if curr1 in params.FX_Significantly_Material and curr2 in params.FX_Significantly_Material:
                Tb = params.FX_Vega_C1_C1_Threshold
            elif (curr1 in params.FX_Significantly_Material and curr2 in params.FX_Frequently_Traded) or \
                    (curr1 in params.FX_Frequently_Traded and curr2 in params.FX_Significantly_Material):
                Tb = params.FX_Vega_C1_C2_Threshold
            elif curr1 in params.FX_Significantly_Material or curr2 in params.FX_Significantly_Material:
                Tb = params.FX_Vega_C1_C3_Threshold
            else:
                Tb = params.FX_Vega_Others_Threshold

    gp['CR'] = max(1, math.sqrt(abs(gp['AmountUSD']) / Tb))

    return gp[['Qualifier', 'CR']]

def build_concentration_risk(pos_gp, params, margin):
    risk_class = pos_gp.RiskClass.unique()[0]
    if risk_class not in ['IR', 'FX']:
        bucket = pos_gp.Bucket.unique()[0]

    #is_vega_factor = pos_gp.RiskType.unique()[0] in params.Vega_Factor
    #is_curvature_factor = pos_gp.RiskType.unique()[0] in params.Curvature_Factor

    #f = lambda x: max(1, math.sqrt(abs(x) / Tb))
    f = lambda x: 100
    if margin == 'Vega' or margin == 'Curvature':
        f = lambda x: 1

    if risk_class == 'IR':
        #Tb = params.IR_G10_DKK_Threshold

        #gp_curr = pos_gp.Qualifier.unique()[0]

        #if not gp_curr in params.G10_Curr:
        #    Tb = params.IR_Other_Threshold

        #CR = max(1, math.sqrt(abs(pos_gp.AmountUSD.sum() / Tb)))
        #CR = 100
        #if self.__margin == 'Vega' or self.__margin == 'Curvature':
        #    CR = 1
        pos_gp_CR = pos_gp.groupby(['Qualifier', 'Risk_Group']).agg({'AmountUSD': np.sum, 'CR_THR': np.average})
        pos_gp_CR.reset_index(inplace=True)

        pos_gp_CR['CR'] = pos_gp_CR.apply(lambda d: max(1, math.sqrt(abs(d['AmountUSD']) / d['CR_THR'])), axis=1)
        CR = pos_gp_CR['CR'].values

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

            if margin == 'Vega' or margin == 'Curvature':
                tenors = params.Equity_Tenor

            CR = pos_gp.AmountUSD.apply(f)
        elif risk_class == 'Commodity':
            if bucket in params.Commodity_FUEL:
                Tb = params.Commodity_FUEL_Threshold
            elif bucket in params.Commodity_POWER:
                Tb = params.Commodity_POWER_Threshold
            elif bucket in params.Commodity_OTHER:
                Tb = params.Commodity_OTHER_Threshold

            if margin == 'Vega' or margin == 'Curvature':
                tenors = params.Commodity_Tenor

            CR = pos_gp.AmountUSD.apply(f)
        elif risk_class == 'FX':
            pos_gp_CR = pos_gp.groupby(['Qualifier', 'Risk_Group']).agg({'AmountUSD': np.sum, 'CR_THR': np.average})
            pos_gp_CR.reset_index(inplace=True)
            pos_gp_CR['CR'] = pos_gp_CR.apply(lambda d: max(1, math.sqrt(abs(d['AmountUSD']) / d['CR_THR'])), axis=1)

            pos_gp_CR = pd.merge(pos_gp, pos_gp_CR[['Qualifier', 'Risk_Group', 'CR']], how='left')

            CR = pos_gp_CR['CR']

            if margin == 'Vega' or margin == 'Curvature':
                tenors = params.FX_Tenor

        CR = CR.values

        if margin == 'Vega' or margin == 'Curvature':
            CR = np.repeat(CR, len(tenors))

    return CR

def build_in_bucket_correlation(pos_gp, params, margin, CR):
    risk_class = pos_gp.RiskClass.unique()[0]
    if risk_class not in ['IR', 'FX']:
        bucket = pos_gp.Bucket.unique()[0]

    if risk_class == 'IR':
        gp_curr = pos_gp.Qualifier.unique()[0]
        curve = params.IR_Sub_Curve
        if gp_curr == 'USD':
            curve = params.IR_USD_Sub_Curve

        fai = np.zeros((len(curve), len(curve)))
        fai.fill(params.IR_Fai)
        np.fill_diagonal(fai, 1)

        if margin == 'Vega' or margin == 'Curvature':
            fai = 1

        rho = params.IR_Corr
        if margin == 'Curvature':
            rho = rho * rho

        Corr = np.kron(rho, fai)

        pos_inflation = pos_gp[pos_gp.RiskType == 'Risk_Inflation'].copy()
        if len(pos_inflation) > 0:
            inflation_rho = np.ones(len(curve)*len(params.IR_Tenor)) * params.IR_Inflation_Rho
            inflation_rho_column = np.reshape(inflation_rho, (len(inflation_rho), 1))
            Corr = np.append(Corr, inflation_rho_column, axis=1)

            inflation_rho = np.append(inflation_rho, 1)
            inflation_rho = np.reshape(inflation_rho, (1, len(inflation_rho)))
            Corr = np.append(Corr, inflation_rho, axis=0)
    else:
        num_qualifiers = pos_gp.Qualifier.nunique()

        F = np.zeros((len(CR), len(CR)))

        for i in range(len(CR)):
            for j in range(len(CR)):
                CRi = CR[i]
                CRj = CR[j]

                F[i][j] = min(CRi, CRj) / max(CRi, CRj)

        if risk_class in ['CreditQ', 'CreditNonQ']:
            if risk_class == 'CreditQ':
                tenors = params.CreditQ_Tenor
                same_is_rho = params.CreditQ_Rho_Agg_Same_IS
                diff_is_rho = params.CreditQ_Rho_Agg_Diff_IS
                if bucket == 'Residual':
                    same_is_rho = params.CreditQ_Rho_Res_Same_IS
                    diff_is_rho = params.CreditQ_Rho_Res_Diff_IS
            else:
                tenors = params.CreditNonQ_Tenor
                same_is_rho = params.CreditNonQ_Rho_Agg_Same_IS
                diff_is_rho = params.CreditNonQ_Rho_Agg_Diff_IS
                if bucket == 'Residual':
                    same_is_rho = params.CreditNonQ_Rho_Res_Same_IS
                    diff_is_rho = params.CreditNonQ_Rho_Res_Diff_IS

            rho = np.ones((num_qualifiers, num_qualifiers)) * diff_is_rho
            np.fill_diagonal(rho, same_is_rho)

            one_mat = np.ones((len(tenors), len(tenors)))
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

        if margin == 'Curvature':
            rho = rho * rho
            F.fill(1)

        Corr = rho * F
        np.fill_diagonal(Corr, 1)

    return Corr

def build_bucket_correlation(pos_delta, params, margin):
    risk_class = pos_delta.RiskClass.unique()[0]

    g = 0

    if risk_class == 'IR':
        all_curr = pos_delta.Group.unique()
        g = np.ones((len(all_curr), len(all_curr)))

        if not margin == 'Curvature':
            for i in range(len(all_curr)):
                for j in range(len(all_curr)):
                    CRi = pos_delta.iloc[[i]].CR.values[0]
                    CRj = pos_delta.iloc[[j]].CR.values[0]

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

    if margin == 'Curvature':
        g = pow(g, 2)

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