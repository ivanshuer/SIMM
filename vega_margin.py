import numpy as np
import pandas as pd
import os
import logging
import math
import margin_lib as mlib
from scipy.stats import norm

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

class VegaMargin(object):

    def __init__(self):
        self.__margin = 'Vega'

    def change_FX_ticker_order(self, gp):
        curr1 = gp['Qualifier'][0:3]
        curr2 = gp['Qualifier'][3:6]

        curr_pair = set([curr1, curr2])
        curr_pair = "".join(curr_pair)

        gp['Qualifier'] = curr_pair

        return gp

    def net_sensitivities(self, pos, params):
        risk_class = pos.RiskClass.unique()[0]

        if risk_class == 'IR':
            factor_group = ['ProductClass', 'RiskType', 'Qualifier', 'Label1', 'RiskClass']
        elif risk_class == 'FX':
            pos = pos.apply(self.change_FX_ticker_order, axis=1)
            factor_group = ['ProductClass', 'RiskType', 'Qualifier', 'Label1', 'RiskClass']
        elif risk_class in ['CreditQ', 'CreditNonQ', 'Equity', 'Commodity']:
            factor_group = ['ProductClass', 'RiskType', 'Qualifier', 'Bucket', 'Label1', 'RiskClass']

        pos_gp = pos.groupby(factor_group)
        pos_vega = pos_gp.agg({'AmountUSD': np.sum})
        pos_vega.reset_index(inplace=True)

        if risk_class in ['Euqity', 'FX', 'Commodity']:
            pos_vega['AmountUSD'] = pos_vega['AmountUSD'] * params.FX_Weights * math.sqrt(365.0 / 14) / norm.ppf(0.99)

        return pos_vega

    def find_factor_idx(self, tenor_factor, tenors):
        idx = 0

        for tenor in tenors:
            if tenor_factor == tenor:
                return idx
            else:
                idx = idx + 1

        return -1

    def build_risk_factors(self, pos_gp, params):

        risk_class = pos_gp.RiskClass.unique()[0]

        if risk_class == 'IR':
            s = np.zeros(len(params.IR_Tenor))

            for i, row in pos_gp.iterrows():
                idx = self.find_factor_idx(row['Label1'], params.IR_Tenor)
                if idx >= 0:
                    s[idx] = row['AmountUSD']
        else:
            if risk_class == 'CreditQ':
                tenors = params.CreditQ_Tenor
            elif risk_class == 'Equity':
                tenors = params.Equity_Tenor
            elif risk_class == 'Commodity':
                tenors = params.Commodity_Tenor
            elif risk_class == 'FX':
                tenors = params.FX_Tenor

            s = np.zeros(pos_gp.Qualifier.nunique() * len(tenors))

            for j in range(pos_gp.Qualifier.nunique()):
                pos_gp_qualifier = pos_gp[pos_gp.Qualifier == pos_gp.sort_values(['Qualifier']).Qualifier.unique()[j]].copy()

                for i, row in pos_gp_qualifier.iterrows():
                    idx = self.find_factor_idx(row['Label1'], tenors)
                    if idx >= 0:
                        s[idx + j * len(tenors)] = row['AmountUSD']
        return s

    def build_risk_weights(self, pos_gp, params):
        risk_class = pos_gp.RiskClass.unique()[0]

        if risk_class == 'IR':
            VRW = params.IR_VRW
        elif risk_class == 'CreditQ':
            VRW = params.CreditQ_VRW
        elif risk_class == 'CreditNonQ':
            VRW = params.CreditNonQ_VRW
        elif risk_class == 'Equity':
            VRW = params.Equity_VRW
        elif risk_class == 'Commodity':
            VRW = params.Commodity_VRW
        elif risk_class == 'FX':
            VRW = params.FX_VRW

        return VRW

    def calculate_CR_Threshold(self, gp, params):

        if gp['RiskClass'] == 'IR':
            if gp['Qualifier'] in params.IR_Low_Vol_Curr:
                Thrd = params.IR_CR_Vega_Low_Vol
            elif gp['Qualifier'] in params.IR_Reg_Vol_Less_Well_Traded_Curr:
                Thrd = params.IR_CR_Vega_Reg_Vol_Less_Well_Traded
            elif gp['Qualifier'] in params.IR_Reg_Vol_Well_Traded_Curr:
                Thrd = params.IR_CR_Vega_Reg_Vol_Well_Traded
            else:
                Thrd = params.IR_CR_Vega_High_Vol

        elif gp['RiskClass'] == 'FX':
            curr1 = gp['Qualifier'][0:3]
            curr2 = gp['Qualifier'][3:6]

            if curr1 in params.FX_Significantly_Material and curr2 in params.FX_Significantly_Material:
                Thrd = params.FX_CR_Vega_C1_C1
            elif (curr1 in params.FX_Significantly_Material and curr2 in params.FX_Frequently_Traded) or \
                    (curr1 in params.FX_Frequently_Traded and curr2 in params.FX_Significantly_Material):
                Thrd = params.FX_CR_Vega_C1_C2
            elif curr1 in params.FX_Significantly_Material or curr2 in params.FX_Significantly_Material:
                Thrd = params.FX_CR_Vega_C1_C3
            else:
                Thrd = params.FX_CR_Vega_Others

        gp['Thrd'] = Thrd

        return gp

    def margin_risk_group(self, gp, params):

        risk_class = gp.RiskClass.unique()[0]

        if risk_class in ['IR', 'FX']:
            logger.info('Calculate {0} Vega Margin for {1}'.format(risk_class, gp.Qualifier.unique()))
        else:
            logger.info('Calculate {0} Vega Margin for {1}'.format(risk_class, gp.Bucket.unique()))

        s = self.build_risk_factors(gp, params)
        RW = self.build_risk_weights(gp, params)
        CR = self.build_concentration_risk(gp, params)

        WS = RW * s * CR

        Corr = mlib.build_in_bucket_correlation(gp, params, self.__margin, CR)

        K = np.mat(WS) * np.mat(Corr) * np.mat(np.reshape(WS, (len(WS), 1)))
        K = math.sqrt(K.item(0))

        ret = gp[['ProductClass', 'RiskType', 'RiskClass']].copy()
        ret.drop_duplicates(inplace=True)
        ret['K'] = K
        ret['S'] = max(min(WS.sum(), K), -K)

        if risk_class == 'IR':
            ret['CR'] = CR
        else:
            ret['CR'] = CR[0]

        if risk_class == 'IR':
            ret['Group'] = gp['Qualifier'].unique()[0]
        elif risk_class == 'FX':
            ret['Group'] = gp['RiskType'].unique()[0]
        else:
            ret['Group'] = gp['Bucket'].unique()[0]

        return ret