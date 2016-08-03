import numpy as np
import pandas as pd
import os
import logging
import math
from margin_lib import Margin

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

class VegaMargin(Margin):

    def net_sensitivities(self, pos, params):
        risk_class = pos.RiskClass.unique()[0]

        if risk_class == ['IR', 'FX']:
            factor_group = ['ProductClass', 'RiskType', 'Qualifier', 'Label1', 'RiskClass']
        elif risk_class in ['CreditQ', 'CreditNonQ', 'Equity', 'Commodity']:
            factor_group = ['ProductClass', 'RiskType', 'Qualifier', 'Bucket', 'Label1', 'RiskClass']

        pos_gp = pos.groupby(factor_group)
        pos_vega = pos_gp.agg({'AmountUSD': np.sum})
        pos_vega.reset_index(inplace=True)

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

    def delta_margin_group(self, gp, params):
        logger.info('Calculate Delta Margin for {0}'.format(gp.Qualifier.unique()))

        risk_class = gp.RiskClass.unique()[0]

        s = self.build_risk_factors(gp, params)
        RW = self.build_risk_weights(gp, params)
        CR = self.build_concentration_risk(gp, params)

        WS = RW * s * CR

        Corr = self.build_in_bucket_correlation(gp, params)

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



