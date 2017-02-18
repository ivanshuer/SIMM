import numpy as np
import pandas as pd
import os
import logging
import math
from margin_lib import Margin
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

class VegaMargin(Margin):

    def __init__(self):
        Margin.__init__(self, 'Vega')

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

        #should separate FX and Comdty/Equity, the latter two uses different weight per bucket
        if risk_class == 'FX':
            pos_vega['AmountUSD'] = pos_vega['AmountUSD'] * params.FX_Weights * math.sqrt(365.0 / 14) / norm.ppf(0.99)

        elif risk_class in ['Equity', 'Commodity']:
            weights = params.Commdty_Weights
            bucket = pd.DataFrame(pos_vega.Bucket.as_matrix(), columns =['bucket'])
            RW = pd.merge(bucket, weights, lef_on=['bucket'], right_on=['bucket'], how='inner')
            pos_vega['AmountUSD'] = pos_vega['AmountUSD'] * RW.weight * math.sqrt(365.0 / 14) / norm.ppf(0.99)

        #for Equity, FX and Comdty vega should sum over tenor j (Label 1) to get VR input
        if risk_class in ['Equity', 'Commodity']:
            pos_vega_gp = pos_vega.groupby['ProductClass', 'RiskType', 'Qualifier', 'Bucket', 'RiskClass'].agg({'AmountUSD': np.sum})
            pos_vega.gp.reset_index(inplace=True)
        elif risk_class == 'FX':
            pos_vega_gp = pos_vega.groupby['ProductClass', 'RiskType', 'Qualifier', 'RiskClass'].agg({'AmountUSD': np.sum})
            pos_vega.gp.reset_index(inplace=True)
        else:
            pos_vega_gp = pos_vega.copy()

        return pos_vega_gp

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
            #Equity, FX and Comdty should not have tenors in risk factors, removing for Credit as well but should modify later
            #if risk_class == 'CreditQ':
            #    tenors = params.CreditQ_Tenor
            # elif risk_class == 'Equity':
            #     tenors = params.Equity_Tenor
            # elif risk_class == 'Commodity':
            #     tenors = params.Commodity_Tenor
            # elif risk_class == 'FX':
            #     tenors = params.FX_Tenor

            # s = np.zeros(pos_gp.Qualifier.nunique() * len(tenors))
            #
            # for j in range(pos_gp.Qualifier.nunique()):
            #     pos_gp_qualifier = pos_gp[pos_gp.Qualifier == pos_gp.sort_values(['Qualifier']).Qualifier.unique()[j]].copy()
            #
            #     for i, row in pos_gp_qualifier.iterrows():
            #         idx = self.find_factor_idx(row['Label1'], tenors)
            #         if idx >= 0:
            #             s[idx + j * len(tenors)] = row['AmountUSD']
            s = np.zeros(pos_gp.Qualifier.nunique())
            idx = 0
            for j in range(pos_gp.Qualifier.nunique()):
                pos_gp_qualifier = pos_gp[pos_gp.Qualifier == pos_gp.sort_values(['Qualifier']).Qualifier.unique()[j]].copy()

                for i, row in pos_gp_qualifier.iterrows():
                    if idx >= 0:
                        s[idx] = row['AmountUSD']
                        idx= idx + 1
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

        Corr = self.build_in_bucket_correlation(gp, params)

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



