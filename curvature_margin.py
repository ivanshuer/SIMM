import numpy as np
import pandas as pd
import os
import logging
import math
from margin_lib import Margin
from vega_margin import VegaMargin

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

class CurvatureMargin(Margin):

    def __init__(self):
        self.__margin = 'Curvature'
        self.__vega_loader = VegaMargin()

    def net_sensitivities(self, pos, params):
        return self.__vega_loader.net_sensitivities(pos, params)

    def build_risk_factors(self, pos_gp, params):
        return self.__vega_loader.build_risk_factors(pos_gp, params)

    def margin_risk_group(self, gp, params):

        risk_class = gp.RiskClass.unique()[0]

        if risk_class in ['IR', 'FX']:
            logger.info('Calculate {0} Curvature Margin for {1}'.format(risk_class, gp.Qualifier.unique()))
        else:
            logger.info('Calculate {0} Curvature Margin for {1}'.format(risk_class, gp.Bucket.unique()))

        s = self.build_risk_factors(gp, params)

        WS = s

        Corr = self.build_in_bucket_correlation(gp, params)

        K = np.mat(WS) * np.mat(Corr) * np.mat(np.reshape(WS, (len(WS), 1)))
        K = math.sqrt(K.item(0))

        ret = gp[['ProductClass', 'RiskType', 'RiskClass']].copy()
        ret.drop_duplicates(inplace=True)
        ret['K'] = K
        ret['S'] = max(min(WS.sum(), K), -K)

        ret['CVR_sum'] = WS.sum()
        ret['CVR_abs_sum'] = abs(WS).sum()

        if risk_class == 'IR':
            ret['Group'] = gp['Qualifier'].unique()[0]
        elif risk_class == 'FX':
            ret['Group'] = gp['RiskType'].unique()[0]
        else:
            ret['Group'] = gp['Bucket'].unique()[0]

        return ret