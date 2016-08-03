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

class InterestRate(object):
    """This is common API for interest risk class IM calculation"""

    def __init__(self):
        self.__risk_class = 'IR'

    def prep_data_IRCurve(self, pos, params):
        """Check data quality for IR Curve factor"""

        # Check Bucket
        pos_no_bucket = pos[~pos.Bucket.isin(params.IR_Bucket)].copy()
        if len(pos_no_bucket) > 0:
            logger.info('{0} IR Curve trades have wrong Bucket'.format(len(pos_no_bucket)))
            pos_no_bucket['reason'] = 'Bucket'

        pos = pos[pos.Bucket.isin(params.IR_Bucket)].copy()

        # Check Label1
        pos_no_label1 = pos[~pos.Label1.isin(params.IR_tenor)].copy()
        if len(pos_no_label1) > 0:
            logger.info('{0} IR Curve trades have wrong Label 1'.format(len(pos_no_label1)))
            pos_no_label1['reason'] = 'Label1'

        pos = pos[pos.Label1.isin(params.IR_tenor)].copy()

        # Check Label2
        pos_no_label2 = pos[~(((pos.Qualifier == 'USD') & pos.Label2.isin(params.IR_USD_sub_curve)) |
                            ((pos.Qualifier != 'USD') & pos.Label2.isin(params.IR_sub_curve)))].copy()

        if len(pos_no_label2) > 0:
            logger.info('{0} IR Curve trades have wrong Label 2'.format(len(pos_no_label2)))
            pos_no_label2['reason'] = 'Label2'

        pos = pos[((pos.Qualifier == 'USD') & pos.Label2.isin(params.IR_USD_sub_curve)) |
                  ((pos.Qualifier != 'USD') & pos.Label2.isin(params.IR_sub_curve))].copy()

        pos = pd.concat([pos, pos_no_bucket, pos_no_label1, pos_no_label2])

        return pos

    def prep_data_IRVol(self, pos, params):
        """Check data quality for IR Vol factor"""

        # Check Label1
        pos_no_label1 = pos[~pos.Label1.isin(params.IR_tenor)].copy()
        if len(pos_no_label1) > 0:
            logger.info('{0} IR Vol trades have wrong Label 1'.format(len(pos_no_label1)))
            pos_no_label1['reason'] = 'Label1'

        pos = pos[pos.Label1.isin(params.IR_tenor)].copy()

        pos = pd.concat([pos, pos_no_label1])

        return pos

    def prep_data(self, pos, params):

        pos_IRCurve = pos[pos.RiskType == 'Risk_IRCurve'].copy()
        pos_IRCurve = self.prep_data_IRCurve(pos_IRCurve, params)

        pos_IRVol = pos[pos.RiskType == 'Risk_IRVol'].copy()
        pos_IRVol = self.prep_data_IRVol(pos_IRVol, params)

        pos_Inflation = pos[pos.RiskType == 'Risk_Inflation'].copy()

        pos = pd.concat([pos_IRCurve, pos_IRVol, pos_Inflation])

        return pos

    def delta_margin(self):
        """Calculate Delta Margin for IR class"""

        return
