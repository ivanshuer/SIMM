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

class Margin(object):

    def build_concentration_risk(self, pos_gp, params):
        risk_class = pos_gp.RiskClass.unique()[0]
        if risk_class not in ['IR', 'FX']:
            bucket = pos_gp.Bucket.unique()[0]

        is_vega_factor = pos_gp.RiskType.unique()[0] in params.Vega_Factor

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

                if is_vega_factor:
                    tenors = params.Equity_Tenor
            elif risk_class == 'Commodity':
                if bucket in params.Commodity_FUEL:
                    Tb = params.Commodity_FUEL_Threshold
                elif bucket in params.Commodity_POWER:
                    Tb = params.Commodity_POWER_Threshold
                elif bucket in params.Commodity_OTHER:
                    Tb = params.Commodity_OTHER_Threshold

                if is_vega_factor:
                    tenors = params.Commodity_Tenor
            elif risk_class == 'FX':
                Tb = params.FX_Threshold

                if is_vega_factor:
                    tenors = params.FX_Tenor

            CR = pos_gp.AmountUSD.apply(f)
            CR = CR.values

            if is_vega_factor:
                CR = np.repeat(CR, len(tenors))

        return CR

    def build_in_bucket_correlation(self, pos_gp, params):
        risk_class = pos_gp.RiskClass.unique()[0]
        if risk_class not in ['IR', 'FX']:
            bucket = pos_gp.Bucket.unique()[0]

        is_vega_factor = pos_gp.RiskType.unique()[0] in params.Vega_Factor

        if risk_class == 'IR':
            gp_curr = pos_gp.Qualifier.unique()[0]
            curve = params.IR_Sub_Curve
            if gp_curr == 'USD':
                curve = params.IR_USD_Sub_Curve

            fai = np.zeros((len(curve), len(curve)))
            fai.fill(params.IR_Fai)
            np.fill_diagonal(fai, 1)

            if is_vega_factor:
                fai = 1

            rho = params.IR_Corr

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

            CR = self.build_concentration_risk(pos_gp, params)

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

            Corr = rho * F
            np.fill_diagonal(Corr, 1)

        return Corr

    def build_bucket_correlation(self, pos_delta, params):
        risk_class = pos_delta.RiskClass.unique()[0]

        g = 0

        if risk_class == 'IR':
            all_curr = pos_delta.Group.unique()
            g = np.zeros((len(all_curr), len(all_curr)))
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

        g = np.mat(g)
        np.fill_diagonal(g, 0)

        return g

    def build_non_residual_S(self, pos_gp, params):
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

    def margin_risk_factor(self, pos, params):
        """Calculate Delta Margin for IR Class"""

        pos_delta = self.net_sensitivities(pos, params)

        product_class = pos_delta.ProductClass.unique()[0]
        risk_class = pos_delta.RiskClass.unique()[0]
        risk_type = pos_delta.RiskType.unique()[0]

        if risk_class == 'IR':
            group = 'Qualifier'
        elif risk_class == 'FX':
            group = 'RiskType'
        else:
            group = 'Bucket'

        #pos_delta_gp_all = []
        #for gp in pos_delta[group].unique():
        #    pos_delta_gp = pos_delta[pos_delta[group] == gp].copy()
        #    pos_delta_gp = self.margin_risk_group(pos_delta_gp, params)
        #    pos_delta_gp_all.append(pos_delta_gp)

        #pos_delta_gp_all = pd.concat(pos_delta_gp_all)

        #pos_delta = pos_delta_gp_all.copy()

        pos_delta = pos_delta.groupby([group]).apply(self.margin_risk_group, params)
        pos_delta.reset_index(inplace=True, drop=True)

        intermediate_path = '{0}\{1}\{2}'.format(os.getcwd(), product_class, risk_class)
        pos_delta.to_csv('{0}\{1}_margin_group.csv'.format(intermediate_path, risk_type), index=False)

        g = self.build_bucket_correlation(pos_delta, params)

        pos_delta_non_residual = pos_delta[pos_delta.Group != 'Residual'].copy()
        pos_delta_residual = pos_delta[pos_delta.Group == 'Residual'].copy()

        S = self.build_non_residual_S(pos_delta_non_residual, params)

        if risk_class != 'FX':
            SS = np.mat(S) * np.mat(g) * np.mat(np.reshape(S, (len(S), 1)))
            SS = SS.item(0)
        else:
            SS = 0

        delta_margin = math.sqrt(np.dot(pos_delta_non_residual.K, pos_delta_non_residual.K) + SS)

        if len(pos_delta_residual) > 0:
            delta_margin = delta_margin + pos_delta_residual.K

        ret_mm = pos_delta[['ProductClass', 'RiskClass']].copy()
        ret_mm['Margin'] = delta_margin

        return ret_mm