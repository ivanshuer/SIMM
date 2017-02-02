import pandas as pd

config_folder = 'config'

Product = ['RatesFX', 'Credit', 'Equity', 'Commodity']

#G10_Curr = ['USD', 'EUR', 'GBP', 'JPY', 'CAD', 'AUD', 'NZD', 'CHF', 'DKK', 'NOK', 'SEK']
#Reg_Vol_Curr = ['USD', 'EUR', 'GBP', 'CHF', 'AUD', 'NZD', 'CAD', 'SEK', 'NOK', 'DKK', 'HKD', 'KRW', 'SGD', 'TWD']
Reg_Vol_Well_Traded_Curr = ['USD', 'EUR', 'GBP']
Reg_Vol_Less_Well_Traded_Curr = ['CHF', 'AUD', 'NZD', 'CAD', 'SEK', 'NOK', 'DKK', 'HKD', 'KRW', 'SGD', 'TWD']
Low_Vol_Curr = ['JPY']

RiskType = ['IR', 'CreditQ', 'CreditNonQ', 'Equity', 'Commodity', 'FX']

IR = ['Risk_IRCurve', 'Risk_IRVol', 'Risk_Inflation']
CreditQ = ['Risk_CreditQ', 'Risk_CreditVol']
CreditNonQ = ['Risk_CreditNonQ', 'Risk_CreditNonQVol']
Equity = ['Risk_Equity', 'Risk_EquityVol']
FX = ['Risk_FX', 'Risk_FXVol']
Commodity = ['Risk_Commodity', 'Risk_CommodityVol']

Delta_Factor = ['Risk_IRCurve', 'Risk_Inflation', 'Risk_CreditQ', 'Risk_CreditNonQ', 'Risk_Equity', 'Risk_FX', 'Risk_Commodity']
Vega_Factor = ['Risk_IRVol', 'Risk_CreditVol', 'Risk_EquityVol', 'Risk_FXVol', 'Risk_CommodityVol']
Curvature_Factor = Vega_Factor
#Curvature_Factor = ['Risk_IRCV', 'Risk_CreditCV', 'Risk_EquityCV', 'Risk_FXCV', 'Risk_CommodityCV']

Risk_Class_Corr = pd.read_csv('{0}/risk_class_correlation_params.csv'.format(config_folder))

IR_Bucket = ['1', '2', '3']
IR_Tenor = ['2w', '1m', '3m', '6m', '1y', '2y', '3y', '5y', '10y', '15y', '20y', '30y']
IR_Sub_Curve = ['OIS', 'Libor1m', 'Libor3m', 'Libor6m', 'Libor12m']
IR_USD_Sub_Curve = IR_Sub_Curve + ['Prime']
#IR_G10_DKK_Threshold = 1.0
#IR_Other_Threshold = 1.0
IR_Delta_High_Vol_Threshold = 7.4
IR_Delta_Reg_Vol_Well_Traded_Threshold = 250
IR_Delta_Reg_Vol_Less_Well_Traded_Threshold = 25
IR_Delta_Low_Vol_Threshold = 17
IR_Vega_High_Vol_Threshold = 120
IR_Vega_Reg_Vol_Well_Traded_Threshold = 3070
IR_Vega_Reg_Vol_Less_Well_Traded_Threshold = 160
IR_Vega_Low_Vol_Threshold = 960
IR_Weights = pd.read_csv('{0}/ir_weights_params.csv'.format(config_folder), dtype={'curr': str})
IR_Corr = pd.read_csv('{0}/ir_correlation_params.csv'.format(config_folder))
IR_Fai = 0.982
IR_Gamma = 0.27
IR_Inflation_Weights = 0.32
IR_Inflation_Rho = 0.33
IR_VRW = 0.21
IR_Curvature_Margin_Scale = 2.3

CreditQ_Bucket = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', 'Residual']
CreditQ_Tenor = ['1y', '2y', '3y', '5y', '10y']
CreditQ_Threshold = 1.0
CreditQ_Weights = pd.read_csv('{0}/creditq_weights_params.csv'.format(config_folder))
CreditQ_Rho_Agg_Same_IS = 0.98
CreditQ_Rho_Agg_Diff_IS = 0.55
CreditQ_Rho_Res_Same_IS = 0.5
CreditQ_Rho_Res_Diff_IS = 0.5
CreditQ_Corr = pd.read_csv('{0}/creditq_correlation_params.csv'.format(config_folder))
CreditQ_VRW = 0.35

CreditNonQ_Bucket = ['1', '2', 'Residual']
CreditNonQ_Tenor = ['1y', '2y', '3y', '5y', '10y']
CreditNonQ_Threshold = 1.0
CreditNonQ_Weights = pd.read_csv('{0}/creditnonq_weights_params.csv'.format(config_folder))
CreditNonQ_Rho_Agg_Same_IS = 0.6
CreditNonQ_Rho_Agg_Diff_IS = 0.21
CreditNonQ_Rho_Res_Same_IS = 0.5
CreditNonQ_Rho_Res_Diff_IS = 0.5
CreditNonQ_Corr = pd.read_csv('{0}/creditnonq_correlation_params.csv'.format(config_folder))
CreditNonQ_VRW = 0.35

Equity_Tenor = ['2w', '1m', '3m', '6m', '1y', '2y', '3y', '5y', '10y', '15y', '20y', '30y']
Equity_Bucket = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', 'Residual']
Equity_EM = ['1', '2', '3', '4', '9']
Equity_DEVELOPED = ['5', '6', '7', '8', '10']
Equity_INDEX = ['11']
Equity_EM_Threshold = 1.0
Equity_DEVELOPED_Threshold = 1.0
Equity_INDEX_Threshold = 1.0
Equity_Weights = pd.read_csv('{0}/equity_weights_params.csv'.format(config_folder))
Equity_Rho = pd.read_csv('{0}/equity_in_bucket_correlation_params.csv'.format(config_folder))
Equity_Corr = pd.read_csv('{0}/equity_correlation_params.csv'.format(config_folder))
Equity_VRW = 0.21

Commodity_Tenor = ['2w', '1m', '3m', '6m', '1y', '2y', '3y', '5y', '10y', '15y', '20y', '30y']
Commodity_Bucket = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16']
Commodity_FUEL = ['1', '2', '3', '4', '5', '6', '7']
Commodity_POWER = ['8', '9', '10']
Commodity_OTHER = ['11', '12', '13', '14', '15', '16']
Commodity_FUEL_Threshold = 1.0
Commodity_POWER_Threshold = 1.0
Commodity_OTHER_Threshold = 1.0
Commodity_Weights = pd.read_csv('{0}/commodity_weights_params.csv'.format(config_folder), dtype={'bucket': str})
Commodity_Rho = pd.read_csv('{0}/commodity_in_bucket_correlation_params.csv'.format(config_folder), dtype={'bucket': str})
Commodity_Corr = pd.read_csv('{0}/commodity_correlation_params.csv'.format(config_folder))
Commodity_VRW = 0.36

FX_Tenor = ['2w', '1m', '3m', '6m', '1y', '2y', '3y', '5y', '10y', '15y', '20y', '30y']
#FX_Threshold = 1.0
FX_Weights = 0.079
FX_Rho = 0.5
FX_VRW = 0.21
FX_Significantly_Material = ['USD', 'EUR', 'JPY', 'GBP', 'AUD', 'CHF', 'CAD']
FX_Frequently_Traded = ['BRL', 'CNY', 'HKD', 'INR', 'KRW', 'MXN', 'NOK', 'NZD', 'RUB', 'SEK', 'SGD', 'TRY', 'ZAR']
FX_Significantly_Material_FX_Threshold = 5200
FX_Frequently_Traded_Threshold = 1300
FX_Others_Threshold = 260
FX_Vega_C1_C1_Threshold = 5500
FX_Vega_C1_C2_Threshold = 3020
FX_Vega_C1_C3_Threshold = 520
FX_Vega_Others_Threshold = 87

