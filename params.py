import pandas as pd

config_folder = 'config'

Product = ['RatesFX', 'Credit', 'Equity', 'Commodity']

G10_Curr = ['USD', 'EUR', 'GBP', 'JPY', 'CAD', 'AUD', 'NZD', 'CHF', 'DKK', 'NOK', 'SEK']
Reg_Vol_Curr = ['USD', 'EUR', 'GBP', 'CHF', 'AUD', 'NZD', 'CAD', 'SEK', 'NOK', 'DKK', 'HKD', 'KRW', 'SGD', 'TWD']
Low_Vol_Curr = ['JPY']

RiskType = ['IR', 'CreditQ', 'CreditNonQ', 'Equity', 'FX', 'Commodity']

IR = ['Risk_IRCurve', 'Risk_IRVol', 'Risk_Inflation']
CreditQ = ['Risk_CreditQ', 'Risk_CreditVol']
CreditNonQ = ['Risk_CreditNonQ', 'Risk_CreditVolNonQ']
Equity = ['Risk_Equity', 'Risk_EquityVol']
FX = ['Risk_FX', 'Risk_FXVol']
Commodity = ['Risk_Commodity', 'Risk_CommodityVol']

IR_Bucket = ['1', '2', '3']
IR_Tenor = ['2w', '1m', '3m', '6m', '1y', '2y', '3y', '5y', '10y', '15y', '20y', '30y']
IR_Sub_Curve = ['OIS', 'Libor1m', 'Libor3m', 'Libor6m', 'Libor12m']
IR_USD_Sub_Curve = IR_Sub_Curve + ['Prime']
IR_G10_DKK_Threshold = 1.0
IR_Other_Threshold = 1.0

IR_Weights = pd.read_csv('{0}/ir_weights_params.csv'.format(config_folder), dtype={'curr': str})
IR_Corr = pd.read_csv('{0}/ir_correlation_params.csv'.format(config_folder))
IR_Fai = 0.982
IR_Gamma = 0.27
IR_Inflation_Weights = 0.0032
IR_Inflation_Rho = 0.33

CreditQ_Bucket = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', 'Residual']
CreditQ_Tenor = ['1y', '2y', '3y', '5y', '10y']
CreditQ_Threshold = 1.0
CreditQ_Weights = pd.read_csv('{0}/creditq_weights_params.csv'.format(config_folder))

CreditNonQ_Bucket = ['1', '2', 'Residual']
CreditNonQ_Tenor = ['1y', '2y', '3y', '5y', '10y']
CreditNonQ_Threshold = 1.0
CreditNonQ_Weights = pd.read_csv('{0}/creditnonq_weights_params.csv'.format(config_folder))

Equity_Bucket = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', 'Residual']
Equity_EM = ['1', '2', '3', '4', '9']
Equity_DEVELOPED = ['5', '6', '7', '8', '10']
Equity_INDEX = ['11']
Equity_EM_Threshold = 1.0
Equity_DEVELOPED_Threshold = 1.0
Equity_INDEX_Threshold = 1.0
Equity_Weights = pd.read_csv('{0}/equity_weights_params.csv'.format(config_folder))

Commodity_Bucket = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16']
Commodity_FUEL = ['1', '2', '3', '4', '5', '6', '7']
Commodity_POWER = ['8', '9', '10']
Commodity_OTHER = ['11', '12', '13', '14', '15', '16']
Commodity_FUEL_Threshold = 1.0
Commodity_POWER_Threshold = 1.0
Commodity_OTHER_Threshold = 1.0
Commodity_Weights = pd.read_csv('{0}/commodity_weights_params.csv'.format(config_folder), dtype={'bucket': str})

FX_Threshold = 1.0
FX_Weights = 0.079



