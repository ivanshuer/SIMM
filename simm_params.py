Product = ['RatesFX', 'Credit', 'Equity', 'Commodity']

Reg_vol_curr = ['USD', 'EUR', 'GBP', 'CHF', 'AUD', 'NZD', 'CAD', 'SEK', 'NOK', 'DKK', 'HKD', 'KRW', 'SGD', 'TWD']
Low_vol_curr = ['JPY']

RiskType = ['IR', 'CreditQ', 'CreditNonQ', 'Equity', 'FX', 'Commodity']

IR = ['Risk_IRCurve', 'Risk_IRVol', 'Risk_Inflation']
CreditQ = ['Risk_CreditQ', 'Risk_CreditVol']
CreditNonQ = ['Risk_CreditNonQ', 'Risk_CreditVolNonQ']
Equity = ['Risk_Equity', 'Risk_EquityVol']
FX = ['Risk_FX', 'Risk_FXVol']
Commodity = ['Risk_Commodity', 'Risk_CommodityVol']

IR_tenor = ['2w', '1m', '3m', '6m', '1y', '2y', '3y', '5y', '10y', '15y', '20y', '30y']
IR_sub_curve = ['OIS', 'Libor1m', 'Libor3m', 'Libor6m', 'Libor12m']
IR_USD_sub_curve = IR_sub_curve + ['Prime']

