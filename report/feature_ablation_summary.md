# Feature Ablation Summary

| Experiment | Features | NMI | Peer Diff | Cov Annual Var |
| --- | --- | ---: | ---: | ---: |
| minilm text | text_business,text_risk | 0.381 | -0.272 |  |
| price + minilm text | price_volatility,price_momentum,price_liquidity,text_business,text_risk | 0.368 | -0.140 |  |
| price + minilm text + 8-K counts | price_volatility,price_momentum,price_liquidity,event_item_frequency,text_business,text_risk | 0.352 | -0.153 |  |
| price + hashed text | price_volatility,price_momentum,price_liquidity,text_business,text_risk | 0.341 | -0.108 |  |
| price | price_volatility,price_momentum,price_liquidity | 0.171 | -0.100 | 0.00994 |
| 8-K counts | event_item_frequency | 0.091 | -0.262 |  |
