# Strategy-Backtest-SVM-based-Orderbook-HFT-trading-strategy

Strategy Overview

The core idea is to use machine learning to model the dynamics of the limit order book and predict short-term price movements 
. The strategy uses a set of 17 features derived from Level-1 order book data to train an SVM classifier 
. This classifier predicts whether the market mid-price will move "up," "down," or remain "flat" in the near future (e.g., over the next few ticks) 
. Trades are triggered on strong "up" or "down" signals
