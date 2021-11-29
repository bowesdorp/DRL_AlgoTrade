# DRL_AlgoTrade
This repository contains code to train a Deep Reinforcement Learning (DRL) Agent to trade stocks. The trained model is connected to a stock broker platform to make real trades in real time, using real money. 
[FinRL](https://github.com/AI4Finance-Foundation/FinRL) was used at the core for the implementation. 

The Agent only trades a single time a day (at the end), with the goal of maximizing profits. The agent is encouraged to buy assets instead of holding cash by introducing a cash penalty. 


# TODO
1. Fetch current stock information (real time)
2. Connect with [De Giro API](https://pypi.org/project/degiroapi/)
3. Link model with De Giro to make trades
4. Set everything on a cloud server
