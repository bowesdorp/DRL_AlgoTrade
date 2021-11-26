import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import datetime
from finrl.apps import config

from DataLoader import DataLoader
from Trainer import Trainer

matplotlib.use('Agg')


def create_folders():
    if not os.path.exists("./" + config.DATA_SAVE_DIR):
        os.makedirs("./" + config.DATA_SAVE_DIR)
    if not os.path.exists("./" + config.TRAINED_MODEL_DIR):
        os.makedirs("./" + config.TRAINED_MODEL_DIR)
    if not os.path.exists("./" + config.TENSORBOARD_LOG_DIR):
        os.makedirs("./" + config.TENSORBOARD_LOG_DIR)
    if not os.path.exists("./" + config.RESULTS_DIR):
        os.makedirs("./" + config.RESULTS_DIR)

if __name__ == "__main__":
    print("Started... the script...")
    create_folders()

    # create DataLoader and get train and trade data
    dl = DataLoader(ticker_list=['AAPL'])
    dl.preprocess()

    df = dl.get_data()
    print(df)

    train_dates = ('2009-01-01','2016-01-01')
    trade_dates = ('2016-01-01','2021-01-01')

    dl.split_dataset(train_dates, trade_dates)

    train_data = dl.get_train_data()
    trade_data = dl.get_trade_data()

    print(train_data.shape, trade_data.shape)
    
    

    # create the environment for the agent
    

    model = Trainer(train_data, trade_data)

    model.set_environment('train')
    model.set_environment('trade')
    model.set_agent()

    model.train()

    print('saving model')

    model.save_model()

    # model.backtest()




    
    print("Done")
    