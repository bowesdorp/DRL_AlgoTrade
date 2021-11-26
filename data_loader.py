import numpy as np
import pandas as pd
from finrl.apps import config
from finrl.neo_finrl.preprocessor.yahoodownloader import YahooDownloader
from finrl.neo_finrl.preprocessor.preprocessors import FeatureEngineer, data_split


class DataLoader:
    """
        Class for loading in the stock data from Yahoo using the Yahoo downlader
    """

    def __init__(self, ticker_list=config.DOW_30_TICKER, start_date=config.START_DATE, 
                    end_date=config.END_DATE) -> None:
        
        self.ticker_list = ticker_list
        self.start_date = start_date
        self.end_date = end_date

        print("Fetching data....")
        print(start_date, end_date, ticker_list)

        self.data = YahooDownloader(self.start_date, self.end_date, self.ticker_list).fetch_data()

    def preprocess(self) -> None:
        """
            Preprocess the historical stock data, add statistical indicators
        """
    
        print("Starting preprocessing...")
        fe = FeatureEngineer(
                    use_technical_indicator=True,
                    tech_indicator_list = config.TECHNICAL_INDICATORS_LIST,
                    use_turbulence=False,
                    user_defined_feature = False)

        processed = fe.preprocess_data(self.data.copy())
        processed['log_volume'] = np.log(processed.volume*processed.close)
        processed['change'] = (processed.close-processed.open)/processed.close
        processed['daily_variance'] = (processed.high-processed.low)/processed.close

        self.data = processed.reset_index(drop=True)


    def split_dataset(self, train_interval, trade_interval) -> None:
        """
            Split the dataset into train and trade data
        """
        print("Splitting dataset into train and trade")
        self.train_data = data_split(self.data, train_interval[0], train_interval[1])
        self.trade_data = data_split(self.data, trade_interval[0], trade_interval[1])


    def get_data(self) -> pd.DataFrame:
        """
            Get all historical data
        """
        return self.data

    def get_train_data(self) -> pd.DataFrame:
        """
            Get all historical train data
        """
        return self.train_data

    def get_trade_data(self) -> pd.DataFrame:
        """
            Get all historical trade data
        """
        return self.trade_data
    



        

