from stable_baselines3 import PPO
from finrl.drl_agents.stablebaselines3.models import DRLAgent
from finrl.neo_finrl.env_stock_trading.env_stocktrading_cashpenalty import StockTradingEnvCashpenalty
from finrl.neo_finrl.preprocessor.yahoodownloader import YahooDownloader
from datetime import date
from datetime import timedelta
from finrl.neo_finrl.preprocessor.preprocessors import FeatureEngineer
from finrl.apps import config

import numpy as np


class StockPredictor:
    """
        StockPredictor enables the trained model to make predictions on current stock prices
    """

    def __init__(self, model_name, model_type) -> None:

        self.__fetch_data(['AAPL'])
        self.__set_env()

        if model_type == 'ppo':
            self.model = PPO.load(model_name)
        else:
            raise TypeError

    def __set_env(self) -> None:
        """
            Set the new environment, containing the current stock prices
        """
        self.gym_env = StockTradingEnvCashpenalty(
            df=self.data, 
            initial_amount=1e6, 
            hmax=5000, 
            turbulence_threshold=None, 
            currency='$',
            buy_cost_pct=3e-3, 
            sell_cost_pct=3e-3, 
            cash_penalty_proportion=0.2,
            cache_indicator_data=True, 
            daily_information_cols = ['daily_variance', 'change', 'log_volume', 'close','day', 
                    'macd', 'rsi_30', 'cci_30', 'dx_30'],
            print_verbosity=500, 
            random_start=False
        )

        self.env = self.gym_env
        
    
    
    def __fetch_data(self, tickers) -> None:
        """
            Fetch the current stock prices and the prices of the pas three weeks

            TODO: Fetch current stock information and add to dataframe
        """
        self.data = YahooDownloader((date.today()-timedelta(days=21)), date.today(), tickers).fetch_data()
        self.__preprocess()
    

    def __preprocess(self) -> None:
        """
            Preprocess the stock data, add statistical indicators
        """
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

    def predict(self) -> None:
        """
            Make a prediction based on the current stock price

            TODO: Return the action to take buy/sell/qty etc
        """
        self.df_account_value, self.df_actions = DRLAgent.DRL_prediction(model=self.model, environment=self.env)
        print(self.df_actions, self.df_account_value)
        
    
