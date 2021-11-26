from finrl.neo_finrl.env_stock_trading.env_stocktrading_cashpenalty import StockTradingEnvCashpenalty
from finrl.drl_agents.stablebaselines3.models import DRLAgent
from finrl.plot import backtest_stats, backtest_plot


class Trainer:
    """
        Trainer class that provides an implementation of the Agents, model and environments
    """
    def __init__(self, train_data, trade_data) -> None:

        self.train_data = train_data
        self.trade_data = trade_data

    
    # Function to create and set a specific environment
    def set_environment(
        self, 
        type, 
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
        random_start=True
    ) -> None:

        """
            Switch function for the creation of the environements
        """

        self.initial_amount = initial_amount
        self.hmax = hmax
        self.turbulence_threshold = turbulence_threshold
        self.currency = currency
        self.buy_cost_pct = buy_cost_pct
        self.sell_cost_pct = sell_cost_pct
        self.cash_penalty_proportion = cash_penalty_proportion
        self.cache_indicator_data = cache_indicator_data
        self.daily_information_cols = daily_information_cols
        self.print_verbosity = print_verbosity
        self.random_start = random_start


        if type == 'train':
            self.__set_train_environment()
        elif type == 'trade':
            self.__set_trade_environment()


    def __set_train_environment(self) -> None:
        """
            Create and set the train environment
        """
        print("Creating training environment")
        self.gym_env_train = StockTradingEnvCashpenalty(
                                df=self.train_data, 
                                initial_amount=self.initial_amount,
                                hmax=self.hmax, 
                                turbulence_threshold=self.turbulence_threshold, 
                                currency=self.currency,
                                buy_cost_pct=self.buy_cost_pct,
                                sell_cost_pct=self.sell_cost_pct,
                                cash_penalty_proportion=self.cash_penalty_proportion,
                                cache_indicator_data=self.cache_indicator_data,
                                daily_information_cols=self.daily_information_cols, 
                                print_verbosity=self.print_verbosity, 
                                random_start=self.random_start)

        self.env_train,_ = self.gym_env_train.get_sb_env()


    def __set_trade_environment(self):
        """
            Create and set the trade environment
        """
        print("Creating trading environment")
        self.gym_env_trade = StockTradingEnvCashpenalty(
                                df=self.trade_data, 
                                initial_amount=self.initial_amount,
                                hmax=self.hmax, 
                                turbulence_threshold=self.turbulence_threshold, 
                                currency=self.currency,
                                buy_cost_pct=self.buy_cost_pct,
                                sell_cost_pct=self.sell_cost_pct,
                                cash_penalty_proportion=self.cash_penalty_proportion,
                                cache_indicator_data=self.cache_indicator_data,
                                daily_information_cols=self.daily_information_cols, 
                                print_verbosity=self.print_verbosity, 
                                random_start=False)

        self.env_trade,_ = self.gym_env_trade.get_sb_env()

    def get_env(self) -> StockTradingEnvCashpenalty:
        """
            Return the trade environment
        """
        return self.env_trade


  
    def set_agent(self) -> None:
        """
            Create and set the Agent and model
        """
        self.agent = DRLAgent(env=self.env_train)
        ppo_params ={
            'n_steps': 256, 
            'ent_coef': 0.0, 
            'learning_rate': 0.000005, 
            'batch_size': 1024, 
            'gamma': 0.99}

        policy_kwargs = {
        #     "activation_fn": ReLU,
            "net_arch": [1024 for _ in range(10)], 
        #     "squash_output": True
        }

        self.model = self.agent.get_model("ppo",  
                                model_kwargs = ppo_params, 
                                policy_kwargs = policy_kwargs, verbose = 0)

    
    def train(
        self,
        total_timesteps=10000, 
        eval_freq=500,
        log_interval=1, 
        tb_log_name='env_cashpenalty_highlr',
        n_eval_episodes=1
    ) -> None:

        """
            Train the model
        """

        self.model.learn(
            total_timesteps=total_timesteps,
            eval_env = self.env_trade,
            eval_freq = eval_freq,
            log_interval = log_interval,
            tb_log_name = tb_log_name,
            n_eval_episodes = n_eval_episodes
        )

    def save_model(self, name) -> None:
        """
            Save the model
        """
        self.model.save(name)

    
    def backtest(self, dates):
        """
            Backtest the performance of the trained model
        """
        print("Starting backtesting")

        self.df_account_value, self.df_actions = DRLAgent.DRL_prediction(model=self.model, environment=self.gym_env_trade)
        
        _ = backtest_stats(account_value=self.df_account_value, value_col_name = 'total_assets')

        return (backtest_plot(self.df_account_value, 
             baseline_ticker = '^DJI', 
             baseline_start = dates[0],
             baseline_end = dates[1], value_col_name = 'total_assets'))








