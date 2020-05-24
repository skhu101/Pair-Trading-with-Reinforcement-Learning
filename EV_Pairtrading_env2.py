import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np

from portfolio import portfolio

class PairtradingEnv(gym.Env):
    """
    Description:
        Pair trading decision makeing based on reinforment learning algorithm
    Observation: 
        Type: Box(4)
        Num	Observation                 Min         Max
        0   price of stock1              
        1   number of stock1 in hand
        2   price of stock2
        3   number of stock2 in hand
        4   cash in hand                0            inf
        
    Actions:
        Variable: (stock1 trade ratio, stock2 trade ratio, trade_flag)
        Type:     (float, float, int)
        Num	Action
        stock1_act: the ratio of stock1 to sell/buy ;
        stock2_act: the ratio of stock2 to sell/buy  ;
        trade_flag: 0 (sell stock 1 based on stock1_act, buy stock 2 based on stock2_act),
                    1 (sell stock 2 based on stock1_act, buy stock 1 based on stock2_act),
                    2 (sell all stock 1 and stock 2 in hand);
        
        Note: we only have three possible actions to deal with pair trading, i.e., 
        buy stock1, sell stock2; buy stock2, sell stock 1; sell all the stocks in hand
    Reward:
        Reward: the difference of portfolio at the current day and the previous day 
    Starting State:
        Choose randly one day in the historic data as the starting point and we have 
        [price of stock1, 0, price of stock2, 0, 10000]
    Episode Termination:
        Arriving the end of the historic data or satisfying the maximum trade period
        Considered solved when the portfolio is greater than or equal to 150000 over 100 consecutive trials.
    """
    
    metadata = {
        'render.modes': ['human']
    }

    def __init__(self, stock1, stock2, cash=1e5, max_trade_period = 200):
        '''
        stock1, stock2 : price lists

        '''
        
        self.stock1 = stock1
        self.stock2 = stock2
        self.portfolio = portfolio(cash)

        self.observation_space = np.zeros(5)
        self.action_space = 4
        self.trade_info = None

        self.total_trade_period = len(stock1)
        self.max_trade_period = min(max_trade_period, self.total_trade_period)

        self.start_trade_time = 0
        self.current_trade_time = 0 
        
        self.start_time = None
        self.end_time = None

        self.seed()
        self.viewer = None

        
    def show(self):
        print('total_trade_period : ' + str(self.total_trade_period))
        print('max_trade_period : ' + str(self.max_trade_period))
        
        print('start_trade_time :   ' + str(self.start_trade_time))
        print('current_trade_time : ' +str(self.current_trade_time))
        
        self.portfolio.show()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        '''
        trade_flag : 
            0: open position, 
            1: close position, 
            2: add position, 
            3 : close position and open position of another direction
        '''

        stock1_num, stock2_num, trade_flag = action

        stock1_price = self.stock1[self.current_trade_time]
        stock2_price = self.stock2[self.current_trade_time]
        
        #print(stock1_price,stock2_price)

        # reward is the rate of increase
        old_value = self.portfolio.value    
        self.portfolio.updatePrice(stock1_price, stock2_price)
        # reward = self.portfolio.value / old_value - 1.0 
        reward = self.portfolio.value / 10000 - 1.0 
        # reward *= 100

 
        if trade_flag == 0:
            self.portfolio.openPosition(num1 = stock1_num, num2 = stock2_num)
        elif trade_flag == 1:
            self.portfolio.closePosition()
        elif trade_flag == 2:
            self.portfolio.addPosition(num1 = stock1_num, num2 = stock2_num)
        elif trade_flag == 3:
            self.portfolio.closePosition()
            self.portfolio.updatePrice(stock1_price, stock2_price)
            self.portfolio.openPosition(num1 = stock1_num, num2 = stock2_num)


        done = False
        if self.current_trade_time >= self.start_trade_time + self.max_trade_period - 1:
            done = True 
        self.current_trade_time += 1

        price1 = self.stock1[self.current_trade_time]
        price1diff = price1 - self.stock1[self.current_trade_time-1]
        price2 = self.stock2[self.current_trade_time]
        price2diff = price2 - self.stock2[self.current_trade_time-1]

        self.trade_info = [self.portfolio, price1, price1diff, price2, price2diff]
        # return np.array([self.portfolio, price1, price1diff, price2, price2diff]), reward, done, {}        
        return np.array([self.portfolio.stock1_num, price1, self.portfolio.stock2_num, price2, \
            self.portfolio.cash]), reward, done, {}

    def reset(self, test=False):
        self.portfolio = portfolio(cash=10000)

        # if not test:
        self.start_trade_time = np.random.randint(self.total_trade_period-self.max_trade_period-5, size=1)[0]
        # else:
        #     self.start_trade_time = 0
        #     self.max_trade_period = self.total_trade_period-1
        self.current_trade_time = self.start_trade_time

        price1 = self.stock1[self.current_trade_time]
        price1diff = 0 
        price2 = self.stock2[self.current_trade_time]
        price2diff = 0

        self.trade_info = [self.portfolio, price1, price1diff, price2, price2diff]
        # return np.array([self.portfolio, price1, price1diff, price2, price2diff])
        return np.array([self.portfolio.stock1_num, price1, self.portfolio.stock2_num, price2, self.portfolio.cash])

    def render(self, mode='human'):
        """Render the visualization process."""
        raise NotImplementedError()

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

