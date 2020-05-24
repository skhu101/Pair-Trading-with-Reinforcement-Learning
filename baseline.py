from EV_Pairtrading_env2 import PairtradingEnv

class baselineConfig(object):
    def __init__(self, mean, std, beta, \
                 upthreshold = 0.9, lowthreshold = -0.9, \
                 neutralup = 0.6, neutrallow = -0.6, percentileCash = 0.6):
        self.upthreshold = upthreshold
        self.lowthreshold = lowthreshold
        
        self.neutralup = neutralup
        self.neutrallow = neutrallow
        
        self.percentileCash = percentileCash

        self.cashcandidate = [0.6, 1, 1.5, 2]
        self.betacandidate = [5,7,9,11,13,15]
        
        self.mean = mean
        self.std = std
        self.beta = beta


class baseline:
    def __init__(self, env, config):
        self.env = env
        
        self.upthreshold = config.upthreshold
        self.lowthreshold = config.lowthreshold
        
        self.neutralup = config.neutralup
        self.neutrallow = config.neutrallow
        
        self.percentileCash = config.percentileCash

        self.cashcandidate = config.cashcandidate
        self.betacandidate = config.betacandidate
        
        self.mean = config.mean
        self.std = config.std
        self.beta = config.beta
        
        self.lastZscore = 0 
        
    def reset(self):
        self.lastZscore = 0
        
       
        
    def compute_action(self, act_cash, act_beta):
        # action = [stock1_num, stock2_num, trade_flag]
        # trade_flag : 
        # 0: open position
        # 1: close position
        # 2: add position
        #3 : close position and open position of another direction
        portfolio = self.env.trade_info[0]
        price1 = self.env.trade_info[1]
        price1diff = self.env.trade_info[2]
        price2 = self.env.trade_info[3]
        price2diff = self.env.trade_info[4]

        self.percentileCash = self.cashcandidate[act_cash]
        self.beta = self.betacandidate[act_beta]
        
        # use 60% cash 
        use_cash = portfolio.cash * self.percentileCash 
    
        # calculate z-score
        spread = self.beta * price1diff - price2diff 
        zScore = (spread - self.mean)/self.std
    
        trade_flag = 4 # do nothing 
        stock1_num = 0
        stock2_num = 0
        # choose an action
        if zScore > self.upthreshold :
            # spread is too high, sell stock1 , buy stock 2
            stock2_num = 100 * int(use_cash / price2 / 100)
            stock1_num = -100 * int(self.beta*stock2_num/100)
        
            if portfolio.stock1_num == 0 and portfolio.stock2_num == 0 :
                trade_flag = 0 
            elif portfolio.stock1_num < 0 :
                trade_flag = 2
            elif portfolio.stock1_num > 0 : 
                trade_flag = 3
            
    
        elif zScore < self.lowthreshold :
            # spread is too low, buy stock 1, sell stock 2
            stock1_num = 100 * int(use_cash / price1 / 100)
            stock2_num = -100 * int(stock1_num/self.beta/100)
        
            if portfolio.stock1_num == 0 and portfolio.stock2_num == 0 :
                trade_flag = 0 
            elif portfolio.stock1_num > 0 :
                trade_flag = 2
            elif portfolio.stock1_num < 0 : 
                trade_flag = 3
            
        elif ( (zScore < self.neutralup and zScore > self.neutrallow) or (self.lastZscore * zScore < 0) ) and (portfolio.stock1_num != 0):
            # spread is near neutral
            # close position
            trade_flag = 1 
            stock1_num = 0
            stock2_num = 0
        
            
        self.lastZscore = zScore
        act = [stock1_num,stock2_num,trade_flag]
        
        #print("zScore : {}, act : {}".format(zScore, act))
        return act


        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    