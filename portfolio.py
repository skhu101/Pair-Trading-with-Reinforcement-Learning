from prettytable import PrettyTable

class portfolio:
    def __init__(self, cash=1e5):
        self.stock1_num = 0
        self.stock1_price_cur = 0
        #self.stock1_price_avg = 0

        self.stock2_num = 0
        self.stock2_price_cur = 0
        #self.stock2_price_avg = 0

        self.cash = cash
        self.value = cash
        
    def show(self):
        table = PrettyTable(['Stock', 'Number of holding', 'Current price'])
        table.add_row(['stock1', self.stock1_num, self.stock1_price_cur])
        table.add_row(['stock2', self.stock2_num, self.stock2_price_cur])
        print(table)
        print('Cash : ' + str(self.cash))
        print('Total value : ' + str(self.value))
    
    def check(self):
        # this function is for test
        # check if the update is satisfied.
        
        value1 = self.stock1_num * self.stock1_price_cur
        value2 = self.stock2_num * self.stock2_price_cur
        
        if abs(value1+value2+self.cash-self.value) > 1:
            print("The portfolio is not equal")
            
    

    def updatePrice(self, stock1_price = None, stock2_price = None):
        # updata portfolio
        # update stock_price_cur
        if stock1_price:
            self.value += self.stock1_num * (stock1_price - self.stock1_price_cur)
            self.stock1_price_cur = stock1_price
        
        if stock2_price:        
            self.value += self.stock2_num * (stock2_price - self.stock2_price_cur)
            self.stock2_price_cur = stock2_price
            
        self.check()
        return 

    # when exceed an action, prices must be updated first
    # three actions : add position, open position, close position

    def openPosition(self, num1 = None, num2 = None):
        # stock1_num = stock2_num = 0 
        if num1:
            self.stock1_num = num1
            self.cash -= num1 *  self.stock1_price_cur
        
        if num2:
            self.stock2_num = num2
            #self.stock2_price_avg = self.stock2_price_cur
            self.cash -= num2 *  self.stock2_price_cur
            
        self.check()
        
        return 


    def closePosition(self):
        if self.stock1_num != 0:
            self.cash += self.stock1_num * self.stock1_price_cur
            
            self.stock1_num = 0
            #self.stock1_price_cur = 0
            #self.stock1_price_avg = 0
            
        if self.stock2_num != 0:
            self.cash += self.stock2_num * self.stock2_price_cur
            
            self.stock2_num = 0
            #self.stock2_price_cur = 0
            #self.stock2_price_avg = 0
            
        self.check()

        return 

    def addPosition(self, num1 = None, num2 = None):
        # assert : self.stocki_num * numi > 
        if num1:
            self.cash -= num1 *  self.stock1_price_cur
            #self.stock1_price_avg = (self.stock1_price_cur * num1 + self.stock1_price_avg * self.stock1_num) / (self.stock1_num + num1)
            self.stock1_num += num1
            
        if num2:
            self.cash -= num2 *  self.stock2_price_cur
            
            #self.stock2_price_avg = (self.stock2_price_cur * num2 + self.stock2_price_avg * self.stock2_num) / (self.stock2_num + num2)
            self.stock2_num += num2
            
        self.check()
            
        return





















       
