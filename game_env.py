import numpy as np




class energy_game:

    def __init__(self,G,capacity,S,A):
        self.S = S
        self.A = A
        self.s = 0
        self.a = 0
        self.s_prev = 0
        self.G = G
        self.asked = 0
        self.capacity = capacity
        #self.lambda2 = lambda2


    def step(self,a):
        self.asked = max(0,a-self.s)
        self.s_prev = self.s
        g = np.random.choice(self.G)
        if a <= self.s:
            self.s = self.s-a+g
        else:
            self.s = g
        self.s = min(self.capacity,self.s)
        self.a = a

    def get_reward(self,price):
        max_reward = (self.A-1)**2
        r = self.a **2 - price*self.asked
        return r/max_reward

    def get_P_matrix(self):
        P = np.zeros([self.S,self.S,self.A])
        for s in range(self.S):
            for a in range(self.A):
                if s <= a:
                    P[0:self.G,s,a] += 1/self.G
                else:
                    if s-a+self.G-1>self.capacity:
                        P[s-a:self.capacity,s,a] += 1/self.G
                        P[self.capacity, s, a] += (-self.capacity+(s-a+self.G))*(1 / self.G)
                    else:
                        P[s - a:s-a+self.G, s, a] += 1 / self.G
        #P = P.reshape([self.S,self.S*self.A])
        return P





