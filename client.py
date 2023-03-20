import numpy as np




class energy_client:
    def __init__(self,A,S,rho):
        self.A = A
        self.S = S
        self.rho = np.reshape(rho,(S,A))
        self.policy = np.copy(self.rho)
        self.R = np.zeros([S,A])
        for s in range(S):
            self.policy[s] = self.policy[s]/np.sum(self.policy[s])

    def get_action(self,s):
        a = np.random.choice(self.A,p=self.policy[s])
        return a

    def update_R(self,r,s,a):
        self.R[s,a]+= r/self.policy[s,a]


#e = energy_client(3,2,np.ones(6))
#print(e.get_action(1))