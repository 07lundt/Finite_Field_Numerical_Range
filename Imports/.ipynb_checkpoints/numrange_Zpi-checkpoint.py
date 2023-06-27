import numpy as np
import matplotlib.pyplot as plt
from Imports.linalg_Zpi import linalg_Zpi

class numrange_Zpi():
    def __init__(self, n, p, M):
        # Load in norm-0 vectors
        str_norm0 = "Imports/norm0/p="+str(p)+"_n="+str(n)+".npz"
        self.norm0 = np.load(str_norm0)["norm0"]
        
        # Load in norm-1 vectors
        str_norm1 = "Imports/norm1/p="+str(p)+"_n="+str(n)+".npz"
        self.norm1 = np.load(str_norm1)["norm1"]
        
        self.linalg = linalg_Zpi(n=n, p=p)
        
        self.M = M
        self.n = n
        self.p = p
        
    def W0(self, plot=True):
        W0 = np.empty(len(self.norm0), dtype=np.csingle)
        
        for i in range(len(self.norm0)):
            x = self.norm0[i]
            xstar = self.linalg.sim_a(np.conj(x.transpose()))
            
            W0[i] = self.linalg.prod((xstar, self.M, x))
            
        out = np.unique(W0)
        
        if plot==True:
            plt.scatter(out.real, out.imag, s=200, c='red')
            plt.axis([0, self.p-1, 0, self.p-1])
            plt.show
            
        return out
    
    def W1(self, plot=True):
        W1 = np.empty(len(self.norm1), dtype=np.csingle)
        
        for i in range(len(self.norm1)):
            x = self.norm1[i]
            xstar = self.linalg.sim_a(np.conj(x.transpose()))
            
            W1[i] = self.linalg.prod((xstar, self.M, x))
            
        out = np.unique(W1)
        
        if plot==True:
            plt.scatter(out.real, out.imag, s=200, c='red')
            plt.axis([0, self.p-1, 0, self.p-1])
            plt.show
        
        return out    