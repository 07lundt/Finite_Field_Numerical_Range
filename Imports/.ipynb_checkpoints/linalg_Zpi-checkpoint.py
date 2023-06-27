'''-----------------------------------------------------------------------
This is a set of tools to do basic linear algebra on square matrices 
with entries from Zp[i].
-----------------------------------------------------------------------'''

import numpy as np
import itertools
from itertools import permutations

class linalg_Zpi():
    # When initializing the tool, input the dimension n and prime p
    def __init__(self, n, p):
        self.size = n
        self.mod = p
        
        # Create table of all possible fractions in Zp
        div_tbl = np.zeros((self.mod, self.mod), dtype=int)

        for i in range(1,self.mod):
            for j in range(1,self.mod):
                for elem in range(1,self.mod):
                    if (elem*i)%self.mod == j:
                        div_tbl[i,j] = elem
                        break

        self.div_tbl = div_tbl
        
        # Create a list of all units in Zp
        combine = list(itertools.combinations_with_replacement(np.arange(0,self.mod), 2))
        store = np.expand_dims(np.array([1,0]), axis=0)
        
        for row in combine:
            sq_sum = 0
            for elem in row:
                sq_sum += elem**2
            if sq_sum % self.mod == 1:
                array = np.array(row)
                expanded = np.expand_dims(array, axis=0)
                store = np.append(store, expanded, axis=0)

        permute = np.expand_dims(np.array([1,0]), axis=0)

        for row in store:
            p = list(permutations(row, 2))
            for item in p:
                array = np.array(item)
                expanded = np.expand_dims(array, axis=0)
                permute = np.append(permute, expanded, axis=0)
                
        permute = np.unique(permute, axis=0).reshape(-1, 2)
        
        u = np.empty((permute.shape[0]), dtype=np.csingle)
        
        for i in range(permute.shape[0]):
            u[i] = complex(permute[i,0], permute[i,1])
            
        self.units = u
        
    # Reduce a scalar to an element of Zp[i]
    def sim_s(self, s):
        return complex(s.real%self.mod, s.imag%self.mod)
        
    # Reduce an array to elements of Zp[i]
    def sim_a(self, a):
        for i in range(a.shape[0]):
            for j in range(a.shape[1]):
                a[i,j] = self.sim_s(a[i,j])

        return a
    
    # Divide numbers in Zp
    def div_real(self, a, b):
        if b%self.mod==0:
            raise Exception("FREAK OUT")
            
        return self.div_tbl[b%self.mod, a%self.mod]
    
    # Divide numbers in Zp[i]
    def div_complex(self, z1, z2):
        num = z1 * np.conj(z2)
        den = z2.real**2 + z2.imag**2

        num = self.sim_s(num)
        den = int(den%self.mod)

        a = self.div_real(int(num.real), den)
        b = self.div_real(int(num.imag), den)

        return complex(a,b)
    
    # Create a square matrix of dimension n
    def mat(self, elems):
        M = np.empty((self.size, self.size), dtype=np.csingle)
        index = 0
        
        for i in range(self.size):
            for j in range(self.size):
                M[i,j] = elems[index]
                index += 1
                
        return self.sim_a(M)
    
    # Create a column vector of dimension n
    def col(self, elems):
        M = np.empty((self.size, 1), dtype=np.csingle)
        index = 0
        
        for i in range(self.size):
            M[i,0] = elems[index]
            index += 1
    
        return self.sim_a(M)
    
    # Create a row vector of dimension n
    def row(self, elems):
        M = np.empty((1, self.size), dtype=np.csingle)
        index = 0
        
        for i in range(self.size):
            M[0,i] = elems[index]
            index += 1
            
        return self.sim_a(M)
    
    # Multiply a tuple of arrays
    def prod(self, Ms):        
        M0 = Ms[0]
        
        for M in Ms[1:]:
            M0 = np.matmul(M0, M)
        
        M = self.sim_a(M0)
        
        return M
    
    # Check if two column vectors are Zp[i] multiples of each other
    def mult(self, v1, v2):
        # This helper function determines whether two elements of Zp[i] are 
        # Zp[i] multiples of each other by row-reducing a matrix 
        def solve_mult(M):
            if M[0,0] == 0:
                temp = np.array(M[0])
                M[0] = M[1]
                M[1] = temp
            
            M[0] = self.div_real(np.array(M[0]), M[0,0])
            M[1] -= M[0]*M[1,0]
            M[1]%self.mod 
            
            M[1] = self.div_real(np.array(M[1]), M[1,1])
            M[0] -= M[1]*M[0,1]
            M[0]%self.mod
            
            # Returns the multiplier
            return complex(M[0,2], M[1,2])
        
        store = np.empty((self.size, 1), dtype=np.csingle)
        
        inds = np.arange(0, self.size)
        
        for i in range(self.size):
            real1 = int(v1[i].real.item())
            imag1 = int(v1[i].imag.item())
            
            real2 = int(v2[i].real.item())
            imag2 = int(v2[i].imag.item())
            
            M = np.array([[real1, -imag1, real2],
                          [imag1,  real1, imag2]])
            
            # Check if the first number is 0 to avoid 0-division errors
            if real1==0 and imag1==0:
                if real2==0 and imag2==0:
                    inds[i]=-1
                    store[i]=-1
                else:
                    return None
            else:
                store[i] = self.sim_s(solve_mult(M))
        
        # Check that all of the multipliers are the same 
        if np.all(np.take(store, inds[inds>=0]) == np.take(store, inds[inds>=0])[0]):
            return np.take(store, inds[inds>=0])[0].item()
        else:
            return None
        
    # Calculate the inverse of an invertible matrix and freak out if non-invertible
    def inv(self, M):
        M = np.concatenate((M, np.identity(self.size, dtype=int)), axis=1)

        for i in range(self.size-1):
            if M[i,i] == 0:
                for j in range(i+1, self.size):
                    if M[j, i] != 0:
                        temp = np.array(M[i])
                        M[i] = M[j]
                        M[j] = temp

            temp = M[i,i]
            for x in range(2*self.size):
                M[i,x] = self.div_complex(M[i,x], temp)

            for k in range(i+1, self.size):
                M[k] -= M[i]*M[k,i]
                for x in range(2*self.size):
                    M[k,x] = self.sim_s(M[k,x])

        temp = M[-1,self.size-1]
        for x in range(2*self.size):
            M[-1,x] = self.div_complex(M[-1,x], temp)
        
        for l in range(self.size-1,0,-1):
            for n in range(l-1,-1,-1):
                M[n] -= M[l]*M[n,l]
                for x in range(2*self.size):
                    M[n,x] = self.sim_s(M[n,x])

        return self.sim_a(M[:,-self.size:])
    
    # Generate matrix from eigenvectors and eigenvalues
    def from_eigen(self, vecs, vals):
        M = np.concatenate(vecs, axis=1)
        M_inv = self.inv(M) 
        V = np.diag(vals)

        return self.prod((M,V,M_inv))