from sklearn.datasets import load_boston
from sklearn import linear_model
import pandas as pd
import numpy as np

boston = load_boston()
boston_df = pd.DataFrame(boston.data, columns=boston.feature_names)
boston_df['PRICE'] = pd.DataFrame(boston.target)

A = np.array([boston_df['CRIM']
             ,boston_df['ZN']
             ,boston_df['INDUS']
             ,boston_df['CHAS']
             ,boston_df['NOX']
             ,boston_df['RM']
             ,boston_df['AGE']
             ,boston_df['DIS']
             ,boston_df['RAD']
             ,boston_df['TAX']
             ,boston_df['PTRATIO']
             ,boston_df['B']
             ,boston_df['LSTAT']
             ]).T

b = boston_df['PRICE']

N = boston.data.shape[0]
M = boston.data.shape[1]

class ADMM:

    max_loop = 10000

    def __init__(self,x,z,y,lambd,rho):
        self.x = x
        self.z = z
        self.y = y
        self.lambd = lambd
        self.rho = rho

    def S(self,a,k):
        if k > a :
            return k - a
        elif -a <= k and k <= a :
            return 0
        elif k < -a:
            return k + a

    def fit(self):
        for loop in range(self.max_loop):
            self.update()

    def update(self):
        self.update_x()
        self.update_y()
        self.update_z()

    def update_x(self):
        Q = np.dot(A.T,A) / N + self.rho * np.identity(M)
        inv_Q = np.linalg.inv(Q)
        P = np.dot(A.T,b) / N + self.rho * self.z - self.y

        self.x = np.dot(inv_Q,P)

    def update_z(self):
        for i in range(M):
            self.z[i] = self.S(self.lambd / self.rho, self.x[i] + self.y[i] / self.rho)

    def update_y(self):
        self.y = self.y + self.rho * (self.x - self.z)

    def pridict(self,u):
        return np.dot(self.x,u)



admm = ADMM(x = np.dot(A.T,b) / N, z = np.dot(A.T,b) / N, y = [0] * M, lambd = 1.0, rho = 1.0)
admm.fit()
print("Lasso by ADMM")

print("coef_")
print(np.round(admm.x,10))

print('\n')

print("Pridicted value : Actual value")
for i in range(N):
    print(admm.pridict(A[i])," : ",b[i])
