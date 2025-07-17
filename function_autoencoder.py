# ###############################################
# This file includes functions to generate data and calculate BER
# ###############################################

import numpy as np

# Functions for data generation
def generate_transmit_data(M, J, num, seed=0):
    #print('Generate transmit data: M = %d, seed = %d' %(M, seed))
    np.random.seed(seed)
    symbol_index = np.random.randint(M,size=num*J)
    X = np.zeros((num*J,M), dtype = 'float32')
    Y = np.zeros((num*J,M), dtype = 'float32')
    for i in range(num*J):
        X[i,symbol_index[i]] = 1
    X = np.reshape(X,[num,M*J])
    Y = X    
    return symbol_index, X, Y


def onehot2symbol(X):
    a = np.sqrt(0.1)
    QAM_16 = np.array([[1,1],[1,3],[3,1],[3,3],[1,-1],[1,-3],[3,-1],[3,-3],
                       [-1,1],[-1,3],[-3,1],[-3,3],[-1,-1],[-1,-3],[-3,-1],[-3,-3]])*a
    symbol = np.matmul(X, QAM_16)
    return symbol

def onehot2bit(X):
    BIT_16 = np.array([[0,0,0,0],[0,0,0,1],[0,0,1,0],[0,0,1,1],
                       [0,1,0,0],[0,1,0,1],[0,1,1,0],[0,1,1,1],
                       [1,0,0,0],[1,0,0,1],[1,0,1,0],[1,0,1,1],
                       [1,1,0,0],[1,1,0,1],[1,1,1,0],[1,1,1,1]])
    bit = np.matmul(X, BIT_16)
    return bit

def SER(M, J, sym_index_test, Y_pred, num_test):
    err = 0
    Y_pred = np.reshape(Y_pred,[num_test*J,M])
    for i in range(num_test*J):
        y_pred_index = np.argmax(Y_pred[i,:])
        if (y_pred_index != sym_index_test[i]):
          err = err + 1       
    ser = err/(num_test*J)
    return ser

def BER(X, sys, sym_index_test, Y_pred, num_test):
    err = 0
    Y_pred[Y_pred<0.5] = 0
    Y_pred[Y_pred>=0.5] = 1
    Y_pred = np.reshape(Y_pred,[num_test*sys.Num_User,sys.k])
    X = np.reshape(X,[num_test*sys.Num_User,sys.M])
    X_bit = onehot2bit(X)
    X_bit = np.reshape(X_bit,[num_test*sys.Num_User,sys.k])
    err = np.sum(np.abs(Y_pred - X_bit))        
    ber = err/(num_test*sys.Num_User*sys.k)
    return ber

def generate_rate_data(M, J):
    symbol_index = np.arange(M)
    X = np.tile(np.eye(M), (1, J))
    Y = X
    return symbol_index, X, Y

def calcul_rate(y, Rece_Ampli, Num_Antenna):
    z = y[:,0:Num_Antenna] + 1j*y[:,Num_Antenna:2*Num_Antenna]
    zt = y[:,0:Num_Antenna] - 1j*y[:,Num_Antenna:2*Num_Antenna]
    rate = np.log2(1 + np.matmul(z, zt.T)/(Rece_Ampli)**2) 
    mean_rate = np.mean(np.diag(rate))
    max_rate = np.max(np.diag(rate))
    return mean_rate, max_rate
    
    
    
    
    
    