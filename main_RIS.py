import scipy.io as sio                     # import scipy.io for .mat file I/O 
import numpy as np                         # imp
import matplotlib.pyplot as plt
import random
import scipy.io as sio
from scipy.io import loadmat
import torch
from torch.autograd import Variable
import function_autoencoder as af          # import our function file
#import function_dnn_pytorch_procoder as dfp
import function_dnn_pytorch_ris_procoder as dfpr
import time
import h5py
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

class Para:
    def __init__(self, Pos_User_x, SNR_train, Channel_BS2RIS, Channel_RIS2User, Channel_BS2User):
        self.M = 16   
        self.k = 4
        self.Num_train = 10000
        self.Num_test = 100000
        self.Num_vali = 10000
        self.SNR_train = SNR_train
        self.Epoch_train = 60
        self.Batch_Size = 160
        self.LR_Factor = 1.2
        self.load_model = 0
        
        self.Pos_BS = np.array([[0,-40]])
        self.Pos_RIS = np.array([[60,10]])
        self.Pos_User_x = np.array(Pos_User_x)#arange(15,66,5)  
        self.Pos_User = np.zeros([1,2])#np.array(Pos_User)#
        self.Pos_User[0,0] = self.Pos_User_x         #Y direction defaults to 0
        
        
        self.Num_BS = self.Pos_BS.shape[0]      #Number of base stations
        self.Num_BS_Antenna = 8                 #Number of base station antennas
        self.Num_Subcarrier = 1                 #Number of subcarriers
        self.Power_Max = 1                      #Maximum power limit of base station
        self.Num_User = 1
        self.Num_User_Antenna = 2               #Number of user antennas
        self.Num_RIS = self.Pos_RIS.shape[0]    #RISQuantity
        self.Num_RIS_Element = 256              #Number of units on each RIS
        self.Phase_Matrix = Variable(torch.ones([1,self.Num_RIS_Element*2])/np.sqrt(2), requires_grad=False)
        self.RIS_radio = 1 
        self.Active_Element = np.random.randint(self.Num_RIS_Element,size=int(self.RIS_radio*self.Num_RIS_Element))
        self.Active_Element = np.hstack([self.Active_Element, self.Active_Element + int(self.Num_RIS_Element)])
        self.Active_Element = np.sort(self.Active_Element)
        
        # 2D distance matrix 
        self.Dis_BS2RIS = self.Distance_Matrix(self.Pos_BS, self.Pos_RIS)
        self.Dis_BS2User = self.Distance_Matrix(self.Pos_BS, self.Pos_User)
        self.Dis_RIS2User = self.Distance_Matrix(self.Pos_RIS, self.Pos_User)
        

        
        # High-dimensional channel matrix
        self.Channel_BS2RIS = self.Channel_Matrix(Channel_BS2RIS[0:self.Num_RIS_Element,0:self.Num_BS_Antenna], self.Num_BS, self.Num_BS_Antenna, 
                                                  self.Num_RIS, self.Num_RIS_Element, self.Dis_BS2RIS, 2, 1, 2)
        self.Channel_RIS2User = self.Channel_Matrix(Channel_RIS2User[0:self.Num_User_Antenna,0:self.Num_RIS_Element], self.Num_RIS, self.Num_RIS_Element, 
                                                    self.Num_User, self.Num_User_Antenna, self.Dis_RIS2User, 2, 0, 2)      
        self.Channel_BS2User = self.Channel_Matrix(Channel_BS2User[0:self.Num_User_Antenna,0:self.Num_BS_Antenna], self.Num_BS, self.Num_BS_Antenna, 
                                                   self.Num_User, self.Num_User_Antenna, self.Dis_BS2User, 3, 0, 3)

        self.SNR_train = 10**(self.SNR_train/10)
        self.Rece_Ampli = 10**(-3.5) #10 ** (-3)* 25.17935662 ** (- 3)
        self.SNR_train = self.SNR_train/((self.Rece_Ampli)**(2))
        
    def Distance_Matrix(self, A, B):
        NumofA = A.shape[0]
        NumofB = B.shape[0]
        Dis = np.zeros([NumofA, NumofB])
        for i in range(NumofA):
            Dis[i,:] = np.linalg.norm(A[i,:] - B, axis=1)
        return Dis
    
    def Channel_Matrix(self, Channel, Num_N, Nt, Num_M, Mr, Dis, gain, LOS, fading):
        Path_Loss = np.sqrt(10 ** (-fading)* Dis ** (- gain))
        Path_Loss = Path_Loss.repeat(Mr,axis=0)
        Path_Loss = Path_Loss.repeat(Nt,axis=1)
        Channel = Path_Loss * Channel
        return Channel

ChaData_BS2User = np.array(sio.loadmat('channel_&_model/ChaData_BS2User.mat')['Channel_BS2User'])
ChaData_RIS2User = np.array(sio.loadmat('channel_&_model/ChaData_RIS2User.mat')['Channel_RIS2User'])
ChaData_BS2RIS = np.array(sio.loadmat('channel_&_model/ChaData_BS2RIS.mat')['Channel_BS2RIS'])
ChaData_BS2RIS.dtype='complex128'
ChaData_RIS2User.dtype='complex128'
ChaData_BS2User.dtype='complex128'
print('ChaData_BS2RIS:', ChaData_BS2RIS.shape)
print('ChaData_RIS2User:', ChaData_RIS2User.shape)
print('ChaData_BS2User:', ChaData_BS2User.shape)
Sample = 300#ChaData_BS2RIS.shape[0]

X_User = np.arange(0,101,10)
#SNR_dB_train = np.array([2,6,7,9,10,5,0,6,13,18,20])
SNR_dB_train = np.array([2,1,0,0,-2,-5,-8,-6,0,2,4])
SNR_dB = np.arange(-5,21,1)
Ber = np.zeros([Sample, X_User.shape[0], SNR_dB.shape[0]]) 
mean_rate = np.zeros([Sample,X_User.shape[0]])
max_rate = np.zeros([Sample,X_User.shape[0]])

R_Error = np.zeros([Sample*X_User.shape[0],150])

for s in range(0,Sample):
    Channel_BS2RIS = ChaData_BS2RIS[s]
    Channel_RIS2User = ChaData_RIS2User[s]
    Channel_BS2User = ChaData_BS2User[s]
    for x in range(X_User.shape[0]):
        sys = Para(X_User[x], SNR_dB_train[x], Channel_BS2RIS, Channel_RIS2User, Channel_BS2User)
        print('Sample:',s,' x:',x, 'position:',X_User[x], 'snr',sys.SNR_train)
        sym_index_train, X_train, Y_train = af.generate_transmit_data(sys.M, sys.Num_User, sys.Num_train, seed=random.randint(0,1000))
        #dfp.train(sym_index_train, X_train, Y_train, sys, sys.SNR_train)
        if x == 0:
            sys.Epoch_train = 200
            sys.LR_Factor = 1.05
            sys.load_model = 0
        time_start=time.time()
        dfpr.train(sym_index_train, X_train, Y_train, sys, sys.SNR_train)
        time_end=time.time()
        print('totally cost',time_end-time_start)
        
        sym_index_rate, X_rate, Y_rate = af.generate_rate_data(sys.M, sys.Num_User)
        Y_pred, y_rate = dfpr.test(sym_index_rate, X_rate, sys, 10**1000)
        mean_rate[s,x], max_rate[s,x] = af.calcul_rate(y_rate, sys.Rece_Ampli, sys.Num_User_Antenna)
        print('The mean rate and max rate are %0.8f, %0.8f, at x = %d m'%(mean_rate[s,x], max_rate[s,x], X_User[x]))
        
        ber_wr = np.zeros(SNR_dB.shape)
        ber = np.zeros(SNR_dB.shape)
        
        for i_snr in range(SNR_dB.shape[0]):
            SNR = 10**(SNR_dB[i_snr]/10)/(sys.Rece_Ampli)**2
            sym_index_test, X_test, Y_test = af.generate_transmit_data(sys.M, sys.Num_User, sys.Num_test, seed=random.randint(0,1000))
    #        #Y_pred = dfp.test(sym_index_test, X_test, sys, SNR)
    #        #ber_wr[i_snr] = af.BER(sys.M, sys.Num_User, sym_index_test, Y_pred, sys.Num_test)
            Y_pred, y_receiver = dfpr.test(sym_index_test, X_test, sys, SNR)
            ber[i_snr] = af.BER(X_test, sys, sym_index_test, Y_pred, sys.Num_test)
            print('The BER at SNR=%d is %0.8f,  %0.8f' %(SNR_dB[i_snr],ber_wr[i_snr],ber[i_snr]))
        Ber[s,x,:] = ber

#sio.savemat('figure(60,10)/E2E_Ber_'+str(sys.Num_RIS_Element)+'.mat', mdict={'Ber': Ber})


