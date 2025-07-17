#!/usr/bin/env python

# Generative Adversarial Networks (GAN) example in PyTorch.
# See related blog post at https://medium.com/@devnag/generative-adversarial-networks-gans-in-50-lines-of-code-pytorch-e81b79659e3f#.sch4xgsa9

import random
import function_autoencoder as af
import numpy as np

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.nn import init
import torch.autograd as autograd
import matplotlib.pyplot as plt
import scipy.io as sio

# Data params
t_learning_rate = 0.001  #GANs 0.0001 Resnet-GANS 0.001   
r_learning_rate = 0.001  #GANs 0.0005 Resnet-GANS 0.0005
ris_learning_rate = 0.001
optim_betas = (0.9, 0.999)
print_interval = 30
weight_gain = 1
bias_gain = 0.1
g_weight_gain = 0.1
g_bias_gain = 0.1
weight_decay=0.0001

    
class BS(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BS, self).__init__()
        
        #self.precoder = nn.Embedding(1, output_size)
        self.map1 = nn.Linear(input_size, hidden_size)
        self.map2 = nn.Linear(hidden_size, hidden_size)
        self.map3 = nn.Linear(hidden_size, output_size)

        torch.nn.init.xavier_normal_(self.map1.weight, weight_gain)   #initialize the weights
        torch.nn.init.constant(self.map1.bias, bias_gain)             #initialize the biases
        torch.nn.init.xavier_normal_(self.map2.weight, weight_gain)
        torch.nn.init.constant(self.map2.bias, bias_gain)
        
    def forward(self, x, cuda_gpu):
        x = F.relu(self.map1(x))        #to get non linearity
        #x = self.bn1(x) 
        x = F.relu(self.map2(x))
        #x = self.bn2(x)
        
        return self.map3(x)

class RIS(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RIS, self).__init__()   
        self.phase = nn.Embedding(1, output_size)
        
    def forward(self, x0, cuda_gpu): 
        m,n = x0.shape                          #2d tensor with shape
        
        p1 =  self.phase(torch.tensor(0))
        norm = torch.empty(1, int(n/2))
        x4 = torch.empty(m, n)
        if (cuda_gpu):
            norm = norm.cuda()
            x4 = x4.cuda()
        p1 = torch.reshape(p1, [2,int(n/2)])
        norm[0,:] = torch.norm(p1,2,0)
        pmax = torch.max(norm[0,:])
        p2 = p1/pmax
        x3 = torch.reshape(p2, [1,n])
        x3 = x3.expand(m,n)
        x4[:,0:int(n/2)] = x3[:,0:int(n/2)]*x0[:,0:int(n/2)] - x3[:,int(n/2):n]*x0[:,int(n/2):n]
        x4[:,int(n/2):n] = x3[:,0:int(n/2)]*x0[:,int(n/2):n] + x3[:,int(n/2):n]*x0[:,0:int(n/2)]
        return x4
    
class Receiver(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Receiver, self).__init__()
        self.map1 = nn.Linear(input_size, hidden_size, bias=True)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.map2 = nn.Linear(hidden_size, output_size, bias=True)
        self.bn2 = nn.BatchNorm1d(output_size)
        
    def forward(self, x):
        x = F.relu(self.map1(x))
        x = self.bn1(x)
        x = (self.map2(x))
        x = self.bn2(x)
        return torch.sigmoid(x)

def extract(v):
    return v.data.storage().tolist()

def stats(d):
    return [np.mean(d), np.std(d)]

def get_generator_input_sampler():
    return lambda m, n: torch.rand(m, n)

def ini_weights(sys, cuda_gpu, gpus):
    B = BS(input_size=sys.k * sys.Num_User, hidden_size=4*sys.M, output_size=sys.Num_BS_Antenna*2)
    Ris= RIS(input_size=int(sys.Num_RIS_Element*sys.RIS_radio)*2, hidden_size=4*sys.M, output_size=sys.Num_RIS_Element*2)
    R=[]
    for i in range(sys.Num_User):
        R.append(Receiver(input_size=sys.Num_User_Antenna*2, hidden_size=2*sys.M, output_size=sys.k))
    if(cuda_gpu):
        B = torch.nn.DataParallel(B).cuda()   #Convert the model to cuda type
        Ris = torch.nn.DataParallel(Ris).cuda()   #Convert the model to cuda type
        for i in range(sys.Num_User):
            R[i] = torch.nn.DataParallel(R[i]).cuda()   #Convert the model to cuda type
    return B, Ris, R

def Channel(t_data, channel, cuda_gpu):
    num = t_data.shape[0]
    t_data = torch.transpose(t_data, 1, 0)
    hr = channel.real
    hi = channel.imag
    h1 = np.concatenate([hr,-hi],1)
    h2 = np.concatenate([hi,hr],1)
    channel = torch.from_numpy(np.concatenate([h1,h2],0)).float()
    if (cuda_gpu):
        channel = channel.cuda()
    r_data = torch.matmul(channel,t_data)
    return torch.transpose(r_data, 1, 0)

gi_sampler = get_generator_input_sampler()
criterion = nn.BCELoss()
c1 = nn.MSELoss()

def train(sym_index_train, X, Y, sys, SNR_train):
    X_train = X   # training data
    Y_train = Y
    k = sys.k
    J = sys.Num_User
    lr_factor = sys.LR_Factor
    batch_size = sys.Batch_Size
    gpu_list = '0'
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list
    num_total = X.shape[0] 
    total_batch = int(num_total / batch_size)
    gpus = [4,5]   #Which GPUs to use for training, select GPU No. 0 here
    cuda_gpu = torch.cuda.is_available()   #判断GPU是否存在可用
    cuda_gpu = False
    B, Ris, R = ini_weights(sys, cuda_gpu, gpus)
    if sys.load_model == 1:
        B, Ris, R = Load_Model(B, Ris, R, J)
            
    b_optimizer = optim.Adam(B.parameters(), lr=r_learning_rate, betas=optim_betas)
    ris_optimizer = optim.Adam(Ris.parameters(), lr=ris_learning_rate, betas=optim_betas)
    r_optimizer=[]
    for i in range(J):
        r_optimizer.append(optim.Adam(R[i].parameters(), lr=r_learning_rate, betas=optim_betas))
    
    if weight_decay>0:
        b_reg = Regularization(B, weight_decay, p=2)
        ris_reg = Regularization(Ris, weight_decay, p=2)
        r_reg = []
        for i in range(J):
            r_reg.append(Regularization(R[i], weight_decay, p=2))
    else:
        print("no regularization")
    
    R_error = []
    Acc = []
    best_ber = 1
    for epoch in range(sys.Epoch_train):
        error_epoch = 0
        for index in range(total_batch):
            noise_train = torch.from_numpy(np.random.randn(batch_size, sys.Num_User_Antenna*2)*np.sqrt(1/(SNR_train))/np.sqrt(2)).float()
            idx = np.random.randint(num_total,size=batch_size)

            B.zero_grad()
            Ris.zero_grad()
            for i in range(J):
                R[i].zero_grad()
            #x_data = B(torch.from_numpy(af.onehot2symbol(X_train[idx,:])).float(), torch.from_numpy(sym_index_train[idx]).long(), cuda_gpu)
            #target = torch.from_numpy(af.onehot2bit(Y_train[idx,:])).float()

            x_data = B(torch.from_numpy(af.onehot2bit(X_train[idx,:])).float(), torch.from_numpy(sym_index_train[idx]).long(), cuda_gpu)
            target = torch.from_numpy(af.onehot2bit(Y_train[idx,:])).float()
            norm = torch.empty(1,batch_size)
            norm[0,:] = torch.norm(x_data,2,1)
            if(cuda_gpu):
                noise_train = noise_train.cuda()
                x_data = x_data.cuda()
                norm = norm.cuda()
                target = target.cuda()
            #print(torch.sum(norm), torch.sum(norm).shape,(torch.sum(norm)).detach().numpy())
            x_data = x_data/torch.t(norm)#*np.sqrt(batch_size/(torch.sum(norm**2)).detach())
            ri_data = Channel(x_data, sys.Channel_BS2RIS, cuda_gpu)
            ro_data = Ris(ri_data, sys, cuda_gpu)
            y_data = Channel(ro_data, sys.Channel_RIS2User, cuda_gpu) + Channel(x_data, sys.Channel_BS2User, cuda_gpu) + noise_train
           
            #y_data =  Channel(x_data, sys.Channel_BS2User, cuda_gpu) + noise_train
            error_user = 0
            for i in range(J):
                r_error = criterion(R[i](y_data), target[:,i*k:(i+1)*k])#+b_reg(B)+r_reg[i](R[i])+ris_reg(Ris)
                r_error.backward(retain_graph=True)
                r_optimizer[i].step()     # Only optimizes D's parameters; changes based on stored gradients from backward()
                ris_optimizer.step()
                b_optimizer.step()
                error_user = error_user + r_error
            error_epoch = error_epoch + error_user/J
        R_error.append(extract(error_user/J))
            
    #---------------------------------------------------------------------------------------------------------
            
        if epoch % print_interval == 0:
            print("Epoch: %s, Loss: %s, LR: %0.5f" % (epoch, extract(error_user/J), b_optimizer.param_groups[0]['lr']))
            
            for i in range(J):
                r_optimizer[i].param_groups[0]['lr'] /= lr_factor
            ris_optimizer.param_groups[0]['lr'] /= lr_factor
            b_optimizer.param_groups[0]['lr'] /= lr_factor
            SNR_vali = np.array([-5,0,5,10,15,20])
            ber = np.zeros(SNR_vali.shape)
            for i_snr in range(SNR_vali.shape[0]):
                SNR = 10**(SNR_vali[i_snr]/10)/(sys.Rece_Ampli)**2
                sym_index_test, X_test, Y_test = af.generate_transmit_data(sys.M, sys.Num_User, sys.Num_vali, seed=random.randint(0,1000))
                Y_pred, y_receiver = test(sym_index_test, X_test, sys, SNR, B, Ris, R)
                ber[i_snr] = af.BER(X_test, sys, sym_index_test, Y_pred, sys.Num_vali)
                print('The BER at SNR=%d is %0.8f' %(SNR_vali[i_snr], ber[i_snr]))  
            Acc.append(ber[3])
            if ber[1]<best_ber:
                Save_Model(B, Ris, R, J)
                best_ber = ber[1]
            power = torch.mean(torch.sum((Channel(ro_data, sys.Channel_RIS2User, cuda_gpu) + Channel(x_data, sys.Channel_BS2User, cuda_gpu))**2, 1))
            SNR_train = ((2.5*1e-6)/power/(10**(0.5)*1e-7)).detach().cpu().numpy()
            
    #plt.plot(R_error)
    #plt.show()
    return R_error
            
def test(sym_index_test, X, sys, SNR_test, B=None, Ris=None, R=None):
    gpus = [0]   #使用哪几个GPU进行训练，这里选择0号GPU
    cuda_gpu = torch.cuda.is_available()   #判断GPU是否存在可用
    cuda_gpu = False
    k = sys.k
    J = sys.Num_User
    if B==None:
        B, Ris, R = ini_weights(sys, cuda_gpu, gpus)
        B, Ris, R = Load_Model(B, Ris, R, J, 1)
    X_test = X
    num_test = X_test.shape[0]
    noise_test = torch.from_numpy(np.random.randn(num_test, sys.Num_User_Antenna*2)*np.sqrt(1/(2*SNR_test))).float()

    #x_data = B(torch.from_numpy(af.onehot2symbol(X_test)).float(), torch.from_numpy(sym_index_test).long(), cuda_gpu)
    x_data = B(torch.from_numpy(af.onehot2bit(X_test)).float(), torch.from_numpy(sym_index_test).long(), cuda_gpu)
    norm = torch.empty(1,num_test)
    if(cuda_gpu):
        noise_test = noise_test.cuda()
        x_data = x_data.cuda()
        norm = norm.cuda()
    norm[0,:] = torch.norm(x_data,2,1)
    #x_data = x_data/torch.t(norm) #*np.sqrt(sys.Num_User)
    x_data = x_data/torch.t(norm)#*np.sqrt(num_test/(torch.sum(norm**2)).detach())
    sio.savemat('x_data_'+str(sys.Num_RIS_Element)+'.mat', mdict={'x_data': x_data.cpu().detach().numpy()})
    ri_data = Channel(x_data, sys.Channel_BS2RIS, cuda_gpu)
    ro_data = Ris(ri_data, sys, cuda_gpu)
    y_data = Channel(ro_data, sys.Channel_RIS2User, cuda_gpu) + Channel(x_data, sys.Channel_BS2User, cuda_gpu) + noise_test
    #y_data =  Channel(x_data, sys.Channel_BS2User, cuda_gpu) + noise_test

    r_decision = []
    for i in range(J):
        r_decision.append(R[i](y_data))
    r_decision = torch.reshape(torch.stack(r_decision,1),[num_test, k*J])
        
#    x_data = torch.autograd.Variable(x_data)
#    y_data = torch.autograd.Variable(y_data)
            
    pred = r_decision.detach().cpu().numpy()
    y_rate = y_data.detach().cpu().numpy()
    return pred, y_rate
    #d_fake_error = criterion(d_fake_decision, Variable(torch.zeros(batch_size,M)) -torch.from_numpy(Y_train[idx,:]))  # zeros = fake

def Load_Model(B, Ris, R, J, test = 0):
    if test == 0:
        B.load_state_dict(torch.load('channel_&_model/b1_0.pkl'))
        Ris.load_state_dict(torch.load('channel_&_model/ris1_0.pkl'))
        for i in range(J):
            R[i].load_state_dict(torch.load('channel_&_model/r1%d_0.pkl'%(i)))
    elif test == 1:
        B.load_state_dict(torch.load('channel_&_model/b1.pkl'))
        Ris.load_state_dict(torch.load('channel_&_model/ris1.pkl'))
        for i in range(J):
            R[i].load_state_dict(torch.load('channel_&_model/r1%d.pkl'%(i)))
    return B, Ris, R

def Save_Model(B, Ris, R, J):
    torch.save(B.state_dict(), "channel_&_model/b1.pkl") 
    torch.save(Ris.state_dict(), "channel_&_model/ris1.pkl")
    for i in range(J):
        torch.save(R[i].state_dict(), "channel_&_model/r1%d.pkl"%(i)) 
 
class Regularization(torch.nn.Module):
    def __init__(self,model,weight_decay,p=2):
        '''
        :param model 模型
        :param weight_decay:正则化参数
        :param p: 范数计算中的幂指数值，默认求2范数,
                  当p=0为L2正则化,p=1为L1正则化
        '''
        super(Regularization, self).__init__()
        if weight_decay <= 0:
            print("param weight_decay can not <=0")
            exit(0)
        self.model=model
        self.weight_decay=weight_decay
        self.p=p
        self.weight_list=self.get_weight(model)
        self.weight_info(self.weight_list)
    def to(self,device):
        '''
        指定运行模式
        :param device: cude or cpu
        :return:
        '''
        self.device=device
        super().to(device)
        return self
    def forward(self, model):
        self.weight_list=self.get_weight(model)#获得最新的权重
        reg_loss = self.regularization_loss(self.weight_list, self.weight_decay, p=self.p)
        return reg_loss
    def get_weight(self,model):
        '''
        获得模型的权重列表
        :param model:
        :return:
        '''
        weight_list = []
        for name, param in model.named_parameters():
            if 'weight' in name:
                weight = (name, param)
                weight_list.append(weight)
        return weight_list
    def regularization_loss(self,weight_list, weight_decay, p=2):
        '''
        计算张量范数
        :param weight_list:
        :param p: 范数计算中的幂指数值，默认求2范数
        :param weight_decay:
        :return:
        '''
        # weight_decay=Variable(torch.FloatTensor([weight_decay]).to(self.device),requires_grad=True)
        # reg_loss=Variable(torch.FloatTensor([0.]).to(self.device),requires_grad=True)
        # weight_decay=torch.FloatTensor([weight_decay]).to(self.device)
        # reg_loss=torch.FloatTensor([0.]).to(self.device)
        reg_loss=0
        for name, w in weight_list:
            l2_reg = torch.norm(w, p=p)
            reg_loss = reg_loss + l2_reg
 
        reg_loss=weight_decay*reg_loss
        return reg_loss
    def weight_info(self,weight_list):
        '''
        打印权重列表信息
        :param weight_list:
        :return:
        '''
#        print("---------------regularization weight---------------")
#        for name ,w in weight_list:
#            print(name)
#        print("---------------------------------------------------")
#   
#  def forward(self, x0, sys, phase_matrix, trainable):
#        m,n = x0.shape 
#        if trainable == 1:
#            x1 = x0[:,sys.Active_Element]
#            x1 = F.relu(self.map1(x1))
#            x1 = self.bn1(x1)
#            x1 = F.relu(self.map2(x1))
#            x1 = self.bn2(x1)
#            x1 = self.map3(x1)
#            x1 = torch.sum(x1, 0, True)/m
#            p1 = phase_matrix + x1 #[1, 32*2]
#    
#            p1 = torch.reshape(p1, [2,int(n/2)])
#            norm = torch.empty(1, int(n/2))
#            norm[0,:] = torch.norm(p1,2,0)
#            p2 = p1/norm
#            p2 = torch.reshape(p2, [1,n])
#        elif trainable == 0:
#            p2 = phase_matrix
#            
#        x4 = torch.empty(m, n)
#        x4[:,0:int(n/2)] = p2[0,0:int(n/2)]*x0[:,0:int(n/2)] - p2[0,int(n/2):n]*x0[:,int(n/2):n]
#        x4[:,int(n/2):n] = p2[0,0:int(n/2)]*x0[:,int(n/2):n] + p2[0,int(n/2):n]*x0[:,0:int(n/2)]
#        
#        return x4, p2