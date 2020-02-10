#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 08:36:29 2018

@author: root
"""

import torch
import torch.nn as nn

import get_bound_for_general_activation_function as get_bound
from models.mlp import BinarizeActivation


class Atan(nn.Module):
    #arctan activation function
    def forward(self, input):
        return torch.atan(input)  
      
class fcNet(nn.Module):
    def __init__(self, input_dimension=28*28, 
                 output_dimension = 10, num_layers = 5,
                 num_neurons = 1024, activation='tanh'):
        # activation could be one of 'tanh', 'atan', 'sigmoid', 'relu', 'ba'
        # num_neurons cound be a single number to specify the number of neurons in every hidden layer
        # num_neurons cound also be a list of numbers of length (num_layers-1),
        # which specifies the number of neurons in each hidden layer
        
        super(fcNet, self).__init__()
        
        self.net = None
        # this net is a FcNet class object. It is used to compute preactivations
        
        self.num_layers = num_layers # number of layers 

        # m: num of layers
        # N: num of samples. batch size
        # n_k: number of neurons in k_th layer
        # num_layers: num of layers (num_hidden_layers + output_layer), equals to m

        self.x = None # data attached to the classifier
        # x is of size (N,self.input_dimension)

        self.l = [None]*(num_layers+1) #l[0] has no use, l[k] is of size (N,n_k), k from 1 to m
        # l[k] is the lower bound of the k-th layer output before relu 
        self.u = [None]*(num_layers+1) #u[0] has no use, u[k] is of size (N,n_k), k from 1 to m
        # u[k] is the upper bound of the k-th layer output before relu

        self.kl = [None]*(num_layers+1)
        self.ku = [None]*(num_layers+1)
        self.bl = [None]*(num_layers+1)
        self.bu = [None]*(num_layers+1)
        # k[0] and b[0] has no use, k[m] and b[m] has no use either
        # k[v] and b[v] v range from 1 to m-1

        self.Au = [None] * (self.num_layers + 1)
        self.Bu = [None] * (self.num_layers + 1)
        self.Al = [None] * (self.num_layers + 1)
        self.Bl = [None] * (self.num_layers + 1)
        # A[k] and B[k] are the coefficients of the linear bounds of the preactivations in the k-th layer
        # namely, Al[k] * x + Bl[k] <= preactivation[k](x) <= Au[k] * x + Bu[k], k from 1 to num_of_layers   
        
        self.W = None # W[k] is the k-th layer weights, k from 1 to num_layers,m
        # W[0] has no use
        self.b = None # b[k] is the k-th layer bias, k from 1 to num_layers,m
        # b[0] has no use
        
        
        if not type(num_neurons) == list:
            num_neurons = [num_neurons] * (num_layers-1)
        self.num_neurons = num_neurons #number of neurons in each layer
        layers = []
        self.input_dimension = input_dimension 
        self.output_dimension = output_dimension
        
        self.activation = activation
        if activation == 'tanh':
            self.activation_function = nn.Tanh()
        elif activation == 'atan':
            self.activation_function = Atan()
        elif activation == 'sigmoid':
            self.activation_function = nn.Sigmoid()
        elif activation == 'ba':
            self.activation_function = BinarizeActivation()
        elif (activation == 'relu' or activation == 'relu_adaptive'):
            self.activation_function = nn.ReLU()
        else:
            raise Exception(activation+' activation function is not supported')
            
        if not num_layers == 1:
            layers.append(nn.Linear(input_dimension, num_neurons[0]))
            layers.append(self.activation_function)
            
            for k in range(num_layers-2):
                layers.append(nn.Linear(num_neurons[k], num_neurons[k+1]))
                layers.append(self.activation_function)
            
            layers.append(nn.Linear(num_neurons[-1], output_dimension))
        else:
            layers.append(nn.Linear(input_dimension, output_dimension))
        
        self.model = nn.Sequential(*layers)
    
    def clear_intermediate_variables(self):
        self.l = [None]*(self.num_layers+1) 
        self.u = [None]*(self.num_layers+1) 

        self.kl = [None]*(self.num_layers+1)
        self.ku = [None]*(self.num_layers+1)
        self.bl = [None]*(self.num_layers+1)
        self.bu = [None]*(self.num_layers+1)

        self.Au = [None] * (self.num_layers + 1)
        self.Bu = [None] * (self.num_layers + 1)
        self.Al = [None] * (self.num_layers + 1)
        self.Bl = [None] * (self.num_layers + 1)

        return 0

    def save_intermediate_variables(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        torch.save({'l':self.l, 'u':self.u, 'kl':self.kl, 'ku':self.ku, 'bl':self.bl, 'bu':self.bu,
                    'Au':self.Au, 'Bu':self.Bu, 'Al':self.Al, 'Bl':self.Bl}, 
                    save_dir + 'bounding_lines.ckpt')
        print('Saved intermediate variables to ' + save_dir + 'bounding_lines.ckpt')
        return 0
        
    def get_kth_layer_output(self, k, x=None):
        #k range from 1 to self.num_layers
        #get the pre-relu output of the k-th layer
        if x is None:
            x = self.x
        else:
            if torch.numel(x) == (x.shape[0] * self.input_dimension):
                x = x.view(-1, self.input_dimension)
            else:
                raise Exception('The input dimension must be %d' 
                                % self.input_dimension)
        seq = self.model[0: (2*(k-1)+1)]
        return seq(x)
        
    def forward(self, x):
        # x is of size N*C*H*W
        if torch.numel(x) == (x.shape[0] * self.input_dimension):
            x = x.view(-1, self.input_dimension)
            x = self.model(x)
            return x
        else:
            raise Exception('The input dimension must be %d' % self.input_dimension)
        
    def attachData(self,x):
        # x is of size N*C*H*W
        if torch.numel(x) == (x.shape[0] * self.input_dimension):
        #if x.shape[1]*x.shape[2]*x.shape[3] == self.input_dimension:
            x = x.view(-1, self.input_dimension)
            self.x = x
        else:
            raise Exception('The input dimension must be %d' % self.input_dimension)
    
    def extractWeight(self, clear_original_model=True):
        model = self.model
        self.W = [0] * (self.num_layers+1)
        self.b = [0] * (self.num_layers+1)
        for i in range(self.num_layers):
            self.W[i+1] = model[2*i].weight.data
            self.b[i+1] = model[2*i].bias.data
        
        if clear_original_model:
            self.model = None
        return 0
        
    def compute2sideBound(self, eps, p, m, x = None):
        # compute bound for the m-th layer, m range from 1 to num_layers
        # assume bounding lines of previous layers have already been computed
        # eps could be a real number, or a tensor of size N
        # p is a real number
        # m is an integer

        n_m = self.W[m].shape[0]
        if x is None:
            x = self.x
        N = x.shape[0] #number of images, batch size
        if type(eps) == torch.Tensor:
            eps = eps.to(x.device)
            
        yU = torch.zeros(N,n_m, device = x.device)#(N,n_m)
        yL = torch.zeros(N,n_m, device = x.device)#(N,n_m)
        
        if m == 1:
            A = self.W[1].unsqueeze(0).expand(N,-1,-1)
            Ou = self.W[1].unsqueeze(0).expand(N,-1,-1)
    
        else:
            for k in range(m-1,-1,-1): #k from m-1 to 0
                # case k = m is handled seperately below, only need to add b[m] for k = m
                if k == m-1:
                    
                    #alpha is of shape (N, n_m, n_k), k=m-1
                    alpha_l = self.kl[k].unsqueeze(1).expand(-1, n_m, -1)
                    alpha_u = self.ku[k].unsqueeze(1).expand(-1, n_m, -1)
                    
                    I = (self.W[k+1] >= 0) #shape[n_m, n_(m-1)]
                    I = I.unsqueeze(0).expand(N, -1, -1).float()
                    lamida = I*alpha_u + (1-I)*alpha_l
                    omiga = I*alpha_l + (1-I)*alpha_u
                    

                    A = self.W[k+1] * lamida #this is A(m-1)
                    Ou = self.W[k+1] * omiga #this is Ou(m-1)
                    # W[k+1] (n_m, n_(m-1)), lamida (N, n_m, n_(m-1)), A[m-1] (N n_m n_(m-1))
                    
                    AW = self.W[k+1].unsqueeze(0).expand(N,-1,-1) #this is AW(m)
                    OW = self.W[k+1].unsqueeze(0).expand(N,-1,-1) #this is OW(m)
                    
                    I_AW = (AW>=0).float()
                    I_OW = (OW>=0).float()
                    #A[m] = identity
                    #AW[m] is of shape[N, n_m, n_(m-1)]
                elif k == 0:
                    A = torch.matmul(A,self.W[1])
                    Ou = torch.matmul(Ou,self.W[1])
                else:
                    
                    alpha_l = self.kl[k].unsqueeze(1).expand(-1, n_m, -1)
                    alpha_u = self.ku[k].unsqueeze(1).expand(-1, n_m, -1)
                    AW = torch.matmul(A, self.W[k+1]) #AW(k+1) = A(k+1) W(k+1)
                    I_AW = (AW>=0).float()
                    lamida = I_AW*alpha_u + (1-I_AW)*alpha_l #lamida(k)
                    A =  AW * lamida #A(k) = AW(k+1) * lamida(k)
                    
                    OW = torch.matmul(Ou,self.W[k+1]) #OW(k+1) = Ou(k+1) W(k+1)
                    I_OW = (OW>=0).float()
                    omiga = I_OW*alpha_l + (1-I_OW)*alpha_u #omiga(k)
                    Ou = OW * omiga #Ou(k) = OW(k+1) * omiga(k)
                 
                if not (k==0):
                    beta_l = self.bl[k].unsqueeze(1).expand(-1, n_m, -1)
                    beta_u = self.bu[k].unsqueeze(1).expand(-1, n_m, -1)
                    
                    H = AW*(I_AW*beta_u + (1-I_AW)*beta_l) #H(k) = AW(k+1) * (I_AW*beta_u(k) + (1-I_AW)*beta_l(k))
                    T = OW*(I_OW*beta_l + (1-I_OW)*beta_u) #T(k) = OW(k+1) * (I_OW*beta_l(k) + (1-I_OW)*beta_u(k))
                   
                   
                    yU = yU + torch.matmul(A, self.b[k]) 
                    yU = yU + H.sum(dim=2)
                    
                    yL = yL + torch.matmul(Ou, self.b[k]) 
                    yL = yL + T.sum(dim=2)
        #After finishing the above loop, A is actually A[0]
        yU = yU + self.b[m]
        self.Bu[m] = yU
        self.Au[m] = A
        yU = yU + torch.matmul(A,x.unsqueeze(2)).squeeze(2)

        yL = yL + self.b[m]
        self.Bl[m] = yL
        self.Al[m] = Ou
        yL = yL + torch.matmul(Ou,x.unsqueeze(2)).squeeze(2)
        
        if p == 1:
            q = float('inf')
        elif (p == 'inf' or p == float('inf')):
            q = 1
        else:
            q = p / (p-1)
        
        if type(eps) == torch.Tensor:
            #eps is a tensor of size N
            yU = yU + eps.unsqueeze(1).expand(-1,
                        n_m) *torch.norm(A,p=q,dim=2)
            yL = yL - eps.unsqueeze(1).expand(-1,
                        n_m)*torch.norm(Ou,p=q,dim=2)

        else:
            yU = yU + eps*torch.norm(A,p=q,dim=2)
            yL = yL - eps*torch.norm(Ou,p=q,dim=2)
             
        return yL,yU
    
    def getLastLayerBound(self, eps, p, x = None, clearIntermediateVariables=False):
        #eps could be a real number, or a tensor of size N
        if self.x is None and x is None:
            raise Exception('You must first attach data to the net or feed data to this function')
        if self.W is None or self.b is None:
            self.extractWeight()
        
        if x is None:
            x = self.x

        for k in range(1,self.num_layers+1):
            # k from 1 to self.num_layers
            yL,yU = self.compute2sideBound(eps, p, k, x=x)
            self.l[k] = yL.detach()
            self.u[k] = yU.detach()
            print('yU-yL', (yU-yL).mean())
            #in this loop, self.u, l
            if k<self.num_layers:
                kl, bl, ku, bu = get_bound.getConvenientGeneralActivationBound(
                                self.l[k], self.u[k], self.activation, use_constant=False)
                self.kl[k] = kl.detach()
                self.ku[k] = ku.detach()
                self.bl[k] = bl.detach()
                self.bu[k] = bu.detach()
        if clearIntermediateVariables:
            self.clear_intermediate_variables()
            # self.l[k] = None
            # self.u[k] = None
            #clear l[k] and u[k] to release memory
        return yL, yU


