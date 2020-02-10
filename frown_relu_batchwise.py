import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

import get_bound_for_general_activation_function as get_bound
from certify_mlp import fcNet

from models.mlp import BinarizeActivation, FcNet
from utils.early_stop import EarlyStop

import time

class FcNetBatchwiseOpt(fcNet):
    def __init__(self, input_dimension=28 *28, output_dimension=10, num_layers=5, num_neurons=1024, activation='relu', 
                    batch_size=64):
        super().__init__(input_dimension=input_dimension, output_dimension=output_dimension, num_layers=num_layers, num_neurons=num_neurons, activation=activation)
        self.kl_idx = [None] * (num_layers + 1)
        self.batch_size = batch_size
        # kl_idx is the same lengh and shape as self.kl
        # self.kl_idx[v] indicates whether self.kl[v] need to be optimized
    def clear_intermediate_variables(self):
        # fcNet.clear_intermediate_variables(self)
        super(FcNetBatchwiseOpt, self).clear_intermediate_variables()
        self.kl_idx = [None] * (self.num_layers + 1)
        return 0

    def optimize_kl_neuronwise_for_last_layer_with_gx0_trick_multi_sample(self, eps, p, x, max_iter = 100, 
            print_loss=True, patience=5, acc=1e-2, init='ori'):
        # optimize kl[1], kl[2], ..., kl[num_layers-1] to get the tighter bound for the final output in the case where
        # gx0 trick is applied. After gx0 trick, the number of out units become batch, the output size will be batch, batch.
        # only the diagonal of the final output is meaningful.
        # The first image corresponds to first diagonal element, the second image corresponds to the second diagonal element, so on and so forth 
        # Therefore we don't need to loop over all the neurons for optimization, 
        # we can perform optimization for every image and their corresponding diagonal elements at once. 
        # To use this function, users must make sure that the first dimension of W[-1], b[-1] equals to the first dimension of x
        # namely, we need make sure W[i,:] = W_ori[true label of x[i], :] - W_ori[target label of x[i], :], for 0 <= i < batch
        # b[i] = b_ori[true label of x[i]] - W_ori[target label of x[i]], for 0 <= i < batch
        
        v = self.num_layers
        
        kl = [] # optimization variable
        kl_ori = [] # store original kl
        idx = self.kl_idx[1:v]

        # init optimization variables and save kl_ori
        for k in range(1,v):
            # kl.append(self.kl[k].clone().detach())
            kl_ori.append(self.kl[k].clone().detach())
            kl_ori[k-1].requires_grad = False

            if init == 'ori':
                kl.append(kl_ori[k-1].clone().detach())
            elif init == 'rand':
                kl.append(torch.rand_like(kl_ori[k-1]))
            else:
                raise Exception('%s initialization not supported' % init)
            kl[k-1].requires_grad = True
            

        optimizer = optim.Adam(kl, lr = 1e-1)
        stopper = EarlyStop(patience, acc=acc)
        for i in range(max_iter):
            for k in range(1,v):
                kl[k-1].data.clamp_(min=0, max=1)
                # kl: slope of lower bounding should be between 0 and 1
                self.kl[k] = kl[k-1] * idx[k-1] + kl_ori[k-1] * (1-idx[k-1])
            
            yL,_ = self.compute2sideBound(eps, p, v, x=x)
            loss1 = -torch.diag(yL).mean()
            if stopper.should_stop(loss1.detach().cpu().item()):
                break
            optimizer.zero_grad()
            loss1.backward()
            optimizer.step()
            if print_loss:
                print('step %d yL mean %.5f' % (i+1, -loss1))
        # yL_opti = torch.diag(yL) # yL is of shape batch, batch; yL_opti is of shape batch
        yL_opti = yL

        # yU is actually not needed in this case, we only need yL to judge whehter we need increase or decrease eps in a binary search step
        # we still keep it because we want to compare it with crown
        for k in range(1,v):
            # kl[k-1] = kl_ori[k-1].clone().detach()
            if init == 'ori':
                kl.append(kl_ori[k-1].clone().detach())
            elif init == 'rand':
                kl.append(torch.rand_like(kl_ori[k-1]))
            else:
                raise Exception('%s initialization not supported' % init)
            kl[k-1].requires_grad = True
        optimizer = optim.Adam(kl, lr = 1e-1)
        stopper = EarlyStop(patience, acc=acc)
        for i in range(max_iter):
            for k in range(1,v):
                # self.kl[k] = torch.clamp(kl[k-1], min=0, max=1) * idx[k-1] + kl_ori[k-1] * (1-idx[k-1])
                kl[k-1].data.clamp_(min=0, max=1)
                self.kl[k] = kl[k-1] * idx[k-1] + kl_ori[k-1] * (1-idx[k-1])
            
            _,yU = self.compute2sideBound(eps, p, v, x=x)
            loss2 = torch.diag(yU).mean()
            if stopper.should_stop(loss2.detach().cpu().item()):
                break
            optimizer.zero_grad()
            loss2.backward()
            optimizer.step()
            if print_loss:
                print('step %d yU mean %.5f' % (i+1, loss2))
        # yU_opti = torch.diag(yU) # yU is of shape batch, batch; yU_opti is of shape batch
        yU_opti = yU

        print('Layer %d: yU-yL mean: %.3f' % (v,(yU_opti-yL_opti).mean()), 
            'optimized lines portion:',[round(index.mean().item()*100,1) for index in idx])

        for k in range(1,v):
            # restore original kl
            self.kl[k] = kl_ori[k-1].clone().detach()
        return yL_opti, yU_opti

    def optimize_kl_batchwise(self, v, eps, p, x, num_neurons, max_iter = 100, print_loss=True,
                                patience=5, acc=1e-2, init='ori'):
        # optimize the lower bounding line slopes of h1_l/U,..,h(v-1)_L/U for yv
        # compute a tighter bound of the v-th layer, v range from 1 to num_layers
        # x should be a tensor of size (batch, input_dimesnion)
        # if this is the last layer and gx0 trick is applied, 
        # we should instead use optimize_kl_neuronwise_for_last_layer_with_gx0_trick_multi_sample
        
        batch_size = self.batch_size
        if v==1:
            yL_opti,yU_opti= self.compute2sideBound(eps, p, v, x=x)
        else:
            kl_ori = []
            idx = self.kl_idx[1:v]
            for k in range(1,v):
                kl_ori.append(self.kl[k].clone().detach())
                kl_ori[k-1].requires_grad = False

            num_neuron = self.b[v].shape[0]
            batch = x.shape[0]
            yL_opti = torch.zeros(batch, num_neuron, device=x.device)
            W_v = self.W[v].detach().clone()
            b_v = self.b[v].detach().clone()
        
            num_iters = np.int(np.ceil(num_neuron / batch_size))
            for j in range(num_iters):
                self.W[v] = W_v[j*batch_size:(j+1)*batch_size,:]
                self.b[v] = b_v[j*batch_size:(j+1)*batch_size]

                # init optimization variables
                kl = []
                for k in range(1,v):
                    if init == 'ori':
                        kl.append(kl_ori[k-1].clone().detach())
                    elif init == 'rand':
                        kl.append(torch.rand_like(kl_ori[k-1]))
                    else:
                        raise Exception('%s initialization not supported' % init)
                    kl[k-1].requires_grad = True

                optimizer = optim.Adam(kl, lr = 1e-1)
                stopper = EarlyStop(patience, acc=acc)

                for i in range(max_iter):
                    for k in range(1,v):
                        # self.kl[k] = torch.clamp(kl[k-1], min=0, max=1) * idx[k-1] + kl_ori[k-1] * (1-idx[k-1])
                        kl[k-1].data.clamp_(min=0, max=1)
                        self.kl[k] = kl[k-1] * idx[k-1] + kl_ori[k-1] * (1-idx[k-1])
                    
                    yL,_ = self.compute2sideBound(eps, p, v, x=x) # shape (batch,batch_size)
                    
                    loss1 = -yL
                    # print('loss1 neuron %d/%d step %d \n' % (j+1, num_neuron, i+1), loss1)
                    loss1 = loss1.mean()
                    if stopper.should_stop(loss1.detach().cpu().item()):
                        break
                    optimizer.zero_grad()
                    # pdb.set_trace()
                    loss1.backward()
                    optimizer.step()
                    if print_loss:
                        print('neuron %d/%d step %d yL mean %.5f' % (j+1, num_neuron, i+1, -loss1))
                
                yL_opti[:,j*batch_size:(j+1)*batch_size] = yL
            
            # pdb.set_trace()
            yU_opti = torch.zeros(batch, num_neuron, device=x.device)
            for j in range(num_iters):
                self.W[v] = W_v[j*batch_size:(j+1)*batch_size,:]
                self.b[v] = b_v[j*batch_size:(j+1)*batch_size]

                # init optimization variables
                kl = []
                for k in range(1,v):
                    # kl.append(kl_ori[k-1].clone().detach())
                    if init == 'ori':
                        kl.append(kl_ori[k-1].clone().detach())
                    elif init == 'rand':
                        kl.append(torch.rand_like(kl_ori[k-1]))
                    else:
                        raise Exception('%s initialization not supported' % init)
                    kl[k-1].requires_grad = True

                optimizer = optim.Adam(kl, lr = 1e-1)
                stopper = EarlyStop(patience, acc=acc)

                for i in range(max_iter):
                    for k in range(1,v):
                        # self.kl[k] = torch.clamp(kl[k-1], min=0, max=1) * idx[k-1] + kl_ori[k-1] * (1-idx[k-1])
                        kl[k-1].data.clamp_(min=0, max=1)
                        self.kl[k] = kl[k-1] * idx[k-1] + kl_ori[k-1] * (1-idx[k-1])
                    
                    _,yU = self.compute2sideBound(eps, p, v, x=x) # shape (batch,1)
                    
                    loss2 = yU
                    # print('loss2 neuron %d/%d step %d \n' % (j+1, num_neuron, i+1), loss2)
                    loss2 = loss2.mean()
                    if stopper.should_stop(loss2.detach().cpu().item()):
                        break
                    optimizer.zero_grad()
                    loss2.backward()
                    optimizer.step()
                    if print_loss:
                        print('neuron %d/%d step %d yU mean %.5f' % (j+1, num_neuron, i+1, loss2))
    
                yU_opti[:,j*batch_size:(j+1)*batch_size] = yU
            
            # print('yL,yU \n', yL_opti, yU_opti)
            self.W[v] = W_v.detach()
            self.b[v] = b_v.detach()
        
        if v == 1: 
            print('Layer %d: yU-yL mean: %.3f' % (v,(yU_opti-yL_opti).mean()))
        else:
            print('Layer %d: yU-yL mean: %.3f' % (v,(yU_opti-yL_opti).mean()), 
                'optimized lines portion:',[round(index.mean().item()*100,1) for index in idx])
            # print([index.mean().item() for index in idx])

        self.l[v] = yL_opti.detach()
        self.u[v] = yU_opti.detach()
        self.kl_idx[v] = ((yL_opti<0) * (yU_opti>0)).float().detach()
        kl, bl, ku, bu = get_bound.getConvenientGeneralActivationBound(
                                self.l[v], self.u[v], self.activation, use_constant=False)
        self.kl[v] = kl.detach()
        self.ku[v] = ku.detach()
        self.bl[v] = bl.detach()
        self.bu[v] = bu.detach()

        for k in range(1,v):
            self.kl[k] = kl_ori[k-1].clone().detach()
        return yL_opti, yU_opti

    def optimize_last_layer_bound(self,  eps, p, x, max_iter, print_loss, gx0_trick, targeted, save_dir=None):
        # self.num_neurons is a list of size self.num_layers - 1
        for v in range(1, self.num_layers): # v from 1 to self.num_layers-1
            yL,yU = self.optimize_kl_batchwise( v, eps, p, x, self.num_neurons[v-1], max_iter = max_iter, 
                    print_loss = print_loss)
        # yL,yU = self.optimize_kl_neuronwise( self.num_layers, eps, p, x, self.output_dimension, max_iter = max_iter, 
        #             print_loss = print_loss, last_layer_gx0_trick_targeted=gx0_trick and targeted, save_dir=save_dir)
        if gx0_trick:
            yL,yU = self.optimize_kl_neuronwise_for_last_layer_with_gx0_trick_multi_sample(eps, p, x, 
                            max_iter = max_iter, print_loss=print_loss)
        else:
            yL,yU = self.optimize_kl_batchwise( self.num_layers, eps, p, x, self.output_dimension, 
                            max_iter = max_iter, print_loss = print_loss)
        return yL,yU

    def getLastLayerBound(self,  eps, p, x, max_iter=100, print_loss=False, gx0_trick=True, targeted=True, clearIntermediateVariables=True):
        # getLastLayerBound is the same as optimize_last_layer_bound
        # we rename it because in the binary search function, we call this function getLastLayerBound
        yL, yU =  self.optimize_last_layer_bound(eps, p, x, max_iter, print_loss, gx0_trick, targeted)
        if clearIntermediateVariables:
            self.clear_intermediate_variables()
        return yL,yU

