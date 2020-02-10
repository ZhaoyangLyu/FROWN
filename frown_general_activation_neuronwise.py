import torch
import torch.nn as nn
import torch.optim as optim

from get_bound_for_general_activation_function import get_tangent_line_short, getConvenientGeneralActivationBound
from certify_mlp import fcNet

from models.mlp import BinarizeActivation, FcNet
from utils.early_stop import EarlyStop

import time

class FcNetGeneralActivationNeuronwiseOpt(fcNet):
    def __init__(self, input_dimension=28 *28, output_dimension=10, num_layers=5, num_neurons=1024, activation='tanh'):
        super().__init__(input_dimension=input_dimension, output_dimension=output_dimension, num_layers=num_layers, num_neurons=num_neurons, activation=activation)
        
        self.sl = [None] * (num_layers+1)
        self.valid_l = [None] * (num_layers+1)
        self.dl_upper = [None] * (num_layers+1)
        self.dl_lower = [None] * (num_layers+1)
        # sl and valid_l are the same lengh and shape as kl,bl.
        # kl[valid_l], bl[valid_l] is the tangent line at point sl[valid_l], 
        # sl[valid_l] is the tangent point chosen by crown. 
        # kl[valid_l], bl[valid_l] are those bounding lines that can be optimized 
        # the optimization variable dl (kl and bl are tangent line at dl) range from dl_lower to dl_upper
        self.su = [None] * (num_layers+1)
        self.valid_u = [None] * (num_layers+1)
        self.du_upper = [None] * (num_layers+1)
        self.du_lower = [None] * (num_layers+1)
        # su and valid_u are the same lengh and shape as ku,bu.
        # ku[valid_u], bu[valid_u] is the tangent line at point su[valid_u]
        # sl[valid_l] is the tangent point chosen by crown. 
        # ku[valid_u], bu[valid_u] are those bounding lines that can be optimized 
        # the optimization variable du (ku and bu are tangent line at du) range from du_lower to du_upper
        
    def clear_intermediate_variables(self):
        super(FcNetGeneralActivationNeuronwiseOpt, self).clear_intermediate_variables()
        self.sl = [None] * (self.num_layers+1)
        self.valid_l = [None] * (self.num_layers+1)
        self.dl_upper = [None] * (self.num_layers+1)
        self.dl_lower = [None] * (self.num_layers+1)
        
        self.su = [None] * (self.num_layers+1)
        self.valid_u = [None] * (self.num_layers+1)
        self.du_upper = [None] * (self.num_layers+1)
        self.du_lower = [None] * (self.num_layers+1)
        return 0

    def optimize_k_neuronwise_for_last_layer_with_gx0_trick_multi_sample(self, eps, p, x, max_iter = 100, 
            print_loss=True, patience=5, acc=1e-2, lr=1e-1, init='middle'):
        # optimize kl/bl[1], kl/bl[2], ..., kl/bl[num_layers-1] to get the tighter bound for the final output in the case where
        # gx0 trick is applied. After gx0 trick, the number of out units become batch, the output size will be batch, batch.
        # only the diagonal of the final output is meaningful.
        # The first image corresponds to first diagonal element, the second image corresponds to the second diagonal element, so on and so forth 
        # Therefore we don't need to loop over all the neurons for optimization, 
        # we can perform optimization for every image and their corresponding diagonal elements at once. 
        # To use this function, users must make sure that the first dimension of W[-1], b[-1] equals to the first dimension of x
        # namely, we need make sure W[i,:] = W_ori[true label of x[i], :] - W_ori[target label of x[i], :], for 0 <= i < batch
        # b[i] = b_ori[true label of x[i]] - W_ori[target label of x[i]], for 0 <= i < batch
        
        v = self.num_layers

        kl_ori = []
        bl_ori = []
        ku_ori = []
        bu_ori = []
        dl = []
        du = []
        # init optimization variables and save kl_ori
        for k in range(1,v):
            # kl.append(self.kl[k].clone().detach())
            kl_ori.append(self.kl[k].clone().detach())
            bl_ori.append(self.bl[k].clone().detach())
            ku_ori.append(self.ku[k].clone().detach())
            bu_ori.append(self.bu[k].clone().detach())
            kl_ori[k-1].requires_grad = False
            bl_ori[k-1].requires_grad = False
            ku_ori[k-1].requires_grad = False
            bu_ori[k-1].requires_grad = False
            if init == 'ori':
                dl.append(self.sl[k].clone().detach())
                dl[k-1].requires_grad = True
                du.append(self.su[k].clone().detach())
                du[k-1].requires_grad = True
            elif init == 'middle':
                dl.append( ((self.dl_lower[k]+self.dl_upper[k])/2).detach() )
                dl[k-1].requires_grad = True
                # du.append(self.su[k].clone().detach())
                du.append( ((self.du_lower[k]+self.du_upper[k])/2).detach() )
                du[k-1].requires_grad = True
            elif init == 'rand':
                dl.append( ( (self.dl_upper[k]-self.dl_lower[k])*torch.rand_like(self.dl_lower[k])+self.dl_lower[k] ).detach() )
                dl[k-1].requires_grad = True
                du.append( ( (self.du_upper[k]-self.du_lower[k])*torch.rand_like(self.du_lower[k])+self.du_lower[k] ).detach() )
                du[k-1].requires_grad = True
            else:
                raise Exception('%s initialization not supported' % init)
            

        optimizer = optim.Adam(dl+du, lr = lr)
        stopper = EarlyStop(patience, acc=acc)
        for i in range(max_iter):
            for k in range(1,v):
                dl[k-1].data = torch.max(self.dl_lower[k], torch.min(dl[k-1], self.dl_upper[k]))
                kl_temp, bl_temp = get_tangent_line_short(dl[k-1], self.activation)
                self.kl[k] = self.valid_l[k] * kl_temp + (1-self.valid_l[k])*kl_ori[k-1]
                self.bl[k] = self.valid_l[k] * bl_temp + (1-self.valid_l[k])*bl_ori[k-1]

                du[k-1].data = torch.max(self.du_lower[k], torch.min(du[k-1], self.du_upper[k]))
                ku_temp, bu_temp = get_tangent_line_short(du[k-1], self.activation)
                self.ku[k] = self.valid_u[k] * ku_temp + (1-self.valid_u[k])*ku_ori[k-1]
                self.bu[k] = self.valid_u[k] * bu_temp + (1-self.valid_u[k])*bu_ori[k-1]
            
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

        dl = []
        du = []
        for k in range(1,v):
            if init == 'ori':
                dl.append(self.sl[k].clone().detach())
                dl[k-1].requires_grad = True
                du.append(self.su[k].clone().detach())
                du[k-1].requires_grad = True
            elif init == 'middle':
                dl.append( ((self.dl_lower[k]+self.dl_upper[k])/2).detach() )
                dl[k-1].requires_grad = True
                du.append( ((self.du_lower[k]+self.du_upper[k])/2).detach() )
                du[k-1].requires_grad = True
            elif init == 'rand':
                dl.append( ( (self.dl_upper[k]-self.dl_lower[k])*torch.rand_like(self.dl_lower[k])+self.dl_lower[k] ).detach() )
                dl[k-1].requires_grad = True
                du.append( ( (self.du_upper[k]-self.du_lower[k])*torch.rand_like(self.du_lower[k])+self.du_lower[k] ).detach() )
                du[k-1].requires_grad = True
            else:
                raise Exception('%s initialization not supported' % init)

        optimizer = optim.Adam(dl+du, lr = lr)
        stopper = EarlyStop(patience, acc=acc)
        for i in range(max_iter):
            for k in range(1,v):
                dl[k-1].data = torch.max(self.dl_lower[k], torch.min(dl[k-1], self.dl_upper[k]))
                kl_temp, bl_temp = get_tangent_line_short(dl[k-1], self.activation)
                self.kl[k] = self.valid_l[k] * kl_temp + (1-self.valid_l[k])*kl_ori[k-1]
                self.bl[k] = self.valid_l[k] * bl_temp + (1-self.valid_l[k])*bl_ori[k-1]

                du[k-1].data = torch.max(self.du_lower[k], torch.min(du[k-1], self.du_upper[k]))
                ku_temp, bu_temp = get_tangent_line_short(du[k-1], self.activation)
                self.ku[k] = self.valid_u[k] * ku_temp + (1-self.valid_u[k])*ku_ori[k-1]
                self.bu[k] = self.valid_u[k] * bu_temp + (1-self.valid_u[k])*bu_ori[k-1]
            
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

        print('Layer %d: yU-yL mean: %.3f' % (v,(yU_opti-yL_opti).mean())) 
        print('optimized lower lines portion:',[round(index.mean().item()*100,1) for index in self.valid_l[1:v] ])
        print('optimized upper lines portion:',[round(index.mean().item()*100,1) for index in self.valid_u[1:v] ])

        self.l[v] = yL_opti.detach()
        self.u[v] = yU_opti.detach()

        for k in range(1,v):
            self.kl[k] = kl_ori[k-1].detach()
            self.bl[k] = bl_ori[k-1].detach()
            self.ku[k] = ku_ori[k-1].detach()
            self.bu[k] = bu_ori[k-1].detach()
        return yL_opti, yU_opti

    def optimize_k_neuronwise(self, v, eps, p, x, num_neurons, max_iter = 100, print_loss=True,
                                patience=5, acc=1e-2, lr=1e-1, init='middle'):
        # optimize the lower/upper bounding lines of h1_l/U,..,h(v-1)_L/U for yv
        # compute a tighter bound of the v-th layer, v range from 1 to num_layers
        # x should be a tensor of size (batch, input_dimesnion)
        # if this is the last layer and gx0 trick is applied, 
        # we should instead use optimize_k_neuronwise_for_last_layer_with_gx0_trick_multi_sample
        
        if v==1:
            yL_opti,yU_opti= self.compute2sideBound(eps, p, v, x=x)
        else:
            kl_ori = []
            bl_ori = []
            ku_ori = []
            bu_ori = []
            for k in range(1,v):
                kl_ori.append(self.kl[k].clone().detach())
                bl_ori.append(self.bl[k].clone().detach())
                ku_ori.append(self.ku[k].clone().detach())
                bu_ori.append(self.bu[k].clone().detach())
                kl_ori[k-1].requires_grad = False
                bl_ori[k-1].requires_grad = False
                ku_ori[k-1].requires_grad = False
                bu_ori[k-1].requires_grad = False

            num_neuron = self.b[v].shape[0]
            batch = x.shape[0]
            yL_opti = torch.zeros(batch, num_neuron, device=x.device)
            W_v = self.W[v].detach().clone()
            b_v = self.b[v].detach().clone()

            for j in range(num_neuron): # for every neuron in this layer, we optimize over it for all images in this batch at once 
                self.W[v] = W_v[j:j+1,:]
                self.b[v] = b_v[j:j+1]

                # init optimization variables
                dl = []
                du = []
                for k in range(1,v):
                    if init == 'ori':
                        dl.append(self.sl[k].clone().detach())
                        dl[k-1].requires_grad = True
                        du.append(self.su[k].clone().detach())
                        du[k-1].requires_grad = True
                    elif init == 'middle':
                        dl.append( ((self.dl_lower[k]+self.dl_upper[k])/2).detach() )
                        dl[k-1].requires_grad = True
                        du.append( ((self.du_lower[k]+self.du_upper[k])/2).detach() )
                        du[k-1].requires_grad = True
                    elif init == 'rand':
                        dl.append( ( (self.dl_upper[k]-self.dl_lower[k])*torch.rand_like(self.dl_lower[k])+self.dl_lower[k] ).detach() )
                        dl[k-1].requires_grad = True
                        du.append( ( (self.du_upper[k]-self.du_lower[k])*torch.rand_like(self.du_lower[k])+self.du_lower[k] ).detach() )
                        du[k-1].requires_grad = True
                    else:
                        raise Exception('%s initialization not supported' % init)

                optimizer = optim.Adam(dl+du, lr = lr)
                stopper = EarlyStop(patience, acc=acc)

                for i in range(max_iter):
                    for k in range(1,v):
                        dl[k-1].data = torch.max(self.dl_lower[k], torch.min(dl[k-1], self.dl_upper[k]))
                        kl_temp, bl_temp = get_tangent_line_short(dl[k-1], self.activation)
                        self.kl[k] = self.valid_l[k] * kl_temp + (1-self.valid_l[k])*kl_ori[k-1]
                        self.bl[k] = self.valid_l[k] * bl_temp + (1-self.valid_l[k])*bl_ori[k-1]

                        du[k-1].data = torch.max(self.du_lower[k], torch.min(du[k-1], self.du_upper[k]))
                        ku_temp, bu_temp = get_tangent_line_short(du[k-1], self.activation)
                        self.ku[k] = self.valid_u[k] * ku_temp + (1-self.valid_u[k])*ku_ori[k-1]
                        self.bu[k] = self.valid_u[k] * bu_temp + (1-self.valid_u[k])*bu_ori[k-1]
                        
                    yL,_ = self.compute2sideBound(eps, p, v, x=x) # shape (batch,1)
                    
                    loss1 = -yL
                    loss1 = loss1.mean()
                    if stopper.should_stop(loss1.detach().cpu().item()):
                        break
                    optimizer.zero_grad()
                    # pdb.set_trace()
                    loss1.backward()
                    optimizer.step()
                    if print_loss:
                        print('neuron %d/%d step %d yL mean %.5f' % (j+1, num_neuron, i+1, -loss1))
                
                yL_opti[:,j] = yL.squeeze(1)
            

            yU_opti = torch.zeros(batch, num_neuron, device=x.device)
            for j in range(num_neuron):
                self.W[v] = W_v[j:j+1,:]
                self.b[v] = b_v[j:j+1]

                # init optimization variables
                dl = []
                du = []
                for k in range(1,v):
                    if init == 'ori':
                        dl.append(self.sl[k].clone().detach())
                        dl[k-1].requires_grad = True
                        du.append(self.su[k].clone().detach())
                        du[k-1].requires_grad = True
                    elif init == 'middle':
                        dl.append( ((self.dl_lower[k]+self.dl_upper[k])/2).detach() )
                        dl[k-1].requires_grad = True
                        du.append( ((self.du_lower[k]+self.du_upper[k])/2).detach() )
                        du[k-1].requires_grad = True
                    elif init == 'rand':
                        dl.append( ( (self.dl_upper[k]-self.dl_lower[k])*torch.rand_like(self.dl_lower[k])+self.dl_lower[k] ).detach() )
                        dl[k-1].requires_grad = True
                        du.append( ( (self.du_upper[k]-self.du_lower[k])*torch.rand_like(self.du_lower[k])+self.du_lower[k] ).detach() )
                        du[k-1].requires_grad = True
                    else:
                        raise Exception('%s initialization not supported' % init)

                optimizer = optim.Adam(dl+du, lr = lr)
                stopper = EarlyStop(patience, acc=acc)

                for i in range(max_iter):
                    for k in range(1,v):
                        dl[k-1].data = torch.max(self.dl_lower[k], torch.min(dl[k-1], self.dl_upper[k]))
                        kl_temp, bl_temp = get_tangent_line_short(dl[k-1], self.activation)
                        self.kl[k] = self.valid_l[k] * kl_temp + (1-self.valid_l[k])*kl_ori[k-1]
                        self.bl[k] = self.valid_l[k] * bl_temp + (1-self.valid_l[k])*bl_ori[k-1]

                        du[k-1].data = torch.max(self.du_lower[k], torch.min(du[k-1], self.du_upper[k]))
                        ku_temp, bu_temp = get_tangent_line_short(du[k-1], self.activation)
                        self.ku[k] = self.valid_u[k] * ku_temp + (1-self.valid_u[k])*ku_ori[k-1]
                        self.bu[k] = self.valid_u[k] * bu_temp + (1-self.valid_u[k])*bu_ori[k-1]

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
                        
                yU_opti[:,j] = yU.squeeze(1)
            
            
            self.W[v] = W_v.detach()
            self.b[v] = b_v.detach()
        
        if v == 1: 
            print('Layer %d: yU-yL mean: %.3f' % (v,(yU_opti-yL_opti).mean()))
        else:
            print('Layer %d: yU-yL mean: %.3f' % (v,(yU_opti-yL_opti).mean())) 
            print('optimized lower lines portion:',[round(index.mean().item()*100,1) for index in self.valid_l[1:v] ])
            print('optimized upper lines portion:',[round(index.mean().item()*100,1) for index in self.valid_u[1:v] ])
            # print([index.mean().item() for index in idx])

        self.l[v] = yL_opti.detach()
        self.u[v] = yU_opti.detach()
        with torch.no_grad():
            kl, bl, ku, bu, sl, sl_valid, su, su_valid = getConvenientGeneralActivationBound(
                                    self.l[v], self.u[v], self.activation, use_constant=False,
                                    remain_tangent_line_info=True)
        self.kl[v] = kl.detach()
        self.ku[v] = ku.detach()
        self.bl[v] = bl.detach()
        self.bu[v] = bu.detach()
        self.sl[v] = sl.detach()
        self.valid_l[v] = sl_valid.detach()
        self.su[v] = su.detach()
        self.valid_u[v] = su_valid.detach()

        idx = ((self.l[v] < 0) * (self.u[v] > 0)).detach().float()

        self.dl_lower[v] = self.l[v].detach().clone()
        self.dl_upper[v] = (idx * self.sl[v] + (1-idx) * self.u[v]).detach()

        self.du_lower[v] = (idx * self.su[v] + (1-idx) * self.l[v]).detach()
        self.du_upper[v] = self.u[v].detach().clone()


        for k in range(1,v):
            self.kl[k] = kl_ori[k-1].detach()
            self.bl[k] = bl_ori[k-1].detach()
            self.ku[k] = ku_ori[k-1].detach()
            self.bu[k] = bu_ori[k-1].detach()
        return yL_opti, yU_opti

    def optimize_last_layer_bound(self,  eps, p, x, max_iter, print_loss, gx0_trick, targeted, save_dir=None):
        # self.num_neurons is a list of size self.num_layers - 1
        for v in range(1, self.num_layers): # v from 1 to self.num_layers-1
            yL,yU = self.optimize_k_neuronwise( v, eps, p, x, self.num_neurons[v-1], max_iter = max_iter, 
                    print_loss = print_loss)
        # yL,yU = self.optimize_kl_neuronwise( self.num_layers, eps, p, x, self.output_dimension, max_iter = max_iter, 
        #             print_loss = print_loss, last_layer_gx0_trick_targeted=gx0_trick and targeted, save_dir=save_dir)
        if gx0_trick:
            yL,yU = self.optimize_k_neuronwise_for_last_layer_with_gx0_trick_multi_sample(eps, p, x, 
                            max_iter = max_iter, print_loss=print_loss)
        else:
            yL,yU = self.optimize_k_neuronwise( self.num_layers, eps, p, x, self.output_dimension, 
                            max_iter = max_iter, print_loss = print_loss)
        return yL,yU

    def getLastLayerBound(self,  eps, p, x, max_iter=100, print_loss=False, gx0_trick=True, targeted=True, clearIntermediateVariables=True):
        # getLastLayerBound is the same as optimize_last_layer_bound
        # we rename it because in the binary search function, we call this function getLastLayerBound
        yL, yU =  self.optimize_last_layer_bound(eps, p, x, max_iter, print_loss, gx0_trick, targeted)
        if clearIntermediateVariables:
            self.clear_intermediate_variables()
        return yL,yU

