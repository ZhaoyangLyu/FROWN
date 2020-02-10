#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 09:11:15 2018

@author: root
"""

import torch
import matplotlib.pyplot as plt
import pdb

def d_tanh(x):
    #the derivative of tanh
    return 1- (torch.tanh(x))**2

def d_atan(x):
    return 1/(1+x**2)

def d_sigmoid(x):
    sx = torch.sigmoid(x)
    return sx*(1-sx)

def get_tangent_line(s, func, d_func):
    # compute the tangent line of func at point s
    k = d_func(s)
    b = func(s) - k*s
    return k,b 

Activation = {'tanh':[torch.tanh, d_tanh],
              'atan':[torch.atan, d_atan],
              'sigmoid':[torch.sigmoid, d_sigmoid],
              'ba':[torch.sign, 0],
              'relu':[torch.relu, 0],
              'relu_adaptive':[torch.relu, 0]}


def get_tangent_line_short(s, act):
    # compute the tangent line of func at point s
    func = Activation[act][0]
    d_func = Activation[act][1]
    k = d_func(s)
    b = func(s) - k*s
    return k,b 

def get_bound_for_relu(l, u, adaptive=False):
    device = l.device
    ku = torch.zeros(u.shape, device = device)
    bu = torch.zeros(u.shape, device = device)
    kl = torch.zeros(l.shape, device = device)
    bl = torch.zeros(l.shape, device = device)

    # case u<=0, the 0 initialization already satisfy this case

    # case l>=0
    idx = l>=0
    kl[idx] = 1
    ku[idx] = 1
    # bl and kl is 0

    # case l<0 and u>0
    idx = (l<0) * (u>0)

    k = (u / (u-l))[idx]
    # k u + b = u -> b = (1-k) * u
    b = (1-k) * u[idx]

    
    ku[idx] = k
    bu[idx] = b
    # bl already 0
    # kl should be between 0 and 1
    if not adaptive: # parallel to the upper line
        kl[idx] = k
    else:
        idx = (l<0) * (u>0) * (u.abs()>=l.abs())
        kl[idx] = 1
        idx = (l<0) * (u>0) * (u.abs()<l.abs())
        kl[idx] = 0
    return kl, bl, ku, bu

def getConvenientGeneralActivationBound(l,u, activation, use_constant=False, remain_tangent_line_info=False):
    if (l>u).sum()>0:
        raise Exception('l must be less or equal to u')
        # print('l greater than u')
        # print(l-u, (l-u).max())
        # if (l-u).max()>1e-4:
        #     raise Exception('l must be less or equal to u')
        # temp = l>u
        # l_temp = l[temp]
        # l[temp] = u[temp]
        # u[temp] = l_temp

    device = l.device
    ku = torch.zeros(u.shape, device = device)
    bu = torch.zeros(u.shape, device = device)
    kl = torch.zeros(l.shape, device = device)
    bl = torch.zeros(l.shape, device = device)
    if use_constant:
        #we have assumed that the activaiton is monotomic
        function = Activation[activation][0]
        bu = function(u)
        bl = function(l)
        return kl, bl, ku, bu
    if activation == 'relu':
        kl, bl, ku, bu = get_bound_for_relu(l, u, adaptive=False)
        return kl, bl, ku, bu
    if activation == 'relu_adaptive':
        kl, bl, ku, bu = get_bound_for_relu(l, u, adaptive=True)
        return kl, bl, ku, bu
    if activation == 'ba':
        # print(u)
        print('binary activation')
        bu = torch.sign(u)
        bl = torch.sign(l)
        idx = (l<0) * (u>0) * (u.abs() > l.abs())
        kl[idx] = 2/u[idx]

        idx = (l<0) * (u>0) * (u.abs() < l.abs())
        ku[idx] = -2/l[idx]

        # idx = (l>0) * (l>0.8*u)
        # ku[idx] = 1/l[idx]
        # #ku l + bu = 1
        # bu[idx] = 1-ku[idx] * l[idx]
        print('uncertain neurons', ((l<0) * (u>0)).float().mean())
        return kl, bl, ku, bu
    
    idx = (l==u)
    if idx.sum()>0:
        bu[idx] = Activation[activation][0](l[idx])
        bl[idx] = Activation[activation][0](l[idx])
        
        ku[idx] = 0
        kl[idx] = 0
    
    valid = (1-idx)
    
    if remain_tangent_line_info:
        su = torch.zeros(u.shape, device = device)
        su_valid = torch.zeros(u.shape, device = device)
        sl = torch.zeros(l.shape, device = device)
        sl_valid = torch.zeros(l.shape, device = device)
    if valid.sum()>0:
        func = Activation[activation][0]
        dfunc = Activation[activation][1]
        if remain_tangent_line_info:
            kl_temp, bl_temp, ku_temp, bu_temp, sl_temp,sl_valid_temp,su_temp,su_valid_temp = getGeneralActivationBound(
                    l[valid],u[valid], func, dfunc, remain_tangent_line_info=True)
        else:
            kl_temp, bl_temp, ku_temp, bu_temp = getGeneralActivationBound(
                    l[valid],u[valid], func, dfunc)
        kl[valid] = kl_temp
        ku[valid] = ku_temp
        bl[valid] = bl_temp
        bu[valid] = bu_temp
        if remain_tangent_line_info:
            su[valid] = su_temp
            su_valid[valid] = su_valid_temp
            sl[valid] = sl_temp
            sl_valid[valid] = sl_valid_temp
    
    if remain_tangent_line_info:
        return kl, bl, ku, bu, sl, sl_valid, su, su_valid
    else:
        return kl, bl, ku, bu

def search_du(l,u,func,d_func, acc=1e-3):
    # we require l<0 and u>0
    # seach du such that the tangent line at du roughly passes througth the point l, func(l)
    # but be above the point l, func(l)
    
    k = d_func(u)
    # k*u + b = func(u)
    b = func(u)-k*u
    diff = k*l+b-func(l)
    valid = diff>=0

    lower = torch.zeros(l[valid].shape, device=l.device)
    upper = u[valid].detach().clone()
    l_valid = l[valid]
    func_l = func(l_valid)
    

    search = (upper-lower) > acc#((upper-lower) / ((upper+lower).abs()/2+1e-8)) > acc
    while search.sum()>0:
        s = (lower[search] + upper[search])/2
        k = d_func(s)
        # k*s + b = func(s)
        b = func(s)-k*s
        # pdb.set_trace()
        diff = k*l_valid[search]+b-func_l[search]

        pos = diff >= 0

        search_copy = search.detach().clone()

        search[search] = pos # set active units in search to pos
        upper[search] = s[pos]
        lower[search_copy-search] = s[1-pos]

        search = (upper-lower) > acc#((upper-lower) / ((upper+lower).abs()/2+1e-8)) > acc

    upper_real = u.detach().clone()
    upper_real[valid] = upper 

    neg = 1-valid #for these points, upper bounding line is not a tagent line,
    # it should be the line passes the 2 end points  
    return upper_real, neg

def search_dl(l,u,func,d_func, acc=1e-3):
    # we require l<0 and u>0
    # seach dl such that the tangent line at dl roughly passes througth the point u, func(u)
    # but be below the point u, func(u)
    
    k = d_func(l)
    # k*l + b = func(l)
    b = func(l)-k*l
    diff = k*u+b-func(u)
    valid = diff<=0

    lower = l[valid].detach().clone()
    upper = torch.zeros(u[valid].shape, device=u.device)
    u_valid = u[valid]
    func_u = func(u_valid)
    

    search = (upper-lower) > acc#((upper-lower) / ((upper+lower).abs()/2+1e-8)) > acc
    while search.sum()>0:
        s = (lower[search] + upper[search])/2
        # print('lower', lower[search])
        # print('s',s)
        # print('upper', upper[search])
        k = d_func(s)
        # k*s + b = func(s)
        b = func(s)-k*s
        # pdb.set_trace()
        diff = k*u_valid[search]+b-func_u[search]

        pos = diff >= 0

        search_copy = search.detach().clone()

        search[search] = pos # set active units in search to pos
        upper[search] = s[pos]
        lower[search_copy-search] = s[1-pos]

        search = (upper-lower) > acc# ((upper-lower) / ((upper+lower).abs()/2+1e-8)) > acc

    lower_real = l.detach().clone()
    lower_real[valid] = lower 

    pos = 1-valid #for these points, lower bounding line is not a tagent line,
    # it should be the line passes the 2 end points  
    return lower_real, pos

def general_lb_pn(l, u, func, d_func, k=None, b=None, remain_tangent_line_info=False):
    # compute the lower bounding line of func
    # require l<0<u
    # k and b is the parameter of the line that passes through the two end points
    lower, pos = search_dl(l,u,func,d_func, acc=1e-3)
    kl,bl = get_tangent_line(lower, func, d_func)

    if k is None:
        yl = func(l)
        yu = func(u)
        k = (yu - yl) / (u-l)
        b = yl - k * l
    kl[pos] = k[pos]
    bl[pos] = b[pos]
    if remain_tangent_line_info:
        return kl,bl,lower,pos
    else:
        return kl,bl

def general_ub_pn(l, u, func, d_func, k=None, b=None, remain_tangent_line_info=False):
    # compute the upper bounding line of func
    # require l<0<u
    upper, neg = search_du(l,u,func,d_func, acc=1e-3)
    ku,bu = get_tangent_line(upper, func, d_func)

    if k is None:
        yl = func(l)
        yu = func(u)
        k = (yu - yl) / (u-l)
        b = yl - k * l
    ku[neg] = k[neg]
    bu[neg] = b[neg]
    if remain_tangent_line_info:
        return ku,bu,upper,neg
    else:
        return ku,bu


def getGeneralActivationBound(l,u, func, dfunc, remain_tangent_line_info=False):
    #l and u are tensors of any shape. l and u must have the same shape
    #the first dimension of l and u is the batch dimension
    #users must make sure that u > l
    
    #initialize the desired variables
    device = l.device
    ku = torch.zeros(u.shape, device = device)
    bu = torch.zeros(u.shape, device = device)
    kl = torch.zeros(l.shape, device = device)
    bl = torch.zeros(l.shape, device = device)

    if remain_tangent_line_info:
        su = torch.zeros(u.shape, device = device)
        su_valid = torch.zeros(u.shape, device = device)
        sl = torch.zeros(l.shape, device = device)
        sl_valid = torch.zeros(l.shape, device = device)
    
    yl = func(l)
    yu = func(u)
    k = (yu - yl) / (u-l)
    b = yl - k * l
    d = (u+l) / 2
    
    func_d = func(d)
    d_func_d = dfunc(d) #derivative of tanh at x=d
    
    #l and u both <=0
    minus = (u <= 0) * (l<=0)
    ku[minus] = k[minus]
    bu[minus] = b[minus] # upper bounding line passes through the two end points
    kl[minus] = d_func_d[minus]
    bl[minus] = func_d[minus] - kl[minus] * d[minus] # lower bounding line is the tangent line at the middle point
    if remain_tangent_line_info: # lower bounding line can be optimized
        sl[minus] = d[minus]
        sl_valid[minus] = 1

    #l and u both >=0
    plus = (l >= 0)
    kl[plus] = k[plus]
    bl[plus] = b[plus] # lower bounding line passes through the two end points
    ku[plus] = d_func_d[plus]
    bu[plus] = func_d[plus] - ku[plus] * d[plus] # upper bounding line is the tangent line at the middle point
    if remain_tangent_line_info: # upper bounding line can be optimized
        su[plus] = d[plus]
        su_valid[plus] = 1

    #l < 0 and u>0
    pn = (l < 0) * (u > 0)
    if remain_tangent_line_info:
        kl[pn], bl[pn], sl[pn], pos = general_lb_pn(l[pn], u[pn], func, dfunc, k=k[pn], b=b[pn], 
                                                    remain_tangent_line_info=True)
        sl_valid[pn] = 1
        pn_copy = pn.detach().clone()
        pn[pn] = pos 
        # in those l<0<u elements, some lower bounding line is not tangent line, don't need to optimize over them
        sl_valid[pn] = 0

        pn = pn_copy
        ku[pn], bu[pn], su[pn], neg = general_ub_pn(l[pn], u[pn], func, dfunc, k=k[pn], b=b[pn], 
                                                    remain_tangent_line_info=True)
        su_valid[pn] = 1
        # pdb.set_trace()
        pn_copy[pn_copy] = neg
        # in those l<0<u elements, some upper bounding line is not tangent line, don't need to optimize over them
        su_valid[pn_copy] = 0

        return kl,bl,ku,bu,sl,sl_valid,su,su_valid
    else:
        kl[pn], bl[pn] = general_lb_pn(l[pn], u[pn], func, dfunc, k=k[pn], b=b[pn])
        ku[pn], bu[pn] = general_ub_pn(l[pn], u[pn], func, dfunc, k=k[pn], b=b[pn])
    
    return kl, bl, ku, bu



    
    
    
    