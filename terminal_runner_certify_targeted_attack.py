import torch
# from certify_mlp_layerwise_optimize_lower_bounding_line import FcNetLayerwiseOpt
# from certify_mlp_neuronwise_optimize import FcNetNeuronwiseOpt
from frown_relu_neuronwise import FcNetNeuronwiseOpt
from frown_relu_batchwise import FcNetBatchwiseOpt
from frown_general_activation_neuronwise import FcNetGeneralActivationNeuronwiseOpt
from frown_general_activation_batchwise import FcNetGeneralActivationBatchwiseOpt
# from certify_binary_mlp_layerwise_optimize import BinaryFcNetLayerwiseOpt
from certify_mlp import fcNet
# from certify_mlp_efficient import fcNet

from models.mlp import FcNet
from utils.sample_data import sample_mnist_data, sample_cifar10_data, sample_drive_data

from shutil import copyfile
import os
import time
import argparse

def getMaximumEps(model,p,true_label, target_label, eps0=1, max_iter=100, x=None, acc=0.001,
                      gx0_trick = True):
        # perform binary search to find the largest certified safety region
        #when (u_eps-l_eps) / [(u_eps+l_eps)/2] < acc, we stop searching
        if model.x is None and x is None:
            raise Exception('You must first attach data to the net or feed data to this function')
        if x is None:
            x = model.x
        N = x.shape[0]
        idx=torch.arange(N)
        l_eps = torch.zeros(N, device = x.device) #lower bound of eps
        u_eps = torch.ones(N, device = x.device) * eps0 #upper bound of eps
        # use gx0_trick: the "equivalent output node" is equal to N (number of samples in one batch)
        if gx0_trick == True:
            print("W[-1] size = {}".format(model.W[-1].shape)) # W[-1] is of shape (num_classes, in_features)
            print("b[-1] size = {}".format(model.b[-1].shape)) # b[-1] is of shape (num_classes)
            W_ori = model.W[-1].detach().clone()
            b_ori = model.b[-1].detach().clone()
            W_gx0 = model.W[-1][true_label,:]-model.W[-1][target_label,:]
            b_gx0 = model.b[-1][true_label]-model.b[-1][target_label]
            model.W[-1] = W_gx0
            model.b[-1] = b_gx0
            # now W is of shape (batch, in_features)
            # W_new_i = W_true[i] - W_target[i]
            print("after gx0_trick W[-1] size = {}".format(model.W[-1].shape))
            print("after gx0_trick b[-1] size = {}".format(model.b[-1].shape))
        
        yL, yU = model.getLastLayerBound(u_eps, p, x=x, 
                                        clearIntermediateVariables=True)
        #yL and yU is of shape (N, N) if gx0_trick == True
        
        if gx0_trick: 
            # only yL is used
            lower = torch.diag(yL)
            increase_u_eps = lower > 0 
            print("initial lower bound of (f_true - f_target) = {}".format(lower))
            print("increase_u_eps = {}".format(increase_u_eps))
        else:
            true_lower = yL[idx,true_label]
            target_upper = yU[idx,target_label]
            increase_u_eps = true_lower > target_upper 
            #indicate whether to further increase the upper bound of eps 
            print('initial true_lower - target_upper \n', true_lower - target_upper)
            print("increase_u_eps = {}".format(increase_u_eps))
        while (increase_u_eps.sum()>0):
            #find true and nontrivial lower bound and upper bound of eps
            num = increase_u_eps.sum()
            l_eps[increase_u_eps] = u_eps[increase_u_eps]
            #for those increase_u_eps is 1, we can increase their l_eps to u_eps
            #for those increase_u_eps is 0, u_eps is big enough, we keep their l_eps and u_eps
            
            u_eps[increase_u_eps ] = u_eps[increase_u_eps ] * 2
            #for those increase_u_eps is 1, increase their increase_u_eps by
            #a factor of 2, try to find a big enough u_eps
            if gx0_trick:
                # make sure that W[-1] and b[-1] corresponds to x[increase_u_eps,:]
                model.W[-1] = W_gx0[increase_u_eps,:] 
                model.b[-1] = b_gx0[increase_u_eps]
            yL, yU = model.getLastLayerBound(u_eps[increase_u_eps], p, 
                        x=x[increase_u_eps,:],clearIntermediateVariables=True)
            #yL and yU only for those equal to 1 in increase_u_eps
            #they are of size (num,_)
            
            if gx0_trick:
                # lower = yL[torch.arange(num),idx[increase_u_eps]]
                lower = torch.diag(yL)
                temp = lower > 0
                #when lower>0, we can further increase u_eps
                print("lower bound of (f_true - f_target) = {}".format(lower))
            else:
                true_lower = yL[torch.arange(num),true_label[increase_u_eps]]
                target_upper = yU[torch.arange(num),target_label[increase_u_eps]]
                temp = true_lower > target_upper #size num
                print('true_lower - target_upper \n', true_lower- target_upper)
        
            increase_u_eps[increase_u_eps] = temp 

        print('Finished finding upper and lower bound')
        print('The upper bound we found is \n', u_eps)
        print('The lower bound we found is \n', l_eps)
        
        search = (u_eps-l_eps) / ((u_eps+l_eps)/2+1e-8) > acc
        #indicate whether to further perform binary search
        
        iteration = 0 
        while(search.sum()>0):
            #perform binary search
            print('Binary search step: %d' % (iteration+1))
            if iteration > max_iter:
                print('Have reached the maximum number of iterations')
                break
            #print(search)
            num = search.sum()
            eps = (l_eps[search]+u_eps[search])/2
            if gx0_trick:
                # make sure that W[-1] and b[-1] corresponds to x[search,:]
                model.W[-1] = W_gx0[search,:]
                model.b[-1] = b_gx0[search]
            yL, yU = model.getLastLayerBound(eps, p, x=x[search,:],
                            clearIntermediateVariables=True)
            if gx0_trick:
                # lower = yL[torch.arange(num),idx[search]]
                lower = torch.diag(yL)
                temp = lower > 0
            else:
                true_lower = yL[torch.arange(num),true_label[search]]
                target_upper = yU[torch.arange(num),target_label[search]]
                temp = true_lower>target_upper
            search_copy = search.data.clone()

            search[search] = temp 
            #set all active units in search to temp
            #original inactive units in search are still inactive
            
            l_eps[search] = eps[temp]
            #increase active and true_lower>target_upper units in l_eps 
            
            u_eps[search_copy-search] = eps[temp==0]
            #decrease active and true_lower<target_upper units in u_eps
            
            # search = (u_eps-l_eps) / [(u_eps+l_eps)/2] > acc #reset active units in search 
            search = (u_eps-l_eps) / ((u_eps+l_eps)/2+1e-8) > acc
            print('u_eps \n', u_eps)
            print('l_eps \n', l_eps)
            print('u_eps - l_eps \n', u_eps - l_eps)
            if gx0_trick:
                print('lower bound of (f_true - f_target) = {}'.format(lower))
            else:
                print('true_lower - target_upper \n', true_lower-target_upper)
            
            iteration = iteration + 1
        if gx0_trick:
            # restore original W[-1] and b[-1]
            model.W[-1] = W_ori.detach()
            model.b[-1] = b_ori.detach() 
        
        return l_eps, u_eps

if __name__ == '__main__':
    '''
    python terminal_runner_certify_targeted_attack.py --cuda -1 --p_norm 200   --eps0 0.2    --acc 0.01 --work_dir pretrained_models/mnist/net_10_20_relu/   --model_name merged_bn_net --num_neurons 20 --num_layers  10   --activation relu_adaptive --batch_size 50 --num_batches 2 --dataset mnist --save_result
    python terminal_runner_certify_targeted_attack.py --cuda -1 --p_norm 200   --eps0 0.2    --acc 0.01 --work_dir pretrained_models/drive/net_8_20_relu/   --model_name merged_bn_net --num_neurons 20 --num_layers  8   --activation relu_adaptive --batch_size 50 --num_batches 2 --dataset drive --neuronwise_optimize
    '''
    parser = argparse.ArgumentParser(description='Clever Terminal Runner')
    parser.add_argument('--cuda', type=int, metavar='c',
                        help='gpu idx to use (necessary). if cuda<0, we will use cpu')
    parser.add_argument('--p_norm', type=int, metavar='p',
                        help='p norm (necessary)')
    parser.add_argument('--eps0', type=float, default=1, metavar='e',
                        help='start point for eps to search (default: 1)')
    parser.add_argument('--acc', type=float, default=0.01,
                        help='required relative precision for eps (default: 0.01)')
    parser.add_argument('--save_result', action='store_true',
                        help='whether to save the computed result (default: False)')
    
    parser.add_argument('--batch_size', type=int, default=1000,
                        help='number of samples to handle at a time (default: 1000)')
    parser.add_argument('--max_batch_size', type=int, default=1000,
                        help='maximum number of samples to handle at a time (default: 1000)')
    parser.add_argument('--num_batches', type=int, default=1,
                        help='number of batches (default: 1)')

    parser.add_argument('--work_dir', default = 'pretrained_models/net_10_20_relu/', type = str, metavar = 'WD',
                        help = 'the directory where the pretrained model is stored and the place to save the computed result')
    parser.add_argument('--model_name', default = 'merged_bn_net', type = str, metavar = 'MN',
                        help = 'the name of the pretrained model (default: merged_bn_net)')
    parser.add_argument('--num_neurons', type=int, default=20,
                        help='number of neurons per layer')
    parser.add_argument('--num_layers', type=int, default=10,
                        help='number of layers')
    parser.add_argument('--activation', type=str, default='relu_adaptive',
                        help='activation function')
    parser.add_argument('--layerwise_optimize', action='store_true',
                        help='whether to layerwise optimize the slopes of the lower bounding lines (default: False)')
    parser.add_argument('--neuronwise_optimize', action='store_true',
                        help='whether to neuronwise optimize the slopes of the lower bounding lines (default: False)')
    parser.add_argument('--batchwise_optimize', action='store_true',
                        help='whether to batchwise optimize the slopes of the lower bounding lines (default: False)')
    parser.add_argument('--batchwise_size', type=int, default=64,
                        help='number of neurons to backward at a time (default: 1000)')
    parser.add_argument('--deterministic', action='store_true',
                        help='whether to use deterministic samples and target labels (default: False)')

    parser.add_argument('--dataset', type=str, default='mnist',
                        help='the dataset to use')

    args = parser.parse_args()
    
    if args.cuda < 0:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:'+str(args.cuda))
    
   

    save_result = args.save_result
    N = args.batch_size #number of samples to handle at a time.\
    num_batches = args.num_batches
    p = args.p_norm
    if p>100:
        p = float('inf')
    print('p', p)

    # use_constant = args.use_constant
    eps0 = args.eps0
    output_dimension_of_all_datasets = {'mnist':10, 'cifar10':10, 'drive':11}
    output_dimension = output_dimension_of_all_datasets[args.dataset]
    input_dimension_of_all_datasets = {'mnist':28*28, 'cifar10':3*32*32, 'drive':48}
    input_dimension = input_dimension_of_all_datasets[args.dataset]
    # input_dimension = 28*28
    num_layers = args.num_layers
    num_neurons = args.num_neurons
    activation = args.activation
    # activation = 'relu'
    layer_wise_optimize = args.layerwise_optimize
    neuron_wise_optimize = args.neuronwise_optimize
    gx0_trick = True
    targeted = True
    
    save_dir   = args.work_dir + 'certified_targeted_bound/%s_norm/%s/' % (str(p), activation)
    model_file = args.work_dir + args.model_name
    if layer_wise_optimize:
        save_dir = save_dir[0:-1] + '_layerwise_optimize/'
        fc = FcNetLayerwiseOpt(input_dimension=input_dimension, 
                    output_dimension = output_dimension, 
                    num_layers = num_layers,
                    num_neurons = num_neurons, activation=activation)
    elif (neuron_wise_optimize and 'relu' in activation):
        save_dir = save_dir[0:-1] + '_neuronwise_optimize/'
        fc = FcNetNeuronwiseOpt(input_dimension=input_dimension, 
                    output_dimension = output_dimension, 
                    num_layers = num_layers,
                    num_neurons = num_neurons, activation=activation)
    elif (neuron_wise_optimize):
        save_dir = save_dir[0:-1] + '_neuronwise_optimize/'
        fc = FcNetGeneralActivationNeuronwiseOpt(input_dimension=input_dimension, 
                    output_dimension = output_dimension, 
                    num_layers = num_layers,
                    num_neurons = num_neurons, activation=activation)
    elif (args.batchwise_optimize and 'relu' in activation):
        save_dir = save_dir[0:-1] + '_batchwise_%s_optimize/' % str(args.batchwise_size)
        fc = FcNetBatchwiseOpt(input_dimension=input_dimension, 
                    output_dimension = output_dimension, 
                    num_layers = num_layers,
                    num_neurons = num_neurons, activation=activation,
                    batch_size = args.batchwise_size)
    elif (args.batchwise_optimize):
        save_dir = save_dir[0:-1] + '_batchwise_%s_optimize/' % str(args.batchwise_size)
        fc = FcNetGeneralActivationBatchwiseOpt(input_dimension=input_dimension, 
                    output_dimension = output_dimension, 
                    num_layers = num_layers,
                    num_neurons = num_neurons, activation=activation,
                    batch_size = args.batchwise_size)
    else:
        fc = fcNet(input_dimension=input_dimension, 
                    output_dimension = output_dimension, 
                    num_layers = num_layers,
                    num_neurons = num_neurons, activation=activation)


    net = FcNet(num_layers=num_layers, num_neurons=num_neurons, 
                input_dimension=input_dimension, 
                output_dimension=output_dimension, bn=False, affine=False, 
                activation=activation)
    print(net)
    
    net.load_state_dict(torch.load(model_file, map_location='cpu'))
    net.eval()
    net.to(device)

    fc.model = net.model
    fc.eval()
    fc.to(device)
    fc.extractWeight(clear_original_model=False)
    
    total_x = torch.zeros(0, device=device)
    total_true_label = torch.zeros(0, device=device).long()
    total_target_label = torch.zeros(0, device=device).long()
    total_l = torch.zeros(0, device=device)
    total_u = torch.zeros(0, device=device)
    total_num = 0
    total_time = 0

    for i in range(num_batches):
        print('Computing bounds for the %d-th batch' % (i+1))
        if args.dataset == 'mnist':
            x, true_label, target_label = sample_mnist_data(N, device, num_labels=10,
                            data_dir='./datasets/mnist', train=False, shuffle=not args.deterministic, 
                            model=fc)
        elif args.dataset == 'cifar10':
            x, true_label, target_label = sample_cifar10_data(N, device, num_labels=10,
                            data_dir='./datasets/cifar10', train=False, shuffle=not args.deterministic, 
                            model=fc)
        elif args.dataset == 'drive':
            x, true_label, target_label = sample_drive_data(N, device, num_labels=11,
                            data_dir='./datasets/drive/', train=False, shuffle=not args.deterministic, 
                            model=fc)
        else:
            raise Exception('%s dataset isnot supported' % args.dataset)

        max_batch_size = args.max_batch_size
        x = x[0:max_batch_size]
        true_label = true_label[0:max_batch_size]
        target_label = target_label[0:max_batch_size]

        if args.deterministic:
            target_label = torch.fmod(true_label + 2, 10)
        if x.shape[0] < 1:
            continue
        if (true_label == target_label).sum()>0:
            raise Exception('Some target labels equal to true label')
        x = x.to(device)
        true_label =  true_label.to(device)
        num = x.shape[0]
        x = x.view(num, -1)

        start = time.time()
        
        # getMaximumEps will update W[-1], b[-1] at the beginning if gx0_trick == True
        # and will restore W[-1], b[-1] to their original value at the end
        l_eps, u_eps = getMaximumEps(fc,p,true_label, target_label, eps0=eps0, max_iter=100, x=x, acc=args.acc,
                        gx0_trick = True)
        end = time.time()
        
        total_time = total_time + end - start
        total_x = torch.cat([total_x, x.detach()])
        total_true_label = torch.cat([total_true_label, true_label])
        total_target_label = torch.cat([total_target_label, target_label])
        total_l = torch.cat([total_l, l_eps.detach()])
        total_u = torch.cat([total_u, u_eps.detach()])
        total_num = total_num + num

        print('statistics of this batch l_eps for %d images' % num)
        print('%.8f %.8f %.8f %.8f' % 
            (l_eps.min(), l_eps.mean(),l_eps.max(), l_eps.std()))

        print('For all the samples, the lower bound we found is:')
        print(total_l)
        print('Computed %s norm certified bound of %d samples for model %s in %.2f seconds' % (str(p), total_num, model_file, total_time))
        print('statistics of l_eps')
        print('min: %.8f mean: %.8f max: %.8f std: %.8f' % 
            (total_l.min(), total_l.mean(),total_l.max(), total_l.std()))
    if save_result:
        os.makedirs(save_dir, exist_ok=True)
        copyfile(__file__, save_dir+__file__)
        torch.save({'true_label':total_true_label, 'target_label':total_target_label, 'x':total_x, 'p':p,
                    'l_eps':total_l, 'u_eps':total_u, 'total_time':total_time}, save_dir+'bound')
        print('Have saved the result to ' + save_dir+'bound')
