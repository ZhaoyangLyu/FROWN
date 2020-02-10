import torch
import torch.nn as nn


def bn_to_linear(bn, matrix = True):
    mean = bn.running_mean
    var = bn.running_var
    eps = bn.eps
    if bn.affine:
        gamma = bn.weight.data
        beta = bn.bias

    multiplier = 1/torch.sqrt(var+eps)
    if bn.affine:
        multiplier = multiplier * gamma
    bias = -mean * multiplier
    if bn.affine:
        bias = bias + bn.bias

    if matrix:
        weight = torch.diag(multiplier)
    else:
        weight = multiplier
    return weight, bias


def merge_linear_and_bn(fc, bn):
    # bn.eval()
    weight2, bias2 = bn_to_linear(bn)
    # num_features is in_features for fc
    weight1 = fc.weight # size (out_features, num_features)
    bias1 = fc.bias # out_features
    new_weight = torch.matmul(weight2, weight1)
    new_bias = torch.matmul(weight2, bias1) + bias2
    in_features = fc.in_features
    out_features = fc.out_features
    new_fc = nn.Linear(in_features, out_features)
    new_fc.weight.data = new_weight
    new_fc.bias.data = new_bias
    return new_fc


def merge_bn_in_Sequential(model):
    # print(len(model))
    layers = []
    for i in range(len(model)):
        # print(i, model[i])
        if (i < len(model)-1) and isinstance(model[i+1], nn.BatchNorm1d):
            if not isinstance(model[i], nn.Linear):
                raise Exception('nn.BatchNorm1d can only be merged  \
                                with nn.Linear')
            new_fc = merge_linear_and_bn(model[i], model[i+1])
            layers.append(new_fc)
        elif isinstance(model[i], nn.BatchNorm1d):
            pass
        elif isinstance(model[i], nn.Dropout):
            pass
        else:
            layers.append(model[i])
    return nn.Sequential(*layers)

