import pandas as pd
import numpy as np
import torch


if __name__ == '__main__':

    dataset=pd.read_csv('Sensorless_drive_diagnosis.txt', sep=" ", header=None)#,names=Feature_columnnnames)
    data = np.array(dataset)
    X = torch.from_numpy(data[:,0:48])
    mean = torch.mean(X, dim=0)
    std = torch.std(X, dim=0)
    
    total_num = data.shape[0]

    test_rate = 0.4
    test_num = np.int(total_num * test_rate)
    train_num = total_num - test_num

    idx = np.random.permutation(total_num)
    train_idx = idx[0:train_num]
    test_idx = idx[train_num:]

    train_data = data[train_idx,:]
    X_train = torch.from_numpy(train_data[:, 0:48])
    y_train = torch.from_numpy(train_data[:,48]).long()

    test_data = data[test_idx,:]
    X_test = torch.from_numpy(test_data[:,0:48])
    y_test = torch.from_numpy(test_data[:,48]).long()

    torch.save({'X':X_train, 'y':y_train, 'mean':mean, 'std':std}, 'train_data.ckpt')
    torch.save({'X':X_test, 'y':y_test, 'mean':mean, 'std':std}, 'test_data.ckpt')
