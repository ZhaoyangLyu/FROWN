# Fastened CROWN: Tightened Neural Network Robustness Certificates

*FROWN* (**F**astened C**ROWN**) is a general algorithm to
tighten the certifiable regions guaranteed by CROWN, which is also theoretically applicable to tighten convolutional neural network certificate CNN-Cert and recurrent neural network certificate POPQORN. We conduct extensive experiments on various networks trained individually to verify the effectiveness of FROWN in safeguarding larger robust regions, and compare FROWN's efficiency with linear programming (LP) based methods in term of improving bounds obtained by CROWN. 


This repo intends to release code and the appendix for our work:


Zhaoyang Lyu\*, Ching-Yun Ko\*, Zhifeng Kong, Ngai Wong, Dahua Lin, Luca Daniel, ["Fastened CROWN: Tightened Neural Network Robustness Certificates"](https://arxiv.org/abs/1912.00574), AAAI 2020

\* Equal contribution

The appendix of our paper is in the file `_appendix.pdf`.

Setup
--------------------------------------------------------------

The code is tested with python 3.7.3, Pytorch 1.1.0 and CUDA 9.0. Run the following
to install pytorch and its CUDA toolkit:

```
conda install pytorch torchvision cudatoolkit=9.0 -c pytorch
```

Then clone this repository:

```
git clone https://github.com/ZhaoyangLyu/FROWN.git
cd FROWN
```
Step 1: Train models
--------------------------------------------------------------
(1) Train classifiers on Sensorless Drive Diagnosis Dataset:\
The dataset is available at https://archive.ics.uci.edu/ml/datasets/Dataset+for+Sensorless+Drive+Diagnosis. We download `Sensorless_drive_diagnosis.txt` to the folder `FROWN/datasets/drive`.
We run the following command to preprocess the data.
```
cd FROWN/datasets/drive
python process_data.py
```
**Remember to add process_data.py to drive folder in the future**
This will split Sensorless_drive_diagnosis.txt into train and test datasets, and save them as pytorch tensors: `train_data.ckpt` and `test_data.ckpt`. (The files are already in the `drive` folder. You don't need to process the data again.)
Then you can run the following command to train a classifier of 8 layers, with 20 neurons in each layer and ReLU activation for each layer.

```
python train_model.py --cuda 0 --num_layers 8 --num_neurons 20  --activation relu --dataset drive --num_epochs 50 --lr 0.1 --lr_decay_interval 10 --lr_decay_factor 0.1
```

The argument **cuda** controls which GPU to use to train the model. You can set it -1 if you want to use CPU for trainning. You can also change the arguments **num_layers**, **num_neurons** and **activation** ('relu', 'sigmoid' and 'tanh' are accepted activation functions) to train classifiers of different architectures.\
After running the above command, the trained model will be saved to the folder `FROWN/pretrained_models/drive/net_8_20_relu`. There will be two files in the folder: `net` and `merged_bn_net`. `net` is the original network with all batch norm (BN) layers. However, in evaluation stage, BN layer is equivalent to a linear transformation. Therefore, we merge every BN layer to the fully connected linear next to it, which simplies robustness evaluation of the network. The resulting network is saved to `merged_bn_net`. We will always use `merged_bn_net` to compute certified bounds in later experiments.


(2) Train classifiers on MNIST or CIFAR10 dataset:\
In order to train classifiers for MNIST or CIFAR10 datasets, you can simply change the argument **dataset** to 'mnist' or 'cifar10' in the above command. The corresponding dataset will be downloaded to the folder `FROWN/datasets` automatically.

Below is a list of models that we provide in the folder `FROWN/pretrained_models`. You can train more models by running the file `train_model.py` to obtain all the models reported in our paper.

|           Dataset          | Number of Layers  | Number of Neurons in Each Layer | Activation |          Folder         |
|:--------------------------:|:-----------------:|:-------------------------------:|:----------:|:-----------------------:|
| Sensorless Drive Diagnosis |         8         |                20               |    relu    |   drive/net_8_20_relu   |
| Sensorless Drive Diagnosis |         8         |                20               |   sigmoid  |  drive/net_8_20_sigmoid |
|            MNIST           |         10        |                20               |    relu    |   mnist/net_10_20_relu  |
|            MNIST           |         10        |                20               |   sigmoid  | mnist/net_10_20_sigmoid |
|            MNIST           |         3         |               100               |    relu    |   mnist/net_3_100_relu  |
|            MNIST           |         3         |               100               |   sigmoid  | mnist/net_3_100_sigmoid |


Run CROWN
--------------------------------------------------------------
To use CROWN to certify bounds for the previously trained 8*[20] ReLU network on Sensorless Drive Diagnosis dataset, run the following command.
```
python terminal_runner_certify_targeted_attack.py --cuda 0 --p_norm 200 --eps0 0.2 --acc 0.01 --work_dir pretrained_models/drive/net_8_20_relu/ --model_name merged_bn_net --num_neurons 20 --num_layers 8 --activation relu --batch_size 500 --num_batches 2 --dataset drive --save_result
```
It will sample examples and their targeted-attack labels randomly, and report the averaged infinity-norm bounds. The complete result will saved to the file **FROWN/pretrained_models/drive/net_8_20_sigmoid/certified_targeted_bound/inf_norm/sigmoid/bound**.

You can change the argument **cuda** to decide on which GPU to run the script (set to -1 if you want to compute on CPU). You can also change **p_norm** to any number between 1 and infinity to specify the p-norm of the bounds. Note that any **p_norm** larger than 100 will considered as infinity-norm.
Our script is able to compute bounds for multiple examples at the same time. The argument **batch_size** controls how many examples to compute bounds for at a time, and **num_batches** controls how many batches to compute. The total number of examples being considered will be **batch_size** * **num_batches**.
In general, you should choose large **batch_size** as long as you have sufficient memory, because computing bounds for multiple examples in parallel is usually faster than computing bound one by one in a loop. 

You can compute bounds for a different model by changing the arguments **work_dir** and **model_name**. Also remember to change the arguments **num_layers**, **num_neurons** and **activation** accordingly. 
The script will load the model from the file **[work_dir]/[model_name]** and save result to the file **[work_dir]/certified_targeted_bound/[p_norm]/[activation]/bound**. 

To compute bounds for classifiers on MNIST or CIFAR10 dataset, you can simply change the argument **dataset** ti mnist or cifar10, and change the other arguments accordingly to specify the corresponding model.

Run FROWN
--------------------------------------------------------------
To use FROWN to certify bounds for the previously trained models, you can simply add **--neuronwise_optimize** to the above CROWN command. For example, if you want to use FROWN to certify bounds for the previously trained 8*[20] ReLU network on Sensorless Drive Diagnosis dataset, run the following command.
```
python terminal_runner_certify_targeted_attack.py --cuda 0 --p_norm 200 --eps0 0.2 --acc 0.01 --work_dir pretrained_models/drive/net_8_20_relu/ --model_name merged_bn_net --num_neurons 20 --num_layers 8 --activation relu --batch_size 500 --num_batches 2 --dataset drive --save_result --neuronwise_optimize
```
This will take longer time while give larger certifed bounds.

As mentioned in Section A.10 in the appendix of our paper, FROWN can be speeded up by optimizing neurons in a layer group by group, instead of one by one. To apply this technique, replace **--neuronwise_optimize** with **--batchwise_optimize** in the above command as below.
```
python terminal_runner_certify_targeted_attack.py --cuda 0 --p_norm 200 --eps0 0.2 --acc 0.01 --work_dir pretrained_models/drive/net_8_20_relu/ --model_name merged_bn_net --num_neurons 20 --num_layers 8 --activation relu --batch_size 500 --num_batches 2 --dataset drive --save_result --batchwise_optimize --batchwise_size 5
``` 
The argument **--batchwise_size** specifies how many neurons to optimize at a time, and it should be a factor of **num_neurons**. The larger **--batchwise_optimize** you choose, the more speed-up you will gain, but the looser bounds you will get. This parameter balances the trade-off between tightness of bounds and efficiency.

Run LP-based methods
--------------------------------------------------------------
To be available soon.
