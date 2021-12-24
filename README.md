# Approximated Oracle Filter Pruning for Destructive CNN Width Optimization

UPDATE: pytorch implementation released. But I am not sure whether it works with multi-processing distributed data parallel. I only tested with a single GPU and multi-GPU data parallel. The Tensorflow version still works, but I would not suggest you read it.

This repository contains the codes for the following ICML-2019 paper 

[Approximated Oracle Filter Pruning for Destructive CNN Width Optimization](http://proceedings.mlr.press/v97/ding19a.html).

Citation:

    @inproceedings{ding2019approximated,
    title={Approximated Oracle Filter Pruning for Destructive CNN Width Optimization},
    author={Ding, Xiaohan and Ding, Guiguang and Guo, Yuchen and Han, Jungong and Yan, Chenggang},
    booktitle={International Conference on Machine Learning},
    pages={1607--1616},
    year={2019}
    }

This demo will show you how to
1. Reproduce 65% pruning ratio of VGG on CIFAR-10.
2. Reproduce 50% pruning ratio of ResNet-56 on CIFAR-10.

About the environment:
1. We used torch==1.3.0, torchvision==0.4.1, CUDA==10.2, NVIDIA driver version==440.82, tensorboard==1.11.0 on a machine with 2080Ti GPUs. 
2. Our method does not rely on any new or deprecated features of any libraries, so there is no need to make an identical environment.
3. If you get any errors regarding tensorboard or tensorflow, you may simply delete the code related to tensorboard or SummaryWriter.


## Introduction

It is not easy to design and run Convolutional Neural Networks (CNNs) due to: 1) finding the optimal number of filters (i.e., the width) at each layer is tricky, given an architecture; and 2) the computational intensity of CNNs impedes the deployment on computationally limited devices. Oracle Pruning is designed to remove the unimportant filters from a well-trained CNN, which estimates the filters’ importance by ablating them in turn and evaluating the model, thus delivers high accuracy but suffers from intolerable time complexity, and requires a given resulting width but cannot automatically find it. To address these problems, we propose Approximated Oracle Filter Pruning (AOFP), which keeps searching for the least important filters in a binary search manner, makes pruning attempts by masking out filters randomly, accumulates the resulting errors, and finetunes the model via a multi-path framework. As AOFP enables simultaneous pruning on multiple layers, we can prune an existing very deep CNN with acceptable time cost, negligible accuracy drop, and no heuristic knowledge, or re-design a model which exerts higher accuracy and faster inference.

## Reproduce 65% pruning ratio of VGG on CIFAR-10.

1. Enter this directory.

2. Make a soft link to your CIFAR-10 directory. If the dataset is not found in the directory, it will be automatically downloaded.
```
ln -s YOUR_PATH_TO_CIFAR cifar10_data
```

3. Set the environment variables.
```
export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=0
```

4. Train the base model.
```
python train_base_model.py -a vc
```

5. Run AOFP. The pruned weights will be saved to "aofp_models/vc_train/finish_pruned.hdf5" and automatically tested.
```
python aofp/do_aofp.py -a vc
```

6. Show the name and shape of weights in the pruned model.
```
python display_hdf5.py aofp_models/vc_train/finish_pruned.hdf5
```

## Reproduce 50% pruning ratio of ResNet-56 on CIFAR-10.

1. Enter this directory.

2. Make a soft link to your CIFAR-10 directory. If the dataset is not found in the directory, it will be automatically downloaded.
```
ln -s YOUR_PATH_TO_CIFAR cifar10_data
```

3. Set the environment variables.
```
export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=0
```

4. Train the base model.
```
python train_base_model.py -a src56
```

5. Run AOFP. The pruned weights will be saved to "aofp_models/src56_train/finish_pruned.hdf5" and automatically tested.
```
python aofp/do_aofp.py -a src56
```

6. Show the name and shape of weights in the pruned model.
```
python display_hdf5.py aofp_models/src56_train/finish_pruned.hdf5
```


## Contact
dxh17@mails.tsinghua.edu.cn

Google Scholar Profile: https://scholar.google.com/citations?user=CIjw0KoAAAAJ&hl=en

My open-sourced papers and repos: 

The **Structural Re-parameterization Universe**:

1. RepMLP (preprint, 2021) **MLP-style building block and Architecture**\
[RepMLPNet: Hierarchical Vision MLP with Re-parameterized Locality](https://arxiv.org/abs/2112.11081)\
[code](https://github.com/DingXiaoH/RepMLP).

2. RepVGG (CVPR 2021) **A super simple and powerful VGG-style ConvNet architecture**. Up to **84.16%** ImageNet top-1 accuracy!\
[RepVGG: Making VGG-style ConvNets Great Again](https://arxiv.org/abs/2101.03697)\
[code](https://github.com/DingXiaoH/RepVGG).

3. ResRep (ICCV 2021) **State-of-the-art** channel pruning (Res50, 55\% FLOPs reduction, 76.15\% acc)\
[ResRep: Lossless CNN Pruning via Decoupling Remembering and Forgetting](https://openaccess.thecvf.com/content/ICCV2021/papers/Ding_ResRep_Lossless_CNN_Pruning_via_Decoupling_Remembering_and_Forgetting_ICCV_2021_paper.pdf)\
[code](https://github.com/DingXiaoH/ResRep).

4. ACB (ICCV 2019) is a CNN component without any inference-time costs. The first work of our Structural Re-parameterization Universe.\
[ACNet: Strengthening the Kernel Skeletons for Powerful CNN via Asymmetric Convolution Blocks](http://openaccess.thecvf.com/content_ICCV_2019/papers/Ding_ACNet_Strengthening_the_Kernel_Skeletons_for_Powerful_CNN_via_Asymmetric_ICCV_2019_paper.pdf).\
[code](https://github.com/DingXiaoH/ACNet). 

5. DBB (CVPR 2021) is a CNN component with higher performance than ACB and still no inference-time costs. Sometimes I call it ACNet v2 because "DBB" is 2 bits larger than "ACB" in ASCII (lol).\
[Diverse Branch Block: Building a Convolution as an Inception-like Unit](https://arxiv.org/abs/2103.13425)\
[code](https://github.com/DingXiaoH/DiverseBranchBlock).

6. COMING SOON

7. COMING SOON

**Model compression and acceleration**:

1. (CVPR 2019) Channel pruning: [Centripetal SGD for Pruning Very Deep Convolutional Networks with Complicated Structure](http://openaccess.thecvf.com/content_CVPR_2019/html/Ding_Centripetal_SGD_for_Pruning_Very_Deep_Convolutional_Networks_With_Complicated_CVPR_2019_paper.html)\
[code](https://github.com/DingXiaoH/Centripetal-SGD)

2. (ICML 2019) Channel pruning: [Approximated Oracle Filter Pruning for Destructive CNN Width Optimization](http://proceedings.mlr.press/v97/ding19a.html)\
[code](https://github.com/DingXiaoH/AOFP)

3. (NeurIPS 2019) Unstructured pruning: [Global Sparse Momentum SGD for Pruning Very Deep Neural Networks](http://papers.nips.cc/paper/8867-global-sparse-momentum-sgd-for-pruning-very-deep-neural-networks.pdf)\
[code](https://github.com/DingXiaoH/GSM-SGD)

