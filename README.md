# Approximated Oracle Filter Pruning for Destructive CNN Width Optimization

I will release PyTorch codes in two weeks.

This repository contains the codes for the following ICML-2019 paper 

[Approximated Oracle Filter Pruning for Destructive CNN Width Optimization](http://proceedings.mlr.press/v97/ding19a.html).

The codes are based on Tensorflow 1.11.

Citation:

    @inproceedings{ding2019approximated,
    title={Approximated Oracle Filter Pruning for Destructive CNN Width Optimization},
    author={Ding, Xiaohan and Ding, Guiguang and Guo, Yuchen and Han, Jungong and Yan, Chenggang},
    booktitle={International Conference on Machine Learning},
    pages={1607--1616},
    year={2019}
    }

## Introduction

It is not easy to design and run Convolutional Neural Networks (CNNs) due to: 1) finding the optimal number of filters (i.e., the width) at each layer is tricky, given an architecture; and 2) the computational intensity of CNNs impedes the deployment on computationally limited devices. Oracle Pruning is designed to remove the unimportant filters from a well-trained CNN, which estimates the filters’ importance by ablating them in turn and evaluating the model, thus delivers high accuracy but suffers from intolerable time complexity, and requires a given resulting width but cannot automatically find it. To address these problems, we propose Approximated Oracle Filter Pruning (AOFP), which keeps searching for the least important filters in a binary search manner, makes pruning attempts by masking out filters randomly, accumulates the resulting errors, and finetunes the model via a multi-path framework. As AOFP enables simultaneous pruning on multiple layers, we can prune an existing very deep CNN with acceptable time cost, negligible accuracy drop, and no heuristic knowledge, or re-design a model which exerts higher accuracy and faster inference.


## Example Usage
  
This repo holds the example code for pruning VGG on CIFAR-10. 

1. Install Tensorflow-gpu-1.11

2. Prepare the CIFAR-10 dataset in tfrecord format. Please follow https://github.com/tensorflow/models/blob/master/research/slim/datasets/download_and_convert_cifar10.py, download the CIFAR-10 dataset, convert it to tfrecord format, rename the two output files as train.tfrecords and validation.tfrecords, and modify the value of CIFAR10_DATASET_DIR in bc_params_factory.py.

3. Train a VGG on CIFAR-10. Then evaluate the model.

```
python aofp/aofp_standalone.py train
python aofp/aofp_standalone.py eval vc_scratch_savedweights.hdf5
```

4. Get a series of models with decreasing number of FLOPs by pruning via AOFP.

```
python aofp/aofp_standalone.py prune
```

5. Check the accuracy of the finetuned model at each pruning iteration.

```
cat bds_overall_logs.txt | grep fted
```

## Acknowledgement

The backbone of the codes is based on the Tensorflow benchmarks codes: https://github.com/tensorflow/benchmarks/tree/master/scripts/tf_cnn_benchmarks

## Contact
dxh17@mails.tsinghua.edu.cn

Google Scholar Profile: https://scholar.google.com/citations?user=CIjw0KoAAAAJ&hl=en

My open-sourced papers and repos:

CNN component (ICCV 2019): [ACNet: Strengthening the Kernel Skeletons for Powerful CNN via Asymmetric Convolution Blocks](http://openaccess.thecvf.com/content_ICCV_2019/papers/Ding_ACNet_Strengthening_the_Kernel_Skeletons_for_Powerful_CNN_via_Asymmetric_ICCV_2019_paper.pdf) (https://github.com/DingXiaoH/ACNet)

Channel pruning (CVPR 2019): [Centripetal SGD for Pruning Very Deep Convolutional Networks with Complicated Structure](http://openaccess.thecvf.com/content_CVPR_2019/html/Ding_Centripetal_SGD_for_Pruning_Very_Deep_Convolutional_Networks_With_Complicated_CVPR_2019_paper.html) (https://github.com/DingXiaoH/Centripetal-SGD)

Channel pruning (ICML 2019): [Approximated Oracle Filter Pruning for Destructive CNN Width Optimization](http://proceedings.mlr.press/v97/ding19a.html) (https://github.com/DingXiaoH/AOFP)

Unstructured pruning (NeurIPS 2019): [Global Sparse Momentum SGD for Pruning Very Deep Neural Networks](http://papers.nips.cc/paper/8867-global-sparse-momentum-sgd-for-pruning-very-deep-neural-networks.pdf) (https://github.com/DingXiaoH/GSM-SGD)




