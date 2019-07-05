# Approximated Oracle Filter Pruning for Destructive CNN Width Optimization

This repository contains the codes for the following ICML-2019 paper 

[Approximated Oracle Filter Pruning for Destructive CNN Width Optimization](http://proceedings.mlr.press/v97/ding19a.html).

The codes are based on Tensorflow 1.11.

I will re-implement this using pytorch after finishing my pytorch experiment platform. Maybe in two months.

Citation:

    @inproceedings{ding2019approximated,
    title={Approximated Oracle Filter Pruning for Destructive CNN Width Optimization},
    author={Ding, Xiaohan and Ding, Guiguang and Guo, Yuchen and Han, Jungong and Yan, Chenggang},
    booktitle={International Conference on Machine Learning},
    pages={1607--1616},
    year={2019}

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




