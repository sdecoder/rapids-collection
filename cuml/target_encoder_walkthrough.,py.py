'''
In this notebook, we walk through the design of target encoding. We start with a motivating example, criteo dataset, to show why target encoding is preferred over one hot encoding and label encoding. The concepts and optimizations of target encoding are introduced step by step. The key takeaway is that target encoding differs from traditional sklearn style encoders in the following aspects:

The ground truth column target is used as input for encoding.
The training data and test data are transformed differently.
Multi-column joint transformation is supported by target encoding.

'''


import os
GPU_id = '0,1,2,3'
os.environ['CUDA_VISIBLE_DEVICES'] = GPU_id
num_gpus = len(GPU_id.split(','))
