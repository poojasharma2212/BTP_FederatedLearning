
import tensorflow as tf
import torch
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
# create a tensor of size 2 x 4
input_var = torch.randn(4)
print(tf.stats.variance(input_var))
print(input_var)
