import tensorflow as tf
from tensorflow.contrib.framework.python.ops import add_arg_scope
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import nn

# We expect Chebychev approx. of graph Laplacian D^(-1/2)(A+I)D^(-1/2)
# This is the implementation suggested by Kipf et al.
# They use glorot but here we use xaxier as default.

def convolutionGraph(inputs,
                     num_outputs,
                     glap,
                     activation_fn=nn.relu,
                     weights_initializler=initializers.xavier_initializer(),
                     biases_initializer=init_ops.zeros_initializer(),
                     reuse=None,
                     scope=None):
    
    """Addes an graph convolution followed by an optional batch_norm layer.
    graph convolution layer creates a variable called `weights`, represeting the
    convolutional kernel, that is convolved with the 'inputs' over graph defined
    by `glap`. If a `normalizer_fn` is provided (such as `batch_norm`), it is 
    then applied. Otherwise, if `normalizer_fn` is None and a `biases_initializer`
    is provided then a `biases` variable would be created and added the activations.
    Finally, if `activation_fn` is not `None`, it is applied to the activations as well"""
    
    input_rank = inputs.get_shape().ndims
    if not input_rank == 3:
        raise ValueError('Graph Convolution not supported for input with rank', input_rank)
    
    input_dim = int(inputs[0].shape[-1])

    with variable_scope.variable_scope(scope, 'GConv', reuse=reuse):
        
        w = tf.get_variable("gweights", shape=[input_dim, num_outputs], initializer=weights_initializler)
        b = tf.get_variable("gbias", shape=[num_outputs], initializer=biases_initializer)
        
        # tf.sparse_tensor_dense_matmul loses the dimension information 
        # when used in map_fn. It might be a bug. We are using dynamic slicing instead.
        inputs_unstacked = tf.unstack(inputs)
        
        # Apply this function to each sample
        def fn(x_slice):
            xw = tf.matmul(x_slice, w)
            DADxw = tf.sparse_tensor_dense_matmul(glap, xw)
            return DADxw+b
        
        outputs = tf.stack([fn(i) for i in inputs_unstacked])
        
    return activation_fn(outputs)

# We expect Chebychev approx. of graph Laplacian D^(-1/2)(A+I)D^(-1/2)
# This is the implementation suggested by Kipf et al.
# They use glorot but here we use xaxier as default.

def convolutionGraph_sc(inputs,
                     num_outputs,
                     glap,
                     activation_fn=nn.relu,
                     weights_initializler=initializers.xavier_initializer(),
                     biases_initializer=init_ops.zeros_initializer(),
                     reuse=None,
                     scope=None):
    
    """Addes an graph convolution followed by an optional batch_norm layer.
    graph convolution layer creates a variable called `weights`, represeting the
    convolutional kernel, that is convolved with the 'inputs' over graph defined
    by `glap`. If a `normalizer_fn` is provided (such as `batch_norm`), it is 
    then applied. Otherwise, if `normalizer_fn` is None and a `biases_initializer`
    is provided then a `biases` variable would be created and added the activations.
    Finally, if `activation_fn` is not `None`, it is applied to the activations as well"""
    
    input_rank = inputs.get_shape().ndims
    if not input_rank == 3:
        raise ValueError('Graph Convolution not supported for input with rank', input_rank)
    
    input_dim = int(inputs[0].shape[-1])

    with variable_scope.variable_scope(scope, 'GConv', reuse=reuse):
        
        w1 = tf.get_variable("gweights1", shape=[input_dim, num_outputs], initializer=weights_initializler)
        w2 = tf.get_variable("gweights2", shape=[input_dim, num_outputs], initializer=weights_initializler)
        b = tf.get_variable("gbias", shape=[num_outputs], initializer=biases_initializer)
        
        # tf.sparse_tensor_dense_matmul loses the dimension information 
        # when used in map_fn. It might be a bug. We are using dynamic slicing instead.
        inputs_unstacked = tf.unstack(inputs)
        
        # Apply this function to each sample
        def fn(x_slice):
            xw1 = tf.matmul(x_slice, w1)
            xw2 = tf.matmul(x_slice, w2)
            DADxw = tf.sparse_tensor_dense_matmul(glap, xw1)
            return DADxw+xw2+b
        
        outputs = tf.stack([fn(i) for i in inputs_unstacked])
        
    return activation_fn(outputs)