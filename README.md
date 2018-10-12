# graphConv.py

Very basic implementation of two graph convolution layers.

# Graph convolution without self connection

```
def convolutionGraph(inputs,
                     num_outputs,
                     glap,
                     activation_fn=nn.relu,
                     weights_initializler=initializers.xavier_initializer(),
                     biases_initializer=init_ops.zeros_initializer(),
                     reuse=None,
                     scope=None)
```

convlolutionGraph() implements a graph convolution layer defined by Kipf et al.
- `inputs` is a 2d tensor that goes into the layer.
- `num_outputs` specifies the number of channels wanted on the output tensor.
- `glap` is an instance of tf.SparseTensor that defines a graph laplacian matrix DAD.


