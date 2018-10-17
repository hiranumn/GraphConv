# graphConv.py

Very basic implementation of two graph convolution layers.

# Code
### Graph convolution without self connection:

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

### Graph convolution with self connection:

```
def convolutionGraph_sc(inputs,
                     num_outputs,
                     glap,
                     activation_fn=nn.relu,
                     weights_initializler=initializers.xavier_initializer(),
                     biases_initializer=init_ops.zeros_initializer(),
                     reuse=None,
                     scope=None):
```

convlolutionGraph_sc() implements a graph convolution layer defined by Kipf et al, **except that self-connection of nodes are allowed.**
- `inputs` is a 2d tensor that goes into the layer.
- `num_outputs` specifies the number of channels wanted on the output tensor.
- `glap` is an instance of tf.SparseTensor that defines a graph laplacian matrix DAD.

### inits.py:
This file contains 4 common initialization methods for network weights, i.e., `uniform()`, `glorot()`, `zeros()`, and `ones()`.
We currently do not use it.

### utils.py:
This file contains
- Two versions of DataFeeder instances.
- Helper functions for data processing. See ipynb/DataProcessing.ipynb for how to use them.
- A class definition for `intx`.

# Test Data Files
We have ppi matrix extracted from BioGrid in March 2018. This defines a graph.
We also have TCGA RNA-seq data for approx 9000 samples. This is the data that are getting convolved over the graph.
You can download datafiles from here.

# Models
See runningGraphConv.py for implemention and training of the models. It is should be straight forward.

For any question, e-mail me at hiranumn at cs dot washington dot edu.
