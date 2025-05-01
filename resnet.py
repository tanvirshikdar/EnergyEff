import sys
import os
import math
import mxnet as mx
import memonger

def ConvModule(sym, num_filter, kernel, pad=(0, 0), stride=(1, 1), fix_gamma=True):
    """
    Creates a convolutional block with Batch Normalization and Leaky ReLU activation.

    Parameters
    ----------
    sym : mxnet.symbol
        Input symbol to the convolutional layer.
    num_filter : int
        Number of output filters for the convolutional layer.
    kernel : tuple
        Size of the convolutional kernel (height, width).
    pad : tuple, optional
        Padding for the convolution (default is (0, 0)).
    stride : tuple, optional
        Stride of the convolution (default is (1, 1)).
    fix_gamma : bool, optional
        Whether to fix gamma in Batch Normalization (default is True).
    
    Returns
    -------
    mxnet.symbol
        Output symbol after applying convolution, batch normalization, and activation.
    """
    conv = mx.sym.Convolution(data=sym, kernel=kernel, stride=stride, pad=pad, num_filter=num_filter)
    bn = mx.sym.BatchNorm(data=conv, fix_gamma=fix_gamma)
    act = mx.sym.LeakyReLU(data=bn, act_type="leaky")  # Use LeakyReLU with the same memory as standard activations
    return act


def ResModule(sym, base_filter, stage, layer, fix_gamma=True):
    """
    Creates a residual block with three convolutional layers.
    
    Parameters
    ----------
    sym : mxnet.symbol
        Input symbol to the residual block.
    base_filter : int
        Base number of filters for the convolutional layers.
    stage : int
        The current stage of the residual network.
    layer : int
        The layer within the current stage.
    fix_gamma : bool, optional
        Whether to fix gamma in Batch Normalization (default is True).
    
    Returns
    -------
    mxnet.symbol
        Output symbol after applying the residual block.
    """
    num_f = base_filter * int(math.pow(2, stage))
    stride = 2 if stage != 0 and layer == 0 else 1  # Apply stride of 2 on the first layer of each stage
    
    # Apply 1x1, 3x3, 1x1 convolutions
    conv1 = ConvModule(sym, num_f, kernel=(1, 1), pad=(0, 0), stride=(1, 1), fix_gamma=fix_gamma)
    conv2 = ConvModule(conv1, num_f, kernel=(3, 3), pad=(1, 1), stride=(stride, stride), fix_gamma=fix_gamma)
    conv3 = ConvModule(conv2, num_f * 4, kernel=(1, 1), pad=(0, 0), stride=(1, 1), fix_gamma=fix_gamma)

    if layer == 0:
        sym = ConvModule(sym, num_f * 4, kernel=(1, 1), pad=(0, 0), stride=(stride, stride), fix_gamma=fix_gamma)

    # Skip connection (residual connection)
    sum_sym = sym + conv3
    sum_sym._set_attr(mirror_stage='True')  # Optimize memory by setting mirror stage

    return sum_sym


def get_symbol(layers):
    """
    Builds a 4-stage residual network.

    Parameters
    ----------
    layers : list
        A list of integers specifying the number of layers per stage.

    Returns
    -------
    mxnet.symbol
        The complete symbolic representation of the network.
    """
    assert len(layers) == 4, "The layers list must have 4 stages."

    base_filter = 64
    data = mx.sym.Variable(name='data')

    # Initial convolution layer and max-pooling
    conv1 = ConvModule(data, base_filter, kernel=(7, 7), pad=(3, 3), stride=(2, 2))
    mp1 = mx.sym.Pooling(data=conv1, pool_type="max", kernel=(3, 3), stride=(2, 2))
    
    sym = mp1
    # Add residual blocks for each stage
    for j in range(len(layers)):
        for i in range(layers[j]):
            sym = ResModule(sym, base_filter, j, i)

    # Final pooling, flattening, and fully connected layer for classification
    avg = mx.sym.Pooling(data=sym, kernel=(7, 7), stride=(1, 1), name="global_pool", pool_type='avg')
    flatten = mx.sym.Flatten(data=avg, name='flatten')
    fc1 = mx.sym.FullyConnected(data=flatten, num_hidden=1000, name='fc1')
    
    # Softmax output layer
    net = mx.sym.SoftmaxOutput(data=fc1, name='softmax')
    
    return net


def calculate_memory_cost(net, dshape):
    """
    Calculate and compare memory cost before and after optimization.

    Parameters
    ----------
    net : mxnet.symbol
        The network symbol.
    dshape : tuple
        The shape of the input data.

    Returns
    -------
    None
    """
    # Memory optimization using memonger
    net_mem_planned = memonger.search_plan(net, data=dshape)
    
    # Calculate memory cost before and after optimization
    old_cost = memonger.get_cost(net, data=dshape)
    new_cost = memonger.get_cost(net_mem_planned, data=dshape)
    
    print(f"Old feature map cost = {old_cost} MB")
    print(f"New feature map cost = {new_cost} MB")


# Configuration for the ResNet
layers = [3, 24, 36, 3]  # Number of layers in each of the 4 stages
batch_size = 32
dshape = (batch_size, 3, 224, 224)

# Generate the network symbol
net = get_symbol(layers)

# Calculate and print the memory usage before and after optimization
calculate_memory_cost(net, dshape)