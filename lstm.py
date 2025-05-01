"""
Example Memory Allocator for LSTM
This LSTM construction code is adapted from the MXNet example.
The `mirror_stage` attribute is set in line 46.
"""
import sys
import mxnet as mx
import numpy as np
from collections import namedtuple
import time
import math
import memonger

# Define LSTM state and parameter structures
LSTMState = namedtuple("LSTMState", ["c", "h"])
LSTMParam = namedtuple("LSTMParam", ["i2h_weight", "i2h_bias",
                                     "h2h_weight", "h2h_bias"])
LSTMModel = namedtuple("LSTMModel", ["rnn_exec", "symbol",
                                     "init_states", "last_states",
                                     "seq_data", "seq_labels", "seq_outputs",
                                     "param_blocks"])

def lstm(num_hidden, indata, prev_state, param, seqidx, layeridx, dropout=0.):
    """
    LSTM Cell symbol.
    
    This function defines the LSTM cell computation and applies dropout if needed.
    Mirror stages are set to optimize memory usage.
    """
    if dropout > 0.:
        indata = mx.sym.Dropout(data=indata, p=dropout)

    # Fully connected layer for input to hidden transformation (i2h)
    i2h = mx.sym.FullyConnected(data=indata,
                                weight=param.i2h_weight,
                                bias=param.i2h_bias,
                                num_hidden=num_hidden * 4,
                                name=f"t{seqidx}_l{layeridx}_i2h")

    # Fully connected layer for hidden to hidden transformation (h2h)
    h2h = mx.sym.FullyConnected(data=prev_state.h,
                                weight=param.h2h_weight,
                                bias=param.h2h_bias,
                                num_hidden=num_hidden * 4,
                                name=f"t{seqidx}_l{layeridx}_h2h")

    # Gates computation (input, forget, output, and in_transform)
    gates = i2h + h2h
    slice_gates = mx.sym.SliceChannel(gates, num_outputs=4,
                                      name=f"t{seqidx}_l{layeridx}_slice")

    in_gate = mx.sym.Activation(slice_gates[0], act_type="sigmoid")
    in_transform = mx.sym.Activation(slice_gates[1], act_type="tanh")
    forget_gate = mx.sym.Activation(slice_gates[2], act_type="sigmoid")
    out_gate = mx.sym.Activation(slice_gates[3], act_type="sigmoid")

    # LSTM cell state updates (next_c and next_h)
    next_c = (forget_gate * prev_state.c) + (in_gate * in_transform)
    next_h = out_gate * mx.sym.Activation(next_c, act_type="tanh")

    # Set mirror stage for memory optimization (this helps with memory reuse)
    next_c._set_attr(mirror_stage=str(seqidx))
    next_h._set_attr(mirror_stage=str(seqidx))

    return LSTMState(c=next_c, h=next_h)

def lstm_unroll(num_lstm_layer, seq_len, input_size,
                num_hidden, num_embed, num_label, dropout=0.,
                concat_decode=True, use_loss=False):
    """
    Unrolls the LSTM network for sequence processing.
    
    This function applies the LSTM over a sequence and optionally applies
    a final softmax output or concatenates the hidden states.
    """
    # Initialize the parameter symbols
    embed_weight = mx.sym.Variable("embed_weight")
    cls_weight = mx.sym.Variable("cls_weight")
    cls_bias = mx.sym.Variable("cls_bias")

    param_cells = []
    last_states = []
    
    # Initialize the parameters for each LSTM layer
    for i in range(num_lstm_layer):
        param_cells.append(LSTMParam(i2h_weight=mx.sym.Variable(f"l{i}_i2h_weight"),
                                     i2h_bias=mx.sym.Variable(f"l{i}_i2h_bias"),
                                     h2h_weight=mx.sym.Variable(f"l{i}_h2h_weight"),
                                     h2h_bias=mx.sym.Variable(f"l{i}_h2h_bias")))

        state = LSTMState(c=mx.sym.Variable(f"l{i}_init_c"),
                          h=mx.sym.Variable(f"l{i}_init_h"))
        last_states.append(state)
        
    assert(len(last_states) == num_lstm_layer)

    last_hidden = []
    for seqidx in range(seq_len):
        # Embedding layer for the input sequence
        with mx.AttrScope(ctx_group='embed'):
            data = mx.sym.Variable(f"t{seqidx}_data")
            hidden = mx.sym.FullyConnected(data=data,
                                           weight=embed_weight,
                                           no_bias=True,
                                           num_hidden=num_embed,
                                           name=f"t{seqidx}_embed")

        # Stack the LSTM layers
        for i in range(num_lstm_layer):
            dp = dropout if i > 0 else 0.  # Dropout for all layers except the first
            with mx.AttrScope(ctx_group=f'layer{i}'):
                next_state = lstm(num_hidden, indata=hidden,
                                  prev_state=last_states[i],
                                  param=param_cells[i],
                                  seqidx=seqidx, layeridx=i, dropout=dp)
                hidden = next_state.h
                last_states[i] = next_state

        # Decoder (final output layer)
        if dropout > 0.:
            hidden = mx.sym.Dropout(data=hidden, p=dropout)
        last_hidden.append(hidden)

    out_prob = []
    if not concat_decode:
        # If not concatenating, apply softmax to each output sequence
        for seqidx in range(seq_len):
            with mx.AttrScope(ctx_group='decode'):
                fc = mx.sym.FullyConnected(data=last_hidden[seqidx],
                                           weight=cls_weight,
                                           bias=cls_bias,
                                           num_hidden=num_label,
                                           name=f"t{seqidx}_cls")
                label = mx.sym.Variable(f"t{seqidx}_label")
                if use_loss:
                    sm = mx.sym.softmax_cross_entropy(fc, label, name=f"t{seqidx}_sm")
                else:
                    sm = mx.sym.SoftmaxOutput(data=fc, label=label, name=f"t{seqidx}_sm")
                out_prob.append(sm)
    else:
        # If concatenating, combine the sequence outputs and apply softmax
        with mx.AttrScope(ctx_group='decode'):
            concat = mx.sym.Concat(*last_hidden, dim=0)
            fc = mx.sym.FullyConnected(data=concat,
                                       weight=cls_weight,
                                       bias=cls_bias,
                                       num_hidden=num_label)
            label = mx.sym.Variable("label")
            if use_loss:
                sm = mx.sym.softmax_cross_entropy(fc, label, name="sm")
            else:
                sm = mx.sym.SoftmaxOutput(data=fc, label=label, name="sm")
            out_prob = [sm]

    # Block the last states to prevent backpropagation into them
    for i in range(num_lstm_layer):
        state = last_states[i]
        state = LSTMState(c=mx.sym.BlockGrad(state.c, name=f"l{i}_last_c"),
                          h=mx.sym.BlockGrad(state.h, name=f"l{i}_last_h"))
        last_states[i] = state

    # Return the full network as a group of output symbols
    unpack_c = [state.c for state in last_states]
    unpack_h = [state.h for state in last_states]
    list_all = out_prob + unpack_c + unpack_h
    return mx.sym.Group(list_all)

def is_param_name(name):
    """Check if the name corresponds to a parameter (weight, bias, etc.)"""
    return name.endswith("weight") or name.endswith("bias") or\
           name.endswith("gamma") or name.endswith("beta")

def get_input_shapes(rnn_sym, batch_size, num_hidden, input_size, seq_len):
    """Determine the input shapes for the LSTM model"""
    arg_names = rnn_sym.list_arguments()
    input_shapes = {}
    for name in arg_names:
        if name.endswith("init_c") or name.endswith("init_h"):
            input_shapes[name] = (batch_size, num_hidden)
        elif name.endswith("data"):
            input_shapes[name] = (batch_size, input_size)
        elif name == "label":
            input_shapes[name] = (batch_size * seq_len, )
        elif name.endswith("label"):
            input_shapes[name] = (batch_size, )
    return input_shapes

# Some configurations for the LSTM
concat_decode = False  # Use a single softmax (increase memory usage)
use_loss = True  # Use softmax cross entropy loss for training

batch_size = 64
seq_len = 1000
num_hidden = 1024
num_embed = 1024
input_size = 50
num_lstm_layer = 4
num_label = 5000

net = lstm_unroll(
    num_lstm_layer=num_lstm_layer,
    seq_len=seq_len, input_size=input_size,
    num_hidden=num_hidden, num_embed=num_embed,
    num_label=num_label,
    concat_decode=concat_decode,
    use_loss=use_loss)

ishapes = get_input_shapes(net,
                           batch_size=batch_size,
                           num_hidden=num_hidden,
                           input_size=input_size,
                           seq_len=seq_len)

# Memory optimization with memonger
net_mem_planned = memonger.search_plan(net, **ishapes)
old_cost = memonger.get_cost(net, **ishapes)
new_cost = memonger.get_cost(net_mem_planned, **ishapes)

print(f'Old feature map cost={old_cost} MB')
print(f'New feature map cost={new_cost} MB')
# The optimized network can now be fed into the MXNet training script.