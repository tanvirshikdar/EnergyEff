# **EnergyEff**

EnergyEff is a Python script designed to optimize memory for deep neural networks. By optimizing energy usage, this tool enables training more extensive and deeper models with limited resources, making it easier to manage the energy consumption of deep learning models while maintaining high performance.

## **How It Works**

EnergyEff leverages memory optimization techniques while training deep neural networks to reduce energy consumption. By using the mirror\_stage attribute, the tool identifies points in the symbolic computation graph where intermediate results can be recomputed instead of stored, thus reducing memory consumption and energy use.

### 

### **Steps to Use EnergyEff:**

1. **Configure Your Network**:

   * Build your network using the **MXNet symbolic API**, just as you normally would.

2. **Provide Memory Hints**:

   * Mark the computation nodes where memory can be reused by setting the `mirror_stage='True'` attribute. These are the points where computations can be recomputed, saving energy and memory.

**Example:**

```python
sym._set_attr(mirror_stage='True')
```


3. **Apply Memory Optimization**:

   * Call `EnergyEff.search_plan` to generate a **memory-optimized symbolic graph** that uses less energy.

4. **Training the Model**:

   * Use the optimized network as usual for training.

```python
net_planned = EnergyEff.search_plan(net)
model = mx.FeedForward(net_planned, ...)
model.fit(...)
```

## **Writing Your Own Energy Optimizer**

MXNet supports symbolic graph attributes like `force_mirroring`, which indicate whether certain results can be recomputed instead of being stored in memory. You can write a custom energy-efficient memory allocator by managing these attributes effectively.

* **Key Attribute**: `sym._set_attr(force_mirroring='True')` marks results as re-computable.

You can create your own memory allocator by managing the **mirror stages** in the symbolic graph to save energy by reusing intermediate results.

---

## **Example: EnergyEff on LSTM**

### **LSTM Cell Symbol:**

This LSTM construction example uses **EnergyEff** to reduce energy consumption by setting **mirror stages**.


```python
import mxnet as mx

def lstm(num_hidden, indata, prev_state, param, seqidx, layeridx, dropout=0.):
    # Apply dropout if needed
    if dropout > 0.:
        indata = mx.sym.Dropout(data=indata, p=dropout)

    # LSTM gates and transformations
    i2h = mx.sym.FullyConnected(data=indata, weight=param.i2h_weight, bias=param.i2h_bias, num_hidden=4 * num_hidden)
    h2h = mx.sym.FullyConnected(data=prev_state.h, weight=param.h2h_weight, bias=param.h2h_bias, num_hidden=4 * num_hidden)

    # Gates computation and activation
    gates = i2h + h2h
    slice_gates = mx.sym.SliceChannel(gates, num_outputs=4, axis=1, squeeze_axis=1)  # 4 gates: input, forget, output, cell

    input_gate = mx.sym.sigmoid(slice_gates[0])
    forget_gate = mx.sym.sigmoid(slice_gates[1])
    output_gate = mx.sym.sigmoid(slice_gates[2])
    candidate_cell = mx.sym.tanh(slice_gates[3])

    # Next cell state and hidden state
    next_c = forget_gate * prev_state.c + input_gate * candidate_cell
    next_h = output_gate * mx.sym.tanh(next_c)

    # Setting mirror stage for memory optimization
    next_c._set_attr(mirror_stage=str(seqidx))
    next_h._set_attr(mirror_stage=str(seqidx))

    # Returning LSTM state
    return mx.rnn.LSTMState(c=next_c, h=next_h)
```


### **Unrolling the LSTM:**

```python
def lstm_unroll(num_lstm_layer, seq_len, input_size, num_hidden, ...):

    # Applying LSTM and decoder with energy optimization
    for seqidx in range(seq_len):
        ...

        for i in range(num_lstm_layer):
            next_state = lstm(num_hidden, hidden, last_states[i], param_cells[i], seqidx, i, dropout=dp)
            hidden = next_state.h
            last_states[i] = next_state
```
---

## **Energy Optimization Cost Comparison**

Use **`EnergyEff.search_plan()`** to compare energy costs before and after optimization.

```python
# Before optimization
old_cost = EnergyEff.get_cost(net, **input_shapes)

# After applying the energy plan
net_mem_planned = EnergyEff.search_plan(net, **input_shapes)
new_cost = EnergyEff.get_cost(net_mem_planned, **input_shapes)

print('Old energy cost=%d MB' % old_cost)
print('New energy cost=%d MB' % new_cost)
```

---

## **Advanced Configuration**

Fine-tune energy optimization by adjusting the following parameters:

* **`concat_decode`**: Controls whether to use a single softmax (may reduce memory but slightly decrease speed).

* **`use_loss`**: Whether to use `softmax_cross_entropy` loss function to further reduce memory and energy consumption during training.

* **`threshold`**: Set a memory allocation threshold to find the optimal energy usage plan using **heuristic search**.

### **Example: ResNet Optimization**

`layers = [3, 24, 36, 3]`

`batch_size = 32`

`net = get_symbol(layers)`

`dshape = (batch_size, 3, 224, 224)`

`# Apply energy optimization`

`net_mem_planned = EnergyEff.search_plan(net, data=dshape)`

`old_cost = EnergyEff.get_cost(net, data=dshape)`

`new_cost = EnergyEff.get_cost(net_mem_planned, data=dshape)`

`print('Old energy cost=%d MB' % old_cost)`

`print('New energy cost=%d MB' % new_cost)`

---

## **Energy Planning and Heuristic Search**

EnergyEff includes a **heuristic search** algorithm to find the best memory and energy plan for your network.

* **`make_mirror_plan()`**: Applies mirror stages based on a given **threshold** for energy allocation.

* **`search_plan()`**: Heuristically searches through possible energy plans to find the most efficient one based on the network configuration and available resources.

---

## **Conclusion**

EnergyEff allows you to significantly reduce the energy consumption of your deep neural networks while maintaining high performance. By intelligently managing memory and reusing computations, you can achieve **energy-efficient deep learning** without compromising model accuracy.
