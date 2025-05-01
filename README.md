# EnergyEff

**EnergyEff** is a Python script designed to optimize memory for deep neural networks. By optimizing energy usage, this tool enables training more extensive and deeper models with limited resources, making it easier to manage the energy consumption of deep learning models while maintaining high performance.

## **How It Works**

EnergyEff leverages memory optimization techniques while training deep neural networks to reduce energy consumption. By using the `mirror_stage` attribute, the tool identifies points in the symbolic computation graph where intermediate results can be recomputed instead of stored, thus reducing memory consumption and energy use.

## **Steps to Use EnergyEff:**

### **1. Configure Your Network:**
Build your network using the MXNet symbolic API, just as you normally would.

### **2. Provide Memory Hints:**
Mark the computation nodes where memory can be reused by setting the `mirror_stage='True'` attribute. These are the points where computations can be recomputed, saving energy and memory.

#### Example:

```python
sym._set_attr(mirror_stage='True')
