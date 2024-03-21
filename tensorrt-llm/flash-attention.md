+ Execution Model   
GPUs have a massive number of threads to execute an operation (called a kernel).
`Each kernel loads inputs from HBM to registers and SRAM`, computes, then writes outputs to HBM.  

+ Performance characteristics  
Depending on the balance of computation and memory accesses, operations
can be classified as either compute-bound or memory-bound. This is commonly measured by the
arithmetic intensity [85], which is the `number of arithmetic operations per byte of memory access`.  

+ Compute-bound
the time taken by the operation is determined by how many arithmetic operations there
are, while time accessing HBM is much smaller. Typical examples are matrix multiply with large inner
dimension, and convolution with large number of channels.
含有大量内乘的矩阵乘法，含有大量通道的卷积    

+ Memory-bound
the time taken by the operation is determined by the number of memory accesses, while
time spent in computation is much smaller. Examples include most other operations: `elementwise (e.g.,
activation, dropout), and reduction (e.g., sum, softmax, batch norm, layer norm)`.

+ Kernel fusion    
The most common approach to accelerate memory-bound operations is kernel fusion: if there are multiple operations applied to the same input, the input can be loaded once from HBM, instead of multiple times for each operation. Compilers can automatically fuse many elementwise operations,However, in the context of model training, the intermediate values still need to be written to HBM to save for the backward pass, reducing the effectiveness of naive kernel fusion.


核心思想：传统减少HBM的访问，将QKV切分为小块后放入SRAM中      
核心方法：tiling, recomputation      

softmax操作是row-wise的，即每行都算一次softmax，所以需要用分块计算softmax。
原始softmax数值不稳定，为了数值稳定性，FlashAttention采用safe softmax   
```
上溢出   
一种严重的误差是上溢出overflow：当数值非常大，超过了计算机的表示范围时，发生上溢出。

当所有的xi 都等于常数c时，考虑 c 是一个非常大的正数（如20000，或比如趋近正无穷)，
则exp^(xi) 趋于 无穷大，分子分母均会上溢出。

下溢出   
一种严重的误差是下溢出underflow：当接近零的数字四舍五入为零时，发生下溢出。
 许多函数在参数为零和参数为一个非常小的正数时，行为是不同的。如：对数函数要求自变量大于零，除法中要求除数非零。   

当所有的xi 都等于常数c时，考虑 c 是一个非常大的负数（如-20000，或比如趋近负无穷)，
则exp^(xi) 趋于 0，此时 softmax分母趋于0，下溢出。   
```

FlashAttention算法的目标：在计算中减少显存占用，从O(N*N) 大小降低到线性，这样就可以把数据从HBM 加载到SRAM中，提高IO速度。  

解决方案：传统Attention在计算中需要用到Q，K，V去计算S，P两个矩阵，FlashAttention引入softmax中的统计量( , ℓ)，结合output O和在SRAM中的Q，K，V块进行计算。

https://zhuanlan.zhihu.com/p/669926191
