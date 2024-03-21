
核心思想：传统减少HBM的访问，将QKV切分为小块后放入SRAM中      
核心方法：tiling, recomputation      

softmax操作是row-wise的，即每行都算一次softmax，所以需要用分块计算softmax。
原始softmax数值不稳定，为了数值稳定性，FlashAttention采用safe softmax   
```
上溢出   
一种严重的误差是上溢出overflow：当数值非常大，超过了计算机的表示范围时，发生上溢出。
对于softmax 的分子，分母均会溢出。

下溢出   
一种严重的误差是下溢出underflow：当接近零的数字四舍五入为零时，发生下溢出。
 许多函数在参数为零和参数为一个非常小的正数时，行为是不同的。如：对数函数要求自变量大于零，除法中要求除数非零。
对于softmax 的分子，分母均会溢出。      

当所有的xi 都等于常数c 时，考虑 c 是一个非常大的负数（如-20000，或比如趋近负无穷)，
则exp^(xi) 趋于 0，此时 softmax分母趋于0，下溢出。   
```

FlashAttention算法的目标：在计算中减少显存占用，从O(N*N) 大小降低到线性，这样就可以把数据加载到SRAM中，提高IO速度。
解决方案：传统Attention在计算中需要用到Q，K，V去计算S，P两个矩阵，FlashAttention引入softmax中的统计量( , ℓ)，结合output O和在SRAM中的Q，K，V块进行计算。

https://zhuanlan.zhihu.com/p/669926191
