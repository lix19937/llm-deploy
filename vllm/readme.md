## 主要是解决 通量的   


采用了 PagedAttention，可以有效管理 attention 的 keys、values    
吞吐量（通量）最多可以达到 huggingface 实现的24倍，并且不需要对模型结构进行任何的改变

State-of-the-art serving **throughput**         
Efficient management of attention key and value memory with PagedAttention    
Continuous batching of incoming requests   
Fast model execution with CUDA/HIP graph    
Quantization: GPTQ, AWQ, SqueezeLLM, FP8 KV Cache    
Optimized CUDA kernels    


### 背景   
+ LLM 的推理，最大的瓶颈在于显存。
+ 自回归模型的 keys 和 values 通常被称为 KV cache，这些 tensors 会存在 GPU 的显存中，用于生成下一个 token。这些 KV cache 都很大，并且大小是动态变化的，难以预测。已有的系统中，由于显存碎片和过度预留，浪费了60%-80%的显存。

### 实现  
受到操作系统中，虚拟内存和分页经典思想的启发 PagedAttention 允许在不连续的内存空间中存储连续的 keys 和 values。   
具体来说，PagedAttention 会将每个序列的 KV cache 划分为块，每个块包含固定数量 tokens 的 keys 和 values。 在注意力计算过程中，PagedAttention 内核有效地识别并获取这些块。
分块之后，这些 KV cache 不再需要连续的内存，从而可以像在操作系统的虚拟内存中一样，更灵活地对这些 KV cache 进行管理。
PagedAttention 对于显存的利用接近理论上的最优值（浪费比例低于4%）。通过对显存进行更好的管理，可以使得单次可以使用更大的 batch size，从而进一步利用 GPU 的并行计算能力。

### memory sharing   
+ memory sharing 是 PagedAttention 的另一个关键特性。  
+ 当用单个 prompt 产出多个不同的序列时，可以共享计算量和显存。  
+ 通过将不同序列的 logical blocks 映射到同一个 physical blocks，可以实现显存共享。    
+ 为了保证共享的安全性，对于 physical blocks 的引用次数进行统计，并实现了 **Copy-on-Write** 机制。这种内存共享机制，可以大幅降低复杂采样算法对于显存的需求（最高可下降55%），从而可以提升2.2倍的吞吐量。   

https://zhuanlan.zhihu.com/p/642802585     
https://blog.vllm.ai/2023/06/20/vllm.html   
