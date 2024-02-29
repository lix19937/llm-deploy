## 主要是解决 通量的   


采用了 PagedAttention，可以有效管理 attention 的 keys、values    
吞吐量（通量）最多可以达到 huggingface 实现的24倍，并且不需要对模型结构进行任何的改变

State-of-the-art serving **throughput**         
Efficient management of attention key and value memory with PagedAttention    
Continuous batching of incoming requests   
Fast model execution with CUDA/HIP graph    
Quantization: GPTQ, AWQ, SqueezeLLM, FP8 KV Cache    
Optimized CUDA kernels    


https://zhuanlan.zhihu.com/p/642802585     
https://blog.vllm.ai/2023/06/20/vllm.html   
