tensorrt-llm 与fastertransformer的关系    

TensorRT-LLM 是 NVIDIA 用于做 LLM（Large Language Model）的可扩展推理方案。   
+ 该方案是基于 TensorRT 深度学习编译框架来构建、编译并执行计算图，      
+ 借鉴了许多 FastTransformer 中高效的 Kernels 实现，   
+ 利用了 NCCL 完成设备之间的通讯。
+ 考虑到技术的发展和需求的差异，开发者还可以定制算子来满足定制需求，比如基于 cutlass 开发定制 GEMM       


除了绿色 TensorRT 编译部分和一些涉及硬件信息的 kernels 外，其他部分都是开源的     
![v2-7353108f2e22fea040d375925ac73a1b_r](https://github.com/lix19937/llm-deploy/assets/38753233/e832b292-7445-4cbb-95fa-503f41a57ada)
 
TensorRT-LLM 还提供了类似于 Pytorch 的 API 来降低开发者的学习成本，并提供了许多预定义好的模型供用户使用。   
![v2-7353108f2e22fea040d375925ac73a1b_r](https://github.com/lix19937/llm-deploy/assets/38753233/45792237-010d-40ae-acfa-fc1157b95219)

考虑到大语言模型比较大，有可能单卡放不下，需要多卡甚至多机推理，因此 TensorRT-LLM 还提供了 Tensor Parallelism 和 Pipeline Parallelism 两种并行机制来支持多卡或多机推理。   


TensorRT-LLM 默认采用 FP16/BF16 的精度推理，并且可以利用业界的量化方法，使用硬件吞吐更高的低精度推理进一步推升推理性能。    


另外一个特性就是 FMHA(fused multi-head attention) kernel 的实现。由于 Transformer 中最为耗时的部分是 self-attention 的计算，因此官方设计了 FMHA 来优化 self-attention 的计算，并提供了累加器分别为 fp16 和 fp32 不同的版本。另外，除了速度上的提升外，对内存的占用也大大降低。我们还提供了基于 flash attention 的实现，可以将 sequence-length 扩展到任意长度。

