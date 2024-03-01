tensorrt-llm 与fastertransformer的关系    

TensorRT-LLM 是 NVIDIA 用于做 LLM（Large Language Model）的可扩展推理方案。   
+ 该方案是基于 TensorRT 深度学习编译框架来构建、编译并执行计算图，      
+ 借鉴了许多 FastTransformer 中高效的 Kernels 实现，   
+ 利用了 NCCL 完成设备之间的通讯。
+ 考虑到技术的发展和需求的差异，开发者还可以定制算子来满足定制需求，比如基于 cutlass 开发定制 GEMM       
