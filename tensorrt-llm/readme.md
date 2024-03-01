tensorrt-llm 与fastertransformer的关系 ？  


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

如下为 FMHA 的详细信息，其中 MQA 为 Multi Query Attention，GQA 为 Group Query Attention。   


另外一个 Kernel 是 MMHA(Masked Multi-Head Attention)。FMHA 主要用在 context phase 阶段的计算，而 MMHA 主要提供 generation phase 阶段 attention 的加速，并提供了 Volta 和之后架构的支持。相比 FastTransformer 的实现，TensorRT-LLM 有进一步优化，性能提升高达 2x。    


另外一个重要特性是量化技术，以更低精度的方式实现推理加速。常用量化方式主要分为 PTQ(Post Training Quantization)和 QAT(Quantization-aware Training)，对于 TensorRT-LLM 而言，这两种量化方式的推理逻辑是相同的。对于 LLM 量化技术，一个重要的特点是算法设计和工程实现的 co-design，即对应量化方法设计之初，就要考虑硬件的特性。否则，有可能达不到预期的推理速度提升。  


TensorRT 中 PTQ 量化步骤一般分为如下几步，首先对模型做量化，然后对权重和模型转化成 TensorRT-LLM 的表示。对于一些定制化的操作，还需要用户自己编写 kernels。常用的 PTQ 量化方法包括 INT8 weight-only、SmoothQuant、GPTQ 和 AWQ，这些方法都是典型的 co-design 的方法。


INT8 weight-only 直接把权重量化到 INT8，但是激活值还是保持为 FP16。该方法的好处就是模型存储2x减小，加载 weights 的存储带宽减半，达到了提升推理性能的目的。这种方式业界称作 W8A16，即权重为 INT8，激活值为 FP16/BF16——以 INT8 精度存储，以 FP16/BF16 格式计算。该方法直观，不改变 weights，容易实现，具有较好的泛化性能。  


第二个量化方法是 SmoothQuant，该方法是 NVIDIA 和社区联合设计的。它观察到权重通常服从高斯分布，容易量化，但是激活值存在离群点，量化比特位利用不高。


