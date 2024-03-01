tensorrt-llm 与fastertransformer的关系 ？  


TensorRT-LLM 是 NVIDIA 用于做 LLM（Large Language Model）的可扩展推理方案          
+ 该方案是基于 TensorRT 深度学习编译框架来构建、编译并执行计算图          
+ 借鉴了许多 FastTransformer 中高效的 Kernels 实现      
+ 利用了 NCCL 完成设备之间的通讯      
+ 考虑到技术的发展和需求的差异，开发者还可以定制算子来满足定制需求，比如基于 cutlass 开发定制 GEMM       

------- 

除了绿色 TensorRT 编译部分和一些涉及硬件信息的 kernels 外，其他部分都是开源的     
![v2-7353108f2e22fea040d375925ac73a1b_r](https://github.com/lix19937/llm-deploy/assets/38753233/e832b292-7445-4cbb-95fa-503f41a57ada)

-------   
TensorRT-LLM 还提供了类似于 Pytorch 的 API 来降低开发者的学习成本，并提供了许多预定义好的模型供用户使用。   
![v2-7353108f2e22fea040d375925ac73a1b_r](https://github.com/lix19937/llm-deploy/assets/38753233/45792237-010d-40ae-acfa-fc1157b95219)

-------   
考虑到大语言模型比较大，有可能单卡放不下，需要多卡甚至多机推理，因此 TensorRT-LLM 还提供了 Tensor Parallelism 和 Pipeline Parallelism 两种并行机制来支持多卡或多机推理。 ![mgmn](https://github.com/lix19937/llm-deploy/assets/38753233/f09de4e5-d167-4ba5-81a9-d8a087a1b66d)     
  

-------   
TensorRT-LLM 默认采用 FP16/BF16 的精度推理，并且可以利用业界的量化方法，使用硬件吞吐更高的低精度推理进一步推升推理性能。    
![precision](https://github.com/lix19937/llm-deploy/assets/38753233/f315a3fe-5de9-43de-a4fa-a346068a15ce)

-------   
另外一个特性就是 FMHA(fused multi-head attention) kernel 的实现。由于 Transformer 中最为耗时的部分是 self-attention 的计算，因此官方设计了 FMHA 来优化 self-attention 的计算，并提供了累加器分别为 fp16 和 fp32 不同的版本。另外，除了速度上的提升外，对内存的占用也大大降低。我们还提供了基于 flash attention 的实现，可以将 sequence-length 扩展到任意长度。
![fmha](https://github.com/lix19937/llm-deploy/assets/38753233/2fc2682b-6a38-4fa3-a6b1-0e454fa2f89c)

-------   
如下为 FMHA 的详细信息，其中 MQA 为 Multi Query Attention，GQA 为 Group Query Attention。   
![fmha-2](https://github.com/lix19937/llm-deploy/assets/38753233/37886aca-afba-4aad-b966-f16188f88dd9)

-------   
另外一个 Kernel 是 MMHA(Masked Multi-Head Attention)。FMHA 主要用在 context phase 阶段的计算，而 MMHA 主要提供 generation phase 阶段 attention 的加速，并提供了 Volta 和之后架构的支持。相比 FastTransformer 的实现，TensorRT-LLM 有进一步优化，性能提升高达 2x。    
![mmha](https://github.com/lix19937/llm-deploy/assets/38753233/4f01889a-d50f-49f0-8bb2-113b938a20ff)

-------   
另外一个重要特性是量化技术，以更低精度的方式实现推理加速。常用量化方式主要分为 PTQ(Post Training Quantization)和 QAT(Quantization-aware Training)，对于 TensorRT-LLM 而言，这两种量化方式的推理逻辑是相同的。对于 LLM 量化技术，一个重要的特点是算法设计和工程实现的 co-design，即对应量化方法设计之初，就要考虑硬件的特性。否则，有可能达不到预期的推理速度提升。  
![qat](https://github.com/lix19937/llm-deploy/assets/38753233/e57bb8ac-fa5d-4592-a17c-984e5e510e89)

-------   
TensorRT 中 PTQ 量化步骤一般分为如下几步，首先对模型做量化，然后对权重和模型转化成 TensorRT-LLM 的表示。对于一些定制化的操作，还需要用户自己编写 kernels。常用的 PTQ 量化方法包括 INT8 weight-only、SmoothQuant、GPTQ 和 AWQ，这些方法都是典型的 co-design 的方法。
![ptq](https://github.com/lix19937/llm-deploy/assets/38753233/d3f4ea46-ae70-4be7-805c-046a1f47a40f)

-------   
INT8 weight-only 直接把权重量化到 INT8，但是激活值还是保持为 FP16。该方法的好处就是模型存储2x减小，加载 weights 的存储带宽减半，达到了提升推理性能的目的。这种方式业界称作 W8A16，即权重为 INT8，激活值为 FP16/BF16——以 INT8 精度存储，以 FP16/BF16 格式计算。该方法直观，不改变 weights，容易实现，具有较好的泛化性能。  
![int8-w-o](https://github.com/lix19937/llm-deploy/assets/38753233/2a099933-e92e-4d66-8024-0fbd142a3385)

-------   
第二个量化方法是 SmoothQuant，该方法是 NVIDIA 和社区联合设计的。它观察到权重通常服从高斯分布，容易量化，但是激活值存在离群点，量化比特位利用不高。
![smooth-q](https://github.com/lix19937/llm-deploy/assets/38753233/ae2b3c11-2e5d-4458-85f0-a1e55ae9c439)


-------   
SmoothQuant 通过先对激活值做平滑操作即除以一个scale将对应分布进行压缩，同时为了保证等价性，需要对权重乘以相同的 scale。之后，权重和激活都可以量化。对应的存储和计算精度都可以是 INT8 或者 FP8，可以利用 INT8 或者 FP8 的 TensorCore 进行计算。在实现细节上，权重支持 Per-tensor 和 Per-channel 的量化，激活值支持 Per-tensor 和 Per-token 的量化。
![smooth-q-2](https://github.com/lix19937/llm-deploy/assets/38753233/b944fc71-abde-4e05-8bab-adbf89e3093b)


-------   
第三个量化方法是 GPTQ，一种逐层量化的方法，通过最小化重构损失来实现。GPTQ 属于 weight-only 的方式，计算采用 FP16 的数据格式。该方法用在量化大模型时，由于量化本身开销就比较大，所以作者设计了一些 trick 来降低量化本身的开销，比如 Lazy batch-updates 和以相同顺序量化所有行的权重。GPTQ 还可以与其他方法结合使用如 grouping 策略。并且，针对不同的情况，TensorRT-LLM 提供了不同的实现优化性能。具体地，对 batch size 较小的情况，用 cuda core 实现；相对地，batch size 较大时，采用 tensor core 实现。   
![gptq](https://github.com/lix19937/llm-deploy/assets/38753233/7d244815-d347-4c3f-b4ad-9b2638e1de33)


------   
第四种量化方式是 AWQ。该方法认为不是所有权重都是同等重要的，其中只有 0.1%-1% 的权重（salient weights）对模型精度贡献更大，并且这些权重取决于激活值分布而不是权重分布。该方法的量化过程类似于 SmoothQuant，差异主要在于 scale 是基于激活值分布计算得到的。   
![awq](https://github.com/lix19937/llm-deploy/assets/38753233/735c7595-70e7-4047-8a62-7818c470ac18)     

![awq-2](https://github.com/lix19937/llm-deploy/assets/38753233/4b14226e-637e-43e8-ac84-6d215a4c568b)


------   
除了量化方式之外，TensorRT-LLM 另外一个提升性能的方式是利用多机多卡推理。在一些场景中，大模型过大无法放在单个 GPU 上推理，或者可以放下但是影响了计算效率，都需要多卡或者多机进行推理。   
![mgmn-2](https://github.com/lix19937/llm-deploy/assets/38753233/a319e764-4a18-4c12-b973-47bad419475c)


-----------     
TensorRT-LLM 目前提供了两种并行策略，Tensor Parallelism 和 Pipeline Parallelism。TP 是垂直地分割模型然后将各个部分置于不同的设备上，这样会引入设备之间频繁的数据通讯，一般用于设备之间有高度互联的场景，如 NVLINK。另一种分割方式是横向切分，此时只有一个横前面，对应通信方式是点对点的通信，适合于设备通信带宽较弱的场景。    
![tp-pp](https://github.com/lix19937/llm-deploy/assets/38753233/e0161bac-4426-4614-8511-2603625958b2)


--------------   
最后一个要强调的特性是 In-flight batching。Batching 是提高推理性能一个比较常用的做法，但在 LLM 推理场景中，一个 batch 中每个 sample/request 的输出长度是无法预测的。如果按照静态batching的方法，一个batch的时延取决于 sample/request 中输出最长的那个。因此，虽然输出较短的 sample/request 已经结束，但是并未释放计算资源，其时延与输出最长的那个 sample/request 时延相同。In-flight batching 的做法是在已经结束的 sample/request 处插入新的 sample/request。这样，不但减少了单个 sample/request 的延时，避免了资源浪费问题，同时也提升了整个系统的吞吐。这是一种优化的调度技术，可以更有效地处理动态负载。它允许 TensorRT-LLM 在其他请求仍在进行时开始执行新请求，从而提高 GPU 利用率。   
传统的 Batching 技术为 Static Batching 的，需要等 Batching 中所有序列推理完成后才能进行下一次批次。下图为一个输出最大 Token 为 8，Batch size 为 4 的推理过程，使用 Static Batching 技术。S3 序列在 T5 时刻就已经完成推理，但是需要等到 S2 序列在 T8 时刻推理完成后才会处理下一个 sequence，存在明显的资源浪费。    
In-Flight Batching 又名 Continuous Batching 或 iteration-level batching，该技术可以提升推理吞吐率，降低推理时延。Continuous Batching 处理过程如下，当 S3 序列处理完成后插入一个新序列 S5 进行处理，提升资源利用率。详情可参考论文 Orca: A Distributed Serving System for Transformer-Based Generative Models。      
![in-flight-batching](https://github.com/lix19937/llm-deploy/assets/38753233/f8acd396-0828-460d-9339-a5210023617a)



--------------     
## TensorRT-LLM 的使用流程    
TensorRT-LLM 与 TensorRT的 使用方法类似，首先需要获得一个预训练好的模型，然后利用 TensorRT-LLM 提供的 API 对模型计算图进行改写和重建，接着用 TensorRT 进行编译优化，然后保存为序列化的 engine 进行推理部署。   
![how-to-use](https://github.com/lix19937/llm-deploy/assets/38753233/370ee07d-c7e6-4b8f-aae2-0456fc36b553)


---------------    
以 Llama 为例，首先安装 TensorRT-LLM，然后下载预训练模型，接着利用 TensorRT-LLM 对模型进行编译，最后进行推理。     
![how-to-use--llama](https://github.com/lix19937/llm-deploy/assets/38753233/958d56f5-a331-4357-b676-7e4490d35780)


---------------     
对于模型推理的调试，TensorRT-LLM 的调试方式与 TensorRT 一致。由于深度学习编译器，即 TensorRT，提供的优化之一是 layer 融合。因此，如果要输出某层的结果，就需要将对应层标记为输出层，以防止被编译器优化掉，然后与 baseline 进行对比分析。同时，每标记一个新的输出层，都要重新编译 TensorRT 的 engine。   
![how-to-debug](https://github.com/lix19937/llm-deploy/assets/38753233/3f7b6cc2-c0cb-41fa-99fc-66bb889a0417)


------------    
对于自定义的层，TensorRT-LLM 提供了许多 Pytorch-like 算子帮助用户实现功能而不必自己编写 kernel。如样例所示，利用 TensorRT-LLM 提供的 API 实现了 rms norm 的逻辑，TensorRT 会自动生成 GPU 上对应的执行代码。    
![how-to-add-custom-op](https://github.com/lix19937/llm-deploy/assets/38753233/64d0e615-d6e7-4a5b-9921-1af500d24059)


-----------------   
如果用户有更高的性能需求或者 TensorRT-LLM 并未提供实现相应功能的 building blocks，此时需要用户自定义 kernel，并封装为 plugin 供 TensorRT-LLM 使用。示例代码是将 SmoothQuant 定制 GEMM 实现并封装成 plugin 后，供 TensorRT-LLM 调用的示例代码。  
![how-to-add-custom-op-2](https://github.com/lix19937/llm-deploy/assets/38753233/c928c44a-f903-4228-8834-af3352c1f834)




LLM 是一个推理成本很高、成本敏感的场景。我们认为，为了实现下一个百倍的加速效果，需要算法和硬件的共同迭代，通过软硬件之间 co-design 来达到这个目标。硬件提供更低精度的量化，而软件角度则利用优化量化、网络剪枝等算法，来进一步提升性能。   


## 问答环节     
Q1：是否每一次计算输出都要反量化？做量化出现精度溢出怎么办？   
A1：目前 TensorRT-LLM 提供了两类方法,即 FP8 和刚才提到的 INT4/INT8 量化方法。低精度如果 INT8 做 GEMM 时，累加器会采用高精度数据类型，如 fp16,甚至 fp32 以防止 overflow。关于反量化，以 fp8 量化为例，TensorRT-LLM 优化计算图时，可能动自动移动反量化结点，合并到其它的操作中达到优化目的。但对于前面介绍的 GPTQ 和 QAT，目前是通过硬编码写在 kernel 中，没有统一量化或反量化节点的处理。

Q2：目前是针对具体模型专门做反量化吗？    
A2：目前的量化的确是这样，针对不同的模型做支持。我们有计划做一个更干净的api或者通过配置项的方式来统一支持模型的量化。     

Q3：针对最佳实践，是直接使用 TensorRT-LLM 还是与 Triton Inference Server 结合在一起使用？如果结合使用是否会有特性上的缺失？   
A3：因为一些功能未开源，如果是自己的 serving 需要做适配工作，如果是 triton 则是一套完整的方案。      

Q4：对于量化校准有几种量化方法，加速比如何？这几种量化方案效果损失有几个点？In-flight branching 中每个 example 的输出长度是不知道的,如何做动态的 batching？     
A4：关于量化性能可以私下聊，关于效果，我们只做了基本的验证，确保实现的 kernel 没问题,并不能保证所有量化算法在实际业务中的结果，因为还有些无法控制的因素，比如量化用到的数据集及影响。关于 in-flight batching，是指在 runtime 的时候去检测、判断某个 sample/request 的输出是否结束。如果是，再将其它到达的 requests 插进来，TensorRT-LLM 不会也不能预告预测输出的长度。

Q5：In-flight branching 的 C++ 接口和 python 接口是否会保持一致？TensorRT-LLM 安装成本高，今后是否有改进计划？TensorRT-LLM 会和 VLLM 发展角度有不同吗？      
A5：我们会尽量提供 c++ runtime 和 python runtime 一致的接口，已经在规划当中。之前团队的重点在提升性能、完善功能上，以后在易用性方面也会不断改善。这里不好直接跟 vllm 的比较，但是 NVIDIA 会持续加大在 TensorRT-LLM 开发、社区和客户支持的投入，为业界提供最好的 LLM 推理方案。   

