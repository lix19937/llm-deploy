## LLM 推理加速   

![2016-openvx-api-slide6](https://github.com/lix19937/llm-deploy/assets/38753233/e9dd22fe-6206-485d-af68-83ac53fd1ed0)
  
* cnn-based(backbone) + cnn(head) [tensorrt](https://github.com/lix19937/trt-samples-for-hackathon-cn/blob/master/cookbook/readme_cn.md)     
基于TRT，自定义插件完成低效模块的计算，最终仍back到TRT中   

* transformer-based(decoder)  [tensorrt-llm](tensorrt-llm/readme.md)  https://github.com/NVIDIA/TensorRT-LLM           
对于无cnn的，如GPT2，不借助onnx解析器，自己搭建网络推理，典型的llama.cpp   llama2.c 这类针对特定结构开发的推理项目，以及 [vllm](./vllm/readme.md)          
https://lilianweng.github.io/posts/2023-01-10-inference-optimization/#distillation     

* cnn + transformer  tensorrt[plugin] + tensorrt-llm         
例如detr3d, vit 需要tensorrt[plugin] 与 tensorrt-llm 一起使用，实现最优效果

-------------------------------   
https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/

## LLM Inference --算法与工程的协作       

|方向/技术点|说明 |   
|----------|----|   
|Inference算法部分| Transformer inference过程及加速原理|  
|解码策略及调参| GreedySearch、BeamSearch、Sampling、top_k、top_p、temperature、no_repeated_ngram_size等优化|   
|多机多卡的GPU集群分布式解码，并行（Tensor/Pipeline/MoE Expert parallelism）| 集群的搭建、不同机器以及卡的高效通信等|   
|高并发处理和优化|负载均衡，batch_size调优等|   
|系统底层相关|不同显卡型号、底层GPU驱动、内存管理、算子等|   
|其他工程相关|GPU集群管理，稳定性，日常维护等|   
         
### Inference优化目标    
关注2个指标：**Latency** 和 **Throughput**，这两个指标一般情况下需要**trade-off**。    
  
|指标 | 说明 |   
|---- | ----|   
| Latency | 关注服务体验，也就是返回结果要快，用户体验好。 <br><br> 延时，主要从用户的视角来看，也就是用户提交一个prompt，然后得到response的时间。特殊情况batch_size=1只给一个用户进行服务，Latency是最低的。计算方法为生成一个token所需要的单位时间数，如16 ms/token。 |   
| Throughput |关注系统成本，高Throughput则系统单位时间处理的量就大，系统利用率高，但是会影响latency。<br><br> 吞吐率，主要是从系统的角度来看，单位时间内能处理的tokens数，如16 tokens/sec。扩大Throughput的方法一般就是提升Batch_size，也就是将一个一个用户的请求由之前的串行改为并行。    |   

高并发时，把用户的prompt合并扩大batch_size能提升Throughput，但会一定程度上损害每个用户的 Latency，因为以前只计算一个请求，现在合并计算多个请求，每个用户等待的时间就长了。从实际的测试结果可以看到，Throuput随着batch_size的增大而增大，但是latency是随着增大的，当然 Latency 在可接受范围内就是ok的。因此指标需要trade-off。     
简单计算，对于一次请求来说：   
```py   
latency=batch_size * output_sequence_length / Throughput
```
提升batch_size会提升Throughput，但Throughput与batch_size并不是同比例增大的，因此导致Latency随着batch_size增大而增大。

### Inference过程   
2个阶段 Prefill Phase和 Decoding Phase（见FlexGen）       

|阶段 | 说明|  
|----| ----|   
|Prefill Phase|预处理/Encoding。计算并缓存每一层的key和value，其他的不需要缓存。每一个请求的prompt需要经过这个阶段，它只计算一次，是并行计算的。这个缓存称为KV Cache，KV Cache是整个解码过程中最为核心关键的一块。|     
|Decoding Phase|生成新token阶段，它是串行的，也就是decode one by one。它用上一步生成的token，称为当前token放到input中，然后生成下一个token。具体包括两步，一是Lookup KV Cache计算并输出当前token最终embedding用来预测下一个token，二是缓存计算过程中得到的当前token在每一层的key和value，update到第一阶段Prefill Phase中的KV Cache中。|    

这样划分的好处是，每次解码，不需要对整个网络全部进行计算，`相当于计算复杂度从O(n^3)变为O(n)了`。Prefill Phase只需要计算一次，核心就是GPT-style的transformer是单向的，而不是双向的。每一个token只与它前面的tokens有关系，其key，value也只与它前面的tokens有关系，因此可以只计算一次且可以并行。后面decoding phase就很快了，避免了重复计算。整个的开销就只有key-value cache的update以及look-up的时间。   
有文章里面提出LLM inference is memory-IO bound, not compute bound。从下面的量化分析来看确实如此。

### Inference量化分析      
Inference的核心是KV Cache，以FP16为例，其对显存的占用量化分析如下。

其对显存占用分为两块，一是Weights、二是KV Cache。   
```
Weights占用大约为layer_num * ( 8 * hidden_size * hidden_size + 4 * hidden_size * MLP_hidden_size )。  head_num=8   
KV Cache的占用为4 * batch_size * layer_num * hidden_size * ( input_length + output_length )。  
```
以OPT-175B 为例（layer_num = 96, hidden_size = 12288, MLP_hidden_size = 49152，batch_size=512，input_length=512, output length=32)。   

Weights差不多占用 325G, KV cache 差不多占用 1.2T。对内存消耗是非常惊人。里面唯一可以调节的参数是batch_size，显存占用基本与batch_size呈正比关系。显存的大小限制了batch_size，而batch_size的大小就限制了Throughput。因此就有很多加速工作就是想办法节省显存，进而扩大batch_size。

### 优化方法   

|优化方向| 方法|     
|---    |---  |     
|Latency|优化底层的OP算子（高效kernel，如FMHA）、矩阵优化、并行、更高效的C++解码等，如FasterTransformer以及DeepSpeed。针对Latency的优化可以提升Throughput，但没有直接用batch_size提升的更显著。<br><br> **量化技术**，如gptq |     
|Throughput|主要是**KV Cache存取优化**，将transformer attention计算中的Key和Value张量集合缓存下来，避免每输出一个token都重复计算。本质是降低显存开销，从而可以提升batch size。这方面工作相对多一些，如**offloading技术**，就是如何高效利用第三方存储CPU/DRAM/Disk，使得GPU显存能空出来进而增大batch_size。<br><br> 如vLLM中的 **PagedAttention** 技术就是借鉴OS中的分页以及虚拟存储思想实现显存动态分配，也能节省很多显存空间。<br>如**continuous batching**，变传统的static batch为动态可复用的batch分配，同样也能尽可能扩大batch_size，进而提升Throughput。|    

### 一些主流加速框架   

| 名称| 出品方| 主打| 方法  |  备注  |     
| ----|------| ----| ---- | -------|    
| FasterTransformer| Nvidia | Latency| 90%的时间消耗在12层Transformer的前向计算上，总结优化点如下：https://zhuanlan.zhihu.com/p/79528308<br>为了减少kernel调用次数，将除了矩阵乘法的kernel都尽可能合并（这个可能是主要的）<br>针对大batch单独进行了kernel优化<br>支持选择最优的矩阵乘法（启发式搜索与cutlass）<br>在使用FP16时使用half2类型，达到half两倍的访存带宽和计算吞吐<br>优化gelu、softmax、layernorm的实现以及选用rsqrt等 <br>使用硬件加速的底层函数，如__expf、__shfl_xor_sync   | - |       
|DeepSpeed|微软|Latency和 Throughput| 优化Latency：a multi-GPU inference solution.<br>parallelism：Tensor parallelism、Pipeline parallelism、Expert Parallelism（MoE）。对多机多卡之间的通信带宽要求较高 <br>communication optimization<br>optimized sparse kernels<br><br>优化Throughput：Zero-Inference也用到了offloading技术<br> 如何结合GPU显存以及其他外部存储设备如DRAM、NVMe等加载大模型，问题变为How to apportion GPU memory among model weights, inference inputs and intermediate results <br> 然后可以接受大的batch size，进而提升Throughput。| - |     
|llama.cpp|ggerganov| Latency| offloading、高效C++解码 <br><br>面向消费级CPU/GPU的Inference框架，主打易用性，CPU支持 <br><br>GPU多核计算能力：通过调用CUDA、OpenCL等API，来利用GPU的并行能力。<br>CPU SIMD和多核：单指令多数据SIMD在x86上有SSEx和AVX等指令，在ARM上有NEON和SVE，都广泛被使用，也有库通过OpenMP再叠加多核能力。| -|      
|vLLM     |UC Berkeley| Throughput| paged attention，动态分配K-V Cache，提升Batch_size <br><br>KV cache占用大量GPU内存，一个13B模型每个输出token对应的KV张量，需要800KB，而最长输出长度2048个token的话，一个请求就需要1.6GB显存。因此vLLM引入类似操作系统中的分页机制，大幅减少了KV cache的碎片化，提高性能。 | -  |  
|FlexGen  |Stanford/UC Berkeley/CMU/META  | Throughput| 在有限资源情况下如何高效利用CPU/Disk以提升Throughput  | -  |  
|Hugging Face pipeline Accelerate  |HuggingFace | Latency| distributed Inference （https://huggingface.co/docs/accelerate/usage_guides/distributed_inference）| -  |  

-------------------------------------------------------

要想最大化提升推理的性能，必须得先了解机器的算力资源以及其峰值算力，优化的过程其实就是不断逼近峰值算力的过程。本文我们仅讨论使用GPU进行推理的场景。图1 中是A10的核心参数，从右侧SPECIFICATIONS中可以看到其FP32最大算力是31.2TFLOPS, Memory BandWidth为600GB/s, 但是这个数值是如何得来的呢？
```
峰值算力公式：单核单周期计算次数 × 处理核（cuda core）个数 × 主频
```
比如A10: 2*1.7x10^9*9216/10^12=31.2TFLOPS
```
峰值带宽公式：内存主频 x 位宽x 2/ 8
```
比如A10:6251x10^6X384x2(DDR)/8=600GB/s

从以上公式可以看出，必须所有的cuda core同时参与计算才可以达到峰值算力，这也给我们优化程序提供了思路，只有尽可能的提升cuda core的使用率，才可以更加的逼近峰值算力。不过影响程序的可能不仅是算力，还有可能是IO，很多时候会因为IO提前到达峰值而导致算力资源没有数据可算，这时就需要分析我们的程序是计算约束还是访存约束了。

## 计算约束or 内存约束

如何判断程序是compute-bound还是memory-bound。假设一个函数的执行通常经过以下流程：   
1）从memory中读取input。2）执行算术运算。3）将output写回memory       
![compute-io](https://github.com/lix19937/llm-deploy/assets/38753233/c5997517-6a82-4f33-8fdc-c654452d97a0)

让我们来看看深度神经网络的一些具体例子，如下表1所示。对于这些例子，我们将比较在V100上算法的算术强度与操作数与字节比. V100的峰值数学运算速率为125 FP16 Tensor TFLOPS，片外存储器带宽约为900 GB / s，片上L2带宽为3. 1 TB / s，使其操作数与字节比率在40和139之间，取决于操作数据的来源（片上或片外存储器)。
![op-bytes](https://github.com/lix19937/llm-deploy/assets/38753233/a2a38029-1e5c-4eb9-8b26-ddc807b244c9)

如表所示，许多常见操作的**算术强度**都很低，有时仅对从内存读取并写入内存的每个2字节元素执行一个操作。请注意，这种分析方法是一种简化，因为我们只计算所使用的算法运算。实际上，函数还包含算法中没有明确表示的操作指令，如内存访问指令、地址计算指令、控制流指令等。

算术强度和操作：字节比分析假设工作负载足够大，足以使给定处理器的数学和内存管道饱和。但是，如果工作负载不够大，或者没有足够的并行性，则处理器的利用率将不足，性能将受到延迟的限制。例如，考虑启动一个线程，该线程将访问16个字节并执行16000个数学运算。虽然算术强度为1000 FLOPS/B，并且在V100 GPU上的执行应该受到数学限制，但仅创建一个线程严重利用GPU不足，几乎所有的数学管道和执行资源都处于空闲状态。此外，算术强度计算假设从存储器访问输入和输出恰好一次。算法实现多次读取输入元素并不罕见，这将显著地降低运算强度。

DNN中到底哪些算子是计算约束的，哪些又是内存约束的呢？   
Elementwise 类型的算子，比如Relu、sigmoid, tanh, Reduction类型的算子比如 pooling、softmax、batchnorm等都属于memory-bound型  
对于卷积、全连接这些，当batch size比较小时也都是memory-bound的。也就是说神经网络的大部分层都是memory-bound，而batch size可以有效减少memory io次数，所以推理时增加batch size可以显著提升吞吐量。

## 主流优化方案的底层原理

+ 算子融合。不管是tensorrt、fastertransformer、还是tensorflow、onnx，在图优化阶段都会有算子融合的优化手段，融合后的算子，其计算量并没有减少，但是其的确可以提升模型的性能，为什么呢，这要从cuda程序的执行来说了，假如有A、B、C三个kernel代表三个算子，他们的执行顺序为A->B->C, kernel函数的执行是在gpu上计算的，但是kernel函数的启动是由cpu控制。每执行一个kernel，cpu需要调用cuda driver来执行LaunchKernel的操作，今年GTC2023，英伟达介绍SwinTransformer里有提到，Launch Kernel之间会存在1us+的开销，如果将A\B\C合成一个算子，那么可以减少两次kernel launch的时间。另外kernel函数的输入和输出都是暂存于DRAM（也就是全局内存中），拿最常见的conv+bn+relu来说，conv算子执行完之后需要把tensor写回DRAM,bn算子再从DRAM中读取tensor进行操作，之后再写回DRAM，接着relu再次从DRAM中读取bn算子写回的tensor，将A、B、C三个算子融合后，conv的执行结果无需写回DRAM中，将在寄存器或者L1 Cache/shared memory中直接参与bn的计算，L1 Cache的带宽是DRAM的10倍以上，所以算子融合可以大幅减少DRAM的访存次数，进而提升模型性能。      
+ 模型量化。模型量化可以降低模型参数占用显存的大小，但模型量化为什么可以多于两倍的性能提速呢。这需要从两个方面来解释，第一，比如由FP32精度量化到FP16精度，相同的访存带宽下可以读写两倍的操作数，同时一个FP32 cuda core也可以一次操作两个FP16的计算，第二，从volta架构以后，nvidia gpu引入了tensor core，这是转为矩阵计算提供的专门硬件，其性能是cuda core的数倍（可以参考：模型推理场景该如何选型GPU - 知乎 (zhihu.com)）。然而大部分神经网络的算子其内部多为一些GEMM操作，在使用低精度推理时都可以用上tensor core，所以模型量化效果会非常显著。
+ batch推理、多cuda流并行、并实例并行。这三个放在一块说主要是因为他们都可以同时处理多条请求，但是他们的底层原理并不一致。        
![3way](https://github.com/lix19937/llm-deploy/assets/38753233/b2e2af5a-28af-4ec8-8629-cc34e40c2613)    
+ FlashAttention是一种创新的注意力计算方法，旨在提高计算效率、节省显存，并降低IO感知。这种方法有效地缓解了传统注意力机制在计算和内存管理方面的问题。FlashAttention并没有减少计算量FLOPs，但其创新之处在于，从IO感知的角度出发，减少了HBM（高带宽存储器）的访问次数。这种优化策略不仅提高了计算效率，还显著减少了计算时间的总体耗费。在论文中，作者使用了"wall-clock time"这个词，该词综合考虑了GPU运行耗时和IO读写阻塞时间。而FlashAttention通过运用tiling技术和算子融合，有效地降低了HBM的访问次数，从而显著减少了时钟时间。FlashAttention之所以能够实现高效的优化，是因为注意力操作是一种memory-bound操作。对于这类操作，减少HBM（DRAM）的访问次数是最有效的优化手段。因此，FlashAttention为解决注意力机制的计算效率和内存管理问题提供了一种新颖且实用的解决方案。

## gpu角度下dnn性能     
[understand-perf ](https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/index.html#understand-perf)   
解读     


https://zhuanlan.zhihu.com/p/649640010




