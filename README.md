## LLM 推理加速   

![acc](./2016-openvx-api-slide6.png)   

* cnn-based  [tensorrt](https://github.com/lix19937/trt-samples-for-hackathon-cn/blob/master/cookbook/readme_cn.md)     
基于TRT，自定义插件完成低效模块的计算，最终仍back到 TRT中   

* transformer-based  [tensorrt-llm](https://github.com/NVIDIA/TensorRT-LLM)      
对于无cnn的，如GPT2，不借助onnx解析器，自己搭建网络推理    
如 llama.cpp   llama2.c  这类针对特定结构开发的推理项目           
https://lilianweng.github.io/posts/2023-01-10-inference-optimization/#distillation

* cnn + transformer both-based  tensorrt[plugin] + tensorrt-llm
例如detr3d, vit 需要tensorrt[plugin] 与 tensorrt-llm 一起使用，实现最优效果      
  
