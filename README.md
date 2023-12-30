## LLM 推理加速   


* cnn-based  [tensorrt](https://github.com/lix19937/trt-samples-for-hackathon-cn/blob/master/cookbook/readme_cn.md)
基于TRT，自定义插件完成低效模块的计算，最终仍back到 TRT中   

* transformer-based  [tensorrt-llm](https://github.com/NVIDIA/TensorRT-LLM)   
对于无cnn的，不借助onnx解析器，自己搭建网络推理 
如 llama.cpp   llama2.c    

* cnn + transformer both-based  tensorrt[plugin] + tensorrt-llm

  
