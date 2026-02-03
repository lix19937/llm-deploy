
TensorRT Edge LLM是NVIDIA的高性能C++推理运行库，用于嵌入式平台上的大型语言模型（LLM）和视觉语言模型（VLM）。
它可以在资源受限的设备（如NVIDIA Jetson和NVIDIA-DRIVE平台）上高效部署最先进的语言模型。

+ 特色   
Advanced KV cache management and quantization support (FP8, INT4)   
Support for LoRA adapters, speculative decoding, and multimodal models     
Python export pipeline, engine builder(c++ ), and runtime(c++ ) in one package      
Python-based toolchain that converts HuggingFace models into ONNX format with quantization (FP8, INT4, NVFP4). 

+ 应用   
车内人工智能助手     
语音控制接口   
场景理解和描述   
驾驶员辅助系统     
自然语言互动    
任务规划和推理   
可视化问答   
人机协作   

+ Large Language Models    
Llama 3.x (1B - 8B)
Qwen 2/2.5/3 (0.5B - 7B)
DeepSeek-R1 Distilled (1.5B, 7B)


+ Vision-Language Models    
Qwen2/2.5/3-VL (2B - 8B)
InternVL3 (1B, 2B)
Phi-4-Multimodal (Phi-4-multimodal-instruct, 5.6B)


+ export onnx 
将HuggingFace模型转换为适用于TensorRT引擎编译的优化ONNX表示。该管道处理模型量化、ONNX导出和LoRA自适应和多模态处理等专门功能。           
模型加载:Load HuggingFace模型和标记器     
量化（可选）  
ONNX导出:将PyTorch模型转换为ONNX格式     
图检查:为TensorRT优化ONNX   
配置生成：创建生成配置文件     
