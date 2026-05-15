
v0.7.0    qwen3 

```
Usage:
    # Export without quantization
    python export_visual.py --model_dir /path/to/model --output_dir /path/to/output
    
    # Export with FP8 quantization
    python export_visual.py --model_dir /path/to/model --output_dir /path/to/output --quantization fp8
    
    # Export with specific device
    python export_visual.py --model_dir /path/to/model --output_dir /path/to/output --device cuda:1
    


tensorrt_edgellm/scripts/export_visual.py  --->

tensorrt_edgellm/onnx_export/visual_export.py --->

tensorrt_edgellm/quantization/visual_quantization.py
```

       
```
# Export with FP8 quantization  fp8(W8A8)  
python export_visual.py --model_dir qwen3.5/ --output_dir vit/output --quantization fp8 --dataset_dir lmms-lab/MMMU  


--> tensorrt_edgellm/scripts/export_visual.py
   |  visual_export()  @  tensorrt_edgellm/onnx_export/visual_export.py  |
       |  _export_qwen_visual()
                 -->   quantize_visual()      @ --> tensorrt_edgellm/quantization/visual_quantization.py
        
                       export_qwen3_vl_visual @ --> tensorrt_edgellm/visual_models/qwen3_vl_model.py

          Export model & processor configurations to JSON
```

### def quantize_visual(model, precision, processor, dataset_dir="lmms-lab/MMMU"):     
<img width="1767" height="931" alt="image" src="https://github.com/user-attachments/assets/b450e93d-1543-4be0-a293-49b6efcb83e0" />

### def export_qwen3_vl_visual    
https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/release/0.7.0/tensorrt_edgellm/visual_models/qwen3_vl_model.py#L283-L371  

### Export model & processor configurations to JSON     
<img width="1429" height="686" alt="image" src="https://github.com/user-attachments/assets/dd961584-95aa-4509-be1a-ce8f7ca83ef9" />
