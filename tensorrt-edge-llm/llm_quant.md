
```
Usage:
    # Quantize with FP8 quantization
    python quantize_llm.py --model_dir /path/to/model --output_dir /path/to/output --quantization fp8

    # Quantize Qwen3-Omni with multimodal calibration (auto-detected)
    python quantize_llm.py --model_dir /path/to/qwen3-omni --output_dir /path/to/output --quantization nvfp4

    # Qwen3-Omni with custom audio/visual calibration datasets
    python quantize_llm.py --model_dir /path/to/qwen3-omni --output_dir /path/to/output \\
        --quantization nvfp4 --audio_dataset_dir openslr/librispeech_asr --visual_dataset_dir lmms-lab/MMMU
```

```
        "--quantization",           choices=["fp8", "int4_awq", "nvfp4", "mxfp8", "int8_sq"],     

        "--dtype",                  choices=["fp16"],

        "--dataset_dir",            default="cnn_dailymail"

        "--lm_head_quantization",   choices=["fp8", "nvfp4", "mxfp8"],  help="Quantization method for language model head (only fp8, nvfp4, and mxfp8 are currently supported)"

        "--kv_cache_quantization",  choices=["fp8"],  help="Attention quantization: enables FP8 KV cache and FP8 FMHA compute (Q/K/V BMM quantizers + BMM2 output quantizer)")
```

```
tensorrt_edgellm/scripts/quantize_llm.py --->
   quantize_and_save_llm() @  tensorrt_edgellm/quantization/llm_quantization.py
      --> load_hf_model()
      --> quantize_llm()
      --> _sanitize_generation_config
      --> export_hf_checkpoint
      --> tokenizer.save_pretrained
      --> processor.save_pretrained
       
```

### load_hf_model   (Load model and tokenizer)   

```
# only support fp16 
torch_dtype = torch.float16

tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True) # AutoTokenizer 来自 transformers 包

# vlm 
if is_vlm(model_dir):
    # Try multimodal loader first; AutoModelForCausalLM would silently drop the visual tower for models that register both classes.
    try:
        model = AutoModelForImageTextToText.from_pretrained(model_dir, torch_dtype=torch_dtype, trust_remote_code=True).to(device) # AutoModelForImageTextToText 来自 transformers 包
    except Exception as e_vlm:
        try:
            model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch_dtype, trust_remote_code=True).to(device) # AutoModelForCausalLM 来自 transformers 包
        except Exception as e:
            raise ValueError(f"Could not load model from {model_dir}. Error: {e}")
else:
    try:
        model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch_dtype, trust_remote_code=True).to(device) # AutoModelForCausalLM 来自 transformers 包
    except Exception as e_causal:
        try:
            model = AutoModelForImageTextToText.from_pretrained(model_dir, torch_dtype=torch_dtype, trust_remote_code=True).to(device) # AutoModelForImageTextToText 来自 transformers 包
        except Exception as e:
            raise ValueError(f"Could not load model from {model_dir}. Error: {e}")

# un-gptq 
if not is_gptq_model(model):
    model.to(torch_dtype) # 强制转换数据精度格式
else:
    casted_params, casted_buffers, skipped_quantized_modules = _cast_non_gptq_float_tensors_to_dtype(model, torch_dtype)
    print(f"GPTQ load dtype normalization: cast {casted_params} params and {casted_buffers} buffers to {torch_dtype}; skipped {skipped_quantized_modules} GPTQ quantized modules.")

    # Set tokenizer padding token if needed
    if tokenizer.pad_token != "<unk>":
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

# Try to load processor if available
try:
    processor = AutoProcessor.from_pretrained( model_dir, trust_remote_code=True, min_pixels=128 * 28 * 28, max_pixels=2048 * 32 * 32) # AutoProcessor 来自 transformers 包
    print(f"Warning: Loaded processor from {model_dir}. The processor will skip image processing for images smaller than 128x28x28 or bigger than 2048x32x32 due to excessive memory usage during image quantization.")
except Exception:
    pass

# return model, tokenizer, processor

```

### _cast_non_gptq_float_tensors_to_dtype 
```py
def _cast_non_gptq_float_tensors_to_dtype(model: nn.Module, target_dtype: torch.dtype) -> Tuple[int, int, int]:
    """
    Cast floating tensors to target_dtype while preserving GPTQ quantized modules.

    Returns: Tuple of (casted_param_count, casted_buffer_count, skipped_quantized_module_count).
    """
    casted_params = 0;    casted_buffers = 0;    skipped_quantized_modules = 0
    with torch.no_grad():
        for module in model.modules():
            if _is_gptq_quantized_module(module):
                skipped_quantized_modules += 1
                continue
            for _, param in module.named_parameters(recurse=False):
                if param.is_floating_point() and param.dtype != target_dtype:
                    param.data = param.data.to(dtype=target_dtype)
                    casted_params += 1
            for buffer_name, buffer in module.named_buffers(recurse=False):
                if buffer.is_floating_point() and buffer.dtype != target_dtype:
                    setattr(module, buffer_name, buffer.to(dtype=target_dtype))
                    casted_buffers += 1
    return casted_params, casted_buffers, skipped_quantized_modules
```
