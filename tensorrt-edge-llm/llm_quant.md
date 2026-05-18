主要针对qwen3   

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

## 1 load_hf_model   (Load model and tokenizer)  @ tensorrt_edgellm/llm_models/model_utils.py    
```py  
# only support fp16 
torch_dtype = torch.float16

# AutoModelForCausalLM        来自 transformers 包
# AutoModelForImageTextToText 来自 transformers 包
# AutoProcessor               来自 transformers 包
# AutoTokenizer               来自 transformers 包

tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True) 

# vlm 
if is_vlm(model_dir):
    # Try multimodal loader first; AutoModelForCausalLM would silently drop the visual tower for models that register both classes.
    try:
        model = AutoModelForImageTextToText.from_pretrained(model_dir, torch_dtype=torch_dtype, trust_remote_code=True).to(device)
    except Exception as e_vlm:
        try:
            model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch_dtype, trust_remote_code=True).to(device) 
        except Exception as e:
            raise ValueError(f"Could not load model from {model_dir}. Error: {e}")
else:
    try:
        model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch_dtype, trust_remote_code=True).to(device) 
    except Exception as e_causal:
        try:
            model = AutoModelForImageTextToText.from_pretrained(model_dir, torch_dtype=torch_dtype, trust_remote_code=True).to(device) 
        except Exception as e:
            raise ValueError(f"Could not load model from {model_dir}. Error: {e}")

# un-gptq 
if not is_gptq_model(model):
    model.to(torch_dtype) # 强制转换数据精度格式
else:
    casted_params, casted_buffers, skipped_quantized_modules = _cast_non_gptq_float_tensors_to_dtype(model, torch_dtype) # 转换un-gptq dtype to fp16
    print(f"GPTQ load dtype normalization: cast {casted_params} params and {casted_buffers} buffers to {torch_dtype}; skipped {skipped_quantized_modules} GPTQ quantized modules.")

    # Set tokenizer padding token if needed
    if tokenizer.pad_token != "<unk>":
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

# Try to load processor if available
try:
    processor = AutoProcessor.from_pretrained( model_dir, trust_remote_code=True, min_pixels=128 * 28 * 28, max_pixels=2048 * 32 * 32) 
    print(f"Warning: Loaded processor from {model_dir}. The processor will skip image processing for images smaller than 128x28x28 or bigger than 2048x32x32 due to excessive memory usage during image quantization.")
except Exception:
    pass

# return model, tokenizer, processor
```

### 1.2 _cast_non_gptq_float_tensors_to_dtype  @ tensorrt_edgellm/llm_models/model_utils.py    
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

## 2 quantize_llm
```py
def quantize_llm(
    model: Union[AutoModelForCausalLM, AutoModelForImageTextToText],
    tokenizer: AutoTokenizer,
    dataset_dir: str, #  用于PTQ 
    quantization:          Optional[str],
    lm_head_quantization:  Optional[str],
    kv_cache_quantization: Optional[str],
    is_omni: bool = False,
    processor=None,
    model_dir: Optional[str] = None,
    audio_dataset_dir:  str = "openslr/librispeech_asr",
    visual_dataset_dir: str = "lmms-lab/MMMU",
) -> Union[AutoModelForCausalLM, AutoModelForImageTextToText]:
    """Quantize a language model using the specified quantization method.

    Qwen3-ASR uses audio-backed calibration for the LLM backbone.
    Qwen3-Omni uses multimodal calibration when a processor is available, and falls back to text-only calibration otherwise.

    Args:
        model: The model to quantize.
        tokenizer: Tokenizer for text processing.
        dataset_dir: Calibration dataset. Text by default; ASR may switch to audio-backed calibration automatically.
        quantization: Quantization method.
        lm_head_quantization:  Optional LM head quantization method.
        kv_cache_quantization: Optional KV cache quantization method.
        is_omni: Use multimodal Omni calibration pipeline.
        processor: HuggingFace processor (Omni multimodal calib).
        model_dir: Original model directory.
        audio_dataset_dir: Audio calibration dataset (Omni).
        visual_dataset_dir: Image calibration dataset (Omni).
    """
    assert (quantization is not None) or (lm_head_quantization is not None) or (kv_cache_quantization is not None), \
        "At least one of 'quantization', 'lm_head_quantization', or 'kv_cache_quantization' must be set (not all None)."
    assert quantization          in [None, "fp8", "int4_awq", "nvfp4", "mxfp8", "int8_sq"]
    assert lm_head_quantization  in [None, "fp8", "nvfp4", "mxfp8"]
    assert kv_cache_quantization in [None, "fp8"]

    # q config 
    quant_config = get_llm_quant_config(quantization, lm_head_quantization, kv_cache_quantization)

    batch_size = 16 if quantization is None or "int4" in quantization else 1

    # calib_dataloader
    data_loader = get_text_calib_dataloader(tokenizer=tokenizer, dataset_dir=dataset_dir, batch_size=batch_size, num_samples=512, max_length=512)
    return quantize_model(model, quant_config, data_loader)
```

### 2.1 get_llm_quant_config  @ tensorrt_edgellm/quantization/llm_quantization.py    
```py
def get_llm_quant_config(
        quantization         : Optional[str],
        lm_head_quantization : Optional[str],
        kv_cache_quantization: Optional[str]) -> Dict[str, Any]:
    """
    Get quantization configuration for LLM models.
    
    Args:
        quantization         : Optional quantization method
        lm_head_quantization : Optional LM head quantization method
        kv_cache_quantization: Optional attention quantization method   (enables FP8 KV cache + FP8 FMHA compute)
        
    Returns: Dict containing quantization configuration
    """
    # Get base config
    if quantization is None:
        quant_cfg = {"quant_cfg": {}, "algorithm": "max"}
    elif quantization == "fp8":
        quant_cfg = mtq.FP8_DEFAULT_CFG.copy()
    elif quantization == "int4_awq":
        quant_cfg = mtq.INT4_AWQ_CFG.copy()
    elif quantization == "nvfp4":
        quant_cfg = mtq.NVFP4_DEFAULT_CFG.copy()
    elif quantization == "mxfp8":
        quant_cfg = mtq.MXFP8_DEFAULT_CFG.copy()
    elif quantization == "int8_sq":
        quant_cfg = mtq.INT8_SMOOTHQUANT_CFG.copy()
    else:
        raise ValueError(f"Unsupported quantization: {quantization}")

    # Add LM head quantization if specified
    if lm_head_quantization is not None:
        # Remove any existing lm_head configuration
        quant_cfg["quant_cfg"] = { k: v  for k, v in quant_cfg["quant_cfg"].items() if "*lm_head" not in k }

        if lm_head_quantization == "fp8":
            quant_cfg["quant_cfg"].update(FP8_LM_HEAD_CONFIG["quant_cfg"])
        elif lm_head_quantization == "nvfp4":
            quant_cfg["quant_cfg"].update(NVFP4_LM_HEAD_CONFIG["quant_cfg"])
        elif lm_head_quantization == "mxfp8":
            quant_cfg["quant_cfg"].update(MXFP8_LM_HEAD_CONFIG["quant_cfg"])

    # Add attention/KV-cache quantization if specified (FP8 KV cache + FP8 FMHA compute)
    if kv_cache_quantization is not None:
        if kv_cache_quantization == "fp8":
            quant_cfg["quant_cfg"].update(mtq.FP8_KV_CFG["quant_cfg"])
            quant_cfg["quant_cfg"].update(FP8_ATTN_CONFIG["quant_cfg"])

    # Disable non-LLM submodules (visual/audio encoders, Phi-4MM embeds, etc.)
    quant_cfg["quant_cfg"].update(DISABLE_NON_LLM_CONFIG["quant_cfg"])
    return quant_cfg
```

### 2.2 get_text_calib_dataloader @ tensorrt_edgellm/quantization/calib_dataloaders.py    
```py
def get_text_calib_dataloader(
    tokenizer: AutoTokenizer,
    dataset_dir: str,
    batch_size: int,
    num_samples: int,
    max_length: int,
) -> DataLoader:
    """
    Create a text calibration dataloader for LLM quantization.

    Args:
        tokenizer  : HuggingFace tokenizer for text processing.
        dataset_dir: Dataset name or local directory path.
        batch_size : Batch size for the dataloader.
        num_samples: Number of samples to use for calibration.
        max_length : Maximum sequence length for tokenization.

    Returns: DataLoader yielding batches of ``input_ids`` tensors.
    """
    if "cnn_dailymail" in dataset_dir:
        dataset = load_dataset(dataset_dir, name="3.0.0", split="train")
        dataset = dataset["article"][:num_samples]
    elif os.path.isdir(dataset_dir):
        print(f"Recognized local dataset repo {dataset_dir} for calibration; assuming the calibration data are in the train split and text column.")
        dataset = load_dataset(dataset_dir, split="train")
        dataset = dataset["text"][:num_samples]
    else:
        raise NotImplementedError(f"Unsupported dataset name or local repo directory: {dataset_dir}.")

    # Use tokenizer __call__ for transformers v5-compatible batch tokenization.
    batch_encoded = tokenizer(dataset, # 这里 tokenizer type is AutoTokenizer 
                              return_tensors="pt",
                              padding=True,
                              truncation=True,
                              max_length=max_length)

    return DataLoader(batch_encoded["input_ids"], # 这里DataLoader from torch.utils.data import DataLoader
                      batch_size=batch_size,
                      shuffle=False)

```

### 2.3 quantize_model  @ tensorrt_edgellm/quantization/quantization_utils.py   
```py
def quantize_model(
    model: torch.nn.Module,
    quant_config: Dict[str, Any],
    calib_dataloader: DataLoader,
) -> torch.nn.Module:
    """
    Quantize a PyTorch model using the specified configuration and calibration data.
    
    Args:
        model: PyTorch model to quantize
        quant_config: Quantization configuration dictionary
        calib_dataloader: DataLoader for calibration data
        
    Returns: Quantized PyTorch model
    """
    # Define calibration loop
    def calibrate_loop(model: torch.nn.Module) -> None:
        """
        Calibration loop that adjusts weights and scaling factors.
        
        Args:  model: Model to calibrate
        """
        # Create progress bar for calibration
        print(f"Calibrating model on {len(calib_dataloader)} samples...")
        pbar = tqdm(calib_dataloader, desc="Calibrating", unit="num_samples")

        # Add extra necessary kwargs for Phi-4-Multimodal
        kwargs = {}
        if hasattr(model, "config") and "phi4mm" in getattr( model.config, "model_type", "").lower():
            # Have already merged the vision LoRA, so set input_mode=0 (LANGUAGE) during quantization
            kwargs["input_mode"] = 0
            # Work around a transformers version mismatch between Phi-4MM and Edge-LLM
            kwargs["use_cache"] = False

        for data in pbar:
            if isinstance(data, dict):
                data = {
                    k:v.to(model.device, dtype=model.dtype if v.is_floating_point() else None) #  v.is_floating_point
                    for k, v in data.items()
                }
                model(**data, **kwargs)
            else:
                data = data.to(model.device)
                model(data, **kwargs)

    # Get quantization config and perform quantization
    mtq.quantize(model, quant_config, forward_loop=calibrate_loop)
    # 检查量化情况  
    mtq.print_quant_summary(model)
    return model

```

## 3 _sanitize_generation_config    


## 4 export_hf_checkpoint    
```
from modelopt.torch.export import export_hf_checkpoint
```

## 5 tokenizer.save_pretrained  
``` 
tokenizer is AutoTokenizer.from_pretrained
```

## 6 processor.save_pretrained   
```
processor is AutoProcessor.from_pretrained 
```
