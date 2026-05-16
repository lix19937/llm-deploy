
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
