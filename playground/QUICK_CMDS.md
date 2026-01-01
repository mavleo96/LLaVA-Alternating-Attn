### BLINK Evaluation Commands
#### Usage:
```
python scripts/visualcorres_blink_eval.py \
  --model_path "liuhaotian/llava-v1.6-mistral-7b" \
  --model_name "llava_mistral" \
  --output_path "results/llava-v1.6-mistral-7b-visualcorres.json" \
  --subtask "Visual_Correspondence" \
  --device "cuda:1" \
  --conv_template "mistral_instruct"
```

#### Usage with LoRA finetuned model:
```
python scripts/visualcorres_blink_eval.py \
  --model_path "/workspace/checkpoints/llava-onevision-qwen2-0.5b-ov-with_alternating_attn" \
  --model_name "llava_qwen"  \
  --lora_weights_path "/workspace/checkpoints/llava-onevision-qwen2-0.5b-ov-with_alternating_attn-finetune/checkpoint-11000" \
  --subtask "Visual_Correspondence" \
  --device "cuda:1" \
  --conv_template "qwen_2"
```

#### Options:
| model_path | model_name | conv_template |
| ---------- | ---------- | ------------- |
| "liuhaotian/llava-v1.6-mistral-7b" | "llava_mistral" | "mistral_instruct" |
| "lmms-lab/llava-onevision-qwen2-7b-ov" | "llava_qwen" | "qwen_2" |
| "lmms-lab/llava-onevision-qwen2-0.5b-ov" | "llava_qwen" | "qwen_2" |

### Command to save attention matrix for Blink evaluation:
```
python playground/attention_matrix_save_for_visualcorres.py \
  --model_path "/workspace/checkpoints/llava-onevision-qwen2-0.5b-ov-with_alternating_attn" \
  --model_name "llava_qwen_with_alternating_attn" \
  --lora_weights_path "/workspace/checkpoints/llava-onevision-qwen2-0.5b-ov-with_alternating_attn-finetune/checkpoint-11000" \
  --subtask Visual_Correspondence \
  --device "cuda:1" \
  --conv_template "qwen_2" \
  --output_path "results/llava-onevision-qwen2-0.5b-ov-with_alternating_attn-finetune-visualcorres.npz"
```

### Command to create unfinetuned model checkpoint:
```
huggingface-cli download lmms-lab/llava-onevision-qwen2-0.5b-ov \
    --local-dir /workspace/checkpoints/llava-onevision-qwen2-0.5b-ov-with_alternating_attn \
    --local-dir-use-symlinks False
```



### LLava-OneVision-Data Subset List For Finetuning

image_folder = "/data/LLaVA-OneVision-Data"

| Dataset Name | Config File Name |
| ---------- | ---------- |
|"aokvqa(cauldron,llava_format)" | "ov_aokvqa_cauldron_llava_format.json" |
| "chartqa(cauldron,llava_format)" |"ov_chartqa_cauldron_llava_format.json" |
| "clevr(cauldron,llava_format)" | "ov_clevr_cauldron_llava_format.json" |
| "tqa(cauldron,llava_format)" | "ov_tqa_cauldron_llava_format.json" |
| "raven(cauldron)" | "ov_raven_cauldron.json" |
| "visual7w(cauldron,llava_format)" | "ov_visual7w_cauldron_llava_format.json" |
| "vision_flan(filtered)" | "ov_vision_flan_filtered.json" |
| "image_textualization(filtered)" | "ov_image_textualization_filtered.json" |

### Synthetic Visual Correspondence Data
Image folder: "/data/synthetic_visualcorres/images"

```
python scripts/download_synthetic_visualcorres_data.py --image_folder "/data/synthetic_visualcorres/images"
```