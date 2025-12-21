python scripts/visualcorres_blink_eval.py \
  --model_path "Intel/llava-gemma-2b" \
  --model_name "llava_gemma" \
  --subtask "Visual_Correspondence" \
  --device "cuda:1" \
  --conv_template "gemma_instruct" \
  --output_path "results/llava-gemma-2b-visualcorres.json"

python playground/attention_matrix_save_for_visualcorres.py \
  --model_path "Intel/llava-gemma-2b" \
  --model_name "llava_gemma" \
  --subtask Visual_Correspondence \
  --device "cuda:1" \
  --conv_template "gemma_instruct" \
  --output_path "results/llava-gemma-2b-visualcorres.npz"

Working models:
lmms-lab/llava-onevision-qwen2-0.5b-ov, qwen_2
liuhaotian/llava-v1.6-mistral-7b, mistral_instruct
liuhaotian/llava-v1.6-vicuna-7b, vicuna_v1
liuhaotian/llava-v1.6-vicuna-13b, vicuna_v1

Conversation templates issues with these models:
lmms-lab/llava-onevision-qwen2-7b-ov, qwen_2

These models have loading issues with the current version of code.
Intel/llava-gemma-2b, gemma_instruct
lmms-lab/LLaVA-OneVision-1.5-4B-Instruct, qwen_2_5
lmms-lab/LLaVA-OneVision-1.5-8B-Instruct, qwen_2_5