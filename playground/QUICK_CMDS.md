python scripts/blink_eval.py \
  --model_path "liuhaotian/llava-v1.6-mistral-7b" \
  --model_name "llava_mistral" \
  --output_path "results/llava-v1.6-mistral-7b-visual_correspondence.json" \
  --subtask "Visual_Correspondence" \
  --device "cuda:1" \
  --conv_template "manual"

python llava/eval/blink_eval.py \
  --model_path "lmms-lab/llava-onevision-qwen2-0.5b-ov" \
  --model_name "llava_qwen"  \
  --subtask Visual_Correspondence \
  --device "cuda:1" \
  --conv_template "qwen_1_5"

model_path, model_name, conv_template
"liuhaotian/llava-v1.6-mistral-7b", "llava_mistral", "manual"
"lmms-lab/llava-onevision-qwen2-7b-ov", "llava_qwen", "qwen_1_5"
"lmms-lab/llava-onevision-qwen2-0.5b-ov", "llava_qwen", "qwen_1_5"

python llava/eval/blink_eval.py \
  --model_path "/workspace/checkpoints/llava-onevision-qwen2-0.5b-ov-with_alternating_attn" \
  --model_name "llava_qwen"  \
  --lora_weights_path "/workspace/checkpoints/llava-onevision-qwen2-0.5b-ov-with_alternating_attn-finetune/checkpoint-11000" \
  --subtask Visual_Correspondence \
  --device "cuda:1" \
  --conv_template "qwen_1_5"

<!-- Download the model -->
huggingface-cli download lmms-lab/llava-onevision-qwen2-0.5b-ov \
    --local-dir /workspace/checkpoints/llava-onevision-qwen2-0.5b-ov-with_alternating_attn \
    --local-dir-use-symlinks False


python scripts/blink_eval.py \
  --model_path "lmms-lab/llava-onevision-qwen2-7b-ov" \
  --model_name "llava_qwen" \
  --output_path "results/llava-onevision-qwen2-7b-ov-visual_correspondence.json" \
  --subtask "Visual_Correspondence" \
  --device "cuda:0" \
  --conv_template "qwen_1_5"
