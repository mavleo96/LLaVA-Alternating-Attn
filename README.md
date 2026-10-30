

# LLaVA-Alternating-Attn

This repo is a fork of **LLaVA-NeXT** adapted for the new **LLaVA-Alternating-Attn** architecture. It adds alternating within-modality and cross-modality attention schedules for Qwen2-based LLaVA models, plus utilities for BLINK evaluation and attention visualization.

## What’s here
- Alternating-attn language models: `llava_qwen_with_alternating_attn.py` (alternating within-modality masks) and `llava_qwen_with_alternating_cross_attn.py` (alternating cross-modality masks).
- Visual correspondence + BLINK eval:
  - `scripts/visualcorres_blink_eval.py`: LLM-based BLINK evaluation for LLaVA models (supports LoRA weights).
  - `scripts/analysis_eval.py`: BLINK evaluation and attention dumps for Hugging Face VLMs (e.g. SmolVLM, InternVL3, Qwen2-VL).
  - `scripts/visualcorres_eval_vision_encoder.py`: direct vision-encoder (CLIP/SigLIP) accuracy on Synthetic-Visual-Correspondence-Data.
  - `playground/attention_matrix_save_for_visualcorres.py`: attention dumps for LLaVA models.
- Finetuning: `scripts/train/finetune_alternating_attn.sh` (LLaVA-OneVision-Data subset) and `scripts/train/finetune_visualcorres.sh` (synthetic visual correspondence data), both LoRA on the language model.
- Quick commands and checkpoints: see `playground/QUICK_CMDS.md`.

## Setup
```bash
conda env create -f env.yml
conda activate llava
pip install -e .
```

## Getting checkpoints
- Alternating-Attn checkpoints: [mavleo96/LLaVA-Alternating-Attn](https://huggingface.co/mavleo96/LLaVA-Alternating-Attn)
  ```bash
  hf download mavleo96/LLaVA-Alternating-Attn \
    --local-dir /workspace/checkpoints/LLaVA-Alternating-Attn
  ```

## Alternating attention: how it works
- Layer schedule: even-numbered layers use the standard causal mask; odd-numbered layers use a modality-aware mask (combined with the causal mask); the final layer always uses the standard causal mask. See `llava_qwen_with_alternating_attn.py` and `llava_qwen_with_alternating_cross_attn.py`.
- Modality ids: `prepare_inputs_labels_for_multimodal` in `llava/model/llava_arch.py` returns `modality_ids` (text: 0, image/video: 1). Tokens generated at inference time are treated as text.
- Mask types (`mask_utils.py`):
  - `modality_ids_to_modality_attention_mask`: text attends only to text and image only to image (no cross-modal attention).
  - `modality_ids_to_cross_modality_attention_mask`: text attends only to image and image only to text, plus each token attends to itself.
- Only eager attention is supported (`attn_implementation="eager"`), since the custom masks are passed as 4D additive masks.

## Model selection
Model classes are chosen by substring match, so checkpoint paths and model names must follow these conventions:

| Variant | Training (`--model_name_or_path` contains) | Inference (`model_name`) |
| ------- | ------------------------------------------ | ------------------------ |
| Within-modality | `alternating-attn-within-modality` | `llava_qwen_with_alternating_attn` |
| Cross-modality | `alternating-attn-cross-modality` | `llava_qwen_with_alternating_cross_attn` |

## Directory highlights
- `llava/model/language_model/`: alternating-attn Qwen2 implementations and mask helpers.
- `playground/`: analysis scripts, attention visualization notebooks, notes, and quick command cheatsheet.
- `scripts/`: evaluation, data download, and training utilities.

## License
Apache 2.0 (upstream LLaVA-NeXT license applies).
