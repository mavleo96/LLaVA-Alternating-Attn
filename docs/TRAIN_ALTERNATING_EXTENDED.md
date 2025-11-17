# Training wrapper: extended notes

This file documents new flags added to `train_alternating.py` and shows examples for alternating-attention and alternating-cross-attention variants.

### Flags

- `--variant` : one of `auto|alternating|alternating_cross|base`. Use this to assert which model variant expect to train. The wrapper uses `config.model_type` to check the checkpoint and will warn or exit if it doesn't match (use `--force` to override).
- `--init_from_backbone` : instantiate the alternating-attn model class and initialize it from a backbone LM checkpoint by copying matching LM parameters (name + shape) only.
- `--backbone_model` : path or HF repo id for the backbone LM to use when `--init_from_backbone` is set.
- `--inspect_data` : load the dataset lazily and print counts + first-sample preview (safe, no heavy image transforms).
- `--inspect_weights` : when used with `--init_from_backbone`, perform the selective-copy of matching parameters and print matched / missing key statistics, then exit (no training).
- `--dry_run` : load and print `config.model_type` and exit.
- `--throughput_log_steps` : attach a small callback that prints samples/sec and GPU memory every N steps (only active for the wrapper's init-from-backbone flow by default).

### Quick decision flow (human steps)

1. Confirm your data manifest is in an accepted LLAVA format (json/jsonl/yaml list). Use `--inspect_data` to preview counts and the first sample.
2. If have an alternating-attn checkpoint, use it directly with `--model_name_or_path` and optionally `--variant alternating`.
3. If it only have a backbone LM checkpoint, set `--init_from_backbone --backbone_model <path-or-repo>` and use `--inspect_weights` first to see which LM parameters will be copied into the alternating model.
4. Once inspection looks good, run the full training command. The wrapper will build the data module using the repo's `make_supervised_data_module()` and then call into the existing training loop.

### Examples — commands you can copy

Dry-run to see model type (fast):

```bash
python train_alternating.py \
  --model_name_or_path lmms-lab/llava-onevision-qwen2-0.5b-ov \
  --dry_run
```

Inspect your dataset (count + raw sample preview):

```bash
python train_alternating.py \
  --model_name_or_path lmms-lab/llava-onevision-qwen2-0.5b-ov \
  --data_path /path/to/laion_subset.json \
  --inspect_data
```

Inspect a backbone -> alternating partial weight mapping (no training):

```bash
python train_alternating.py \
  --backbone_model qwen2-base-or-local \
  --data_path /path/to/laion_subset.json \
  --variant alternating \
  --init_from_backbone \
  --inspect_weights
```

Train starting from an alternating checkpoint (recommended if available):

```bash
python train_alternating.py \
  --model_name_or_path path_or_repo_with_alternating_attn \
  --data_path /path/to/laion_subset_in_llava_format.json \
  --output_dir outputs/alt_qwen2 \
  --variant alternating \
  --per_device_train_batch_size 8 \
  --num_train_epochs 3
```

Train by initializing alternating-attn from a backbone LM (safer partial init):

```bash
python train_alternating.py \
  --backbone_model path_or_repo_for_backbone \
  --data_path /path/to/laion_subset_in_llava_format.json \
  --init_from_backbone \
  --variant alternating \
  --output_dir outputs/alt_qwen2_from_backbone \
  --per_device_train_batch_size 8 \
  --num_train_epochs 3 \
  --throughput_log_steps 50
```

Notes on throughput callback: set `--throughput_log_steps N` to print simple metrics (samples/sec, GPU memory) every N steps. This helps monitor progress early in training.

### What `--init_from_backbone` does (human explanation)

- The wrapper creates the alternating-attn model object (the class is registered in the repo). That model has extra multimodal modules (projectors, resamplers) and possibly slightly different parameter names.
- We then load the backbone LM checkpoint's state dict and copy only those parameters that:
  1) exist in the alternating model's state dict, and
  2) have an identical tensor shape.
- We explicitly skip multimodal-specific parameter groups such as `mm_projector`, `vision_resampler`, and similar modules to avoid shape or semantic mismatches.
- After copying the matching parameters we call `load_state_dict(..., strict=False)` so remaining keys (uninitialized multimodal heads, new layers) are left to random initialization and trained from scratch.

### Short Python snippet: selective-copy logic (what the wrapper does)

```python
# pseudo-code (and actual logic used by the wrapper)
from collections import OrderedDict

# alt_state = alt_model.state_dict()
# backbone_state = torch.load(backbone_ckpt, map_location='cpu')

matched = OrderedDict()
for k, v in backbone_state.items():
    if k in alt_state and alt_state[k].shape == v.shape:
        # skip explicitly multimodal layers by prefix
        if k.startswith('mm_projector') or k.startswith('vision_resampler'):
            continue
        matched[k] = v

alt_model.load_state_dict(matched, strict=False)

print(f"copied {len(matched)} params from backbone into alternating model")
```

### `--inspect_weights` purpose

When you run `--inspect_weights` the wrapper performs the same selective-copy process but does not start training: it prints the number of matched keys, lists a small sample of matched keys, and shows which alternating-model keys are still missing (uninitialized). This makes it easy to decide whether you want to proceed.

### Troubleshooting & tips

- If `--dry_run` reports a `config.model_type` that doesn't match `--variant`, double-check whether the checkpoint repo contains multiple model types (some repos provide both full and variant checkpoints). Use `--force` to continue anyway but be aware of mismatch risks.
- If too few LM parameters are matched during `--init_from_backbone`, check that the backbone LM is exactly the same family (Qwen2 0.5B vs another size may have mismatched layer counts or widths).
- If you plan to do many experiments, create a small 10-100 sample subset and run `--init_from_backbone --inspect_weights` on it locally to validate init behavior before committing CPU/GPU hours.

---

If you'd like, I can also:

- Add an automated 1-step dry-run that loads full weights locally and prints the Python class instantiated (helpful when a repo hosts more than one model type).
- Add the throughput callback into the repo's main `train.py` so it runs for both the wrapper and upstream entrypoints.

