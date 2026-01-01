"""Evaluate visual correspondence on Synthetic-Visual-Correspondence-Data with CLIP/SigLIP.

The script loads parquet shards of the dataset, crops one reference patch from image_1
and multiple candidate patches from image_2 per example, and embeds all patches with a
vision encoder (CLIP or SigLIP). Embeddings are L2-normalized so that dot products
between embeddings correspond to cosine similarities.

For each example, the reference embedding is compared with all candidate embeddings,
the candidate with maximum cosine similarity is selected, and the prediction is counted
as correct if it matches the ground-truth label. The final reported metric is the
top-1 accuracy: correct / total over all evaluated examples.
"""

import argparse
from functools import partial
import json
from typing import Callable, Iterable, List, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from datasets import load_dataset
from transformers import AutoProcessor, CLIPModel, CLIPProcessor, SiglipVisionModel


# Default to loading parquet shards directly from the HF dataset repo.
DATA_FILES = (
    "hf://datasets/mavleo96/"
    "Synthetic-Visual-Correspondence-Data/parquet_data/samples_*.parquet"
)
CLIP_NAME = "openai/clip-vit-large-patch14-336"
SIGLIP_NAME = "google/siglip-so400m-patch14-384"

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)


def build_dataset(num_samples: int):
    """Load the synthetic visual correspondence dataset and optionally subsample."""
    ds = load_dataset("parquet", data_files=DATA_FILES, split="train")
    if num_samples is not None and num_samples < len(ds):
        ds = ds.shuffle(seed=42).select(range(num_samples))
    print("Using subset size:", len(ds))
    return ds


@torch.no_grad()
def embed_images(
    images: List[Image.Image],
    model: torch.nn.Module,
    processor,
    feature_fn,
) -> torch.Tensor:
    """Return L2-normalized image embeddings for a batch of patches."""
    images = [im if isinstance(im, Image.Image) else Image.fromarray(np.array(im)) for im in images]
    inputs = processor(images=images, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    if device == "cuda":
        with torch.cuda.amp.autocast():
            feats = feature_fn(model, **inputs)
    else:
        feats = feature_fn(model, **inputs)
    feats = feats.float()
    return feats / feats.norm(dim=-1, keepdim=True)


def crop_and_pad(pil_img: Image.Image, center_xy: Tuple[float, float], patch_size: int) -> Image.Image:
    """Center crop with padding to keep patch_size square."""
    if not isinstance(pil_img, Image.Image):
        pil_img = Image.fromarray(np.array(pil_img))

    w, h = pil_img.size
    x, y = float(center_xy[0]), float(center_xy[1])

    half = patch_size // 2
    x0, y0 = int(round(x - half)), int(round(y - half))
    x1, y1 = x0 + patch_size, y0 + patch_size

    ix0, iy0 = max(0, x0), max(0, y0)
    ix1, iy1 = min(w, x1), min(h, y1)

    if ix0 >= ix1 or iy0 >= iy1:
        return Image.new("RGB", (patch_size, patch_size), color=(127, 127, 127))

    cropped = pil_img.crop((ix0, iy0, ix1, iy1))
    out = Image.new("RGB", (patch_size, patch_size), color=(127, 127, 127))
    out.paste(cropped, (ix0 - x0, iy0 - y0))
    return out


def evaluate_visual_correspondence(
    examples: Iterable[dict],
    embed_fn: Callable[[List[Image.Image]], torch.Tensor],
    patch_size: int,
    batch_size: int,
) -> dict:
    """Compute accuracy and bookkeeping stats in mini-batches.

    Assumes the Synthetic-Visual-Correspondence-Data schema:
      - labels_dict: JSON string mapping label -> [x, y] in image_2
      - ref_label: label string (e.g. "A", "B", "C", "D")
      - ref_label_pos: JSON list [x, y] in image_1
    See the module docstring for a high-level description of the evaluation math.
    """
    correct = total = skipped = invalid = 0

    # Accumulate examples into mini-batches for efficient encoder calls.
    ref_batch: List[Image.Image] = []
    cand_batch: List[Image.Image] = []
    cand_slices_batch: List[Tuple[int, int]] = []
    gt_indices_batch: List[int] = []

    def flush_batch():
        nonlocal correct, ref_batch, cand_batch, cand_slices_batch, gt_indices_batch
        if not ref_batch:
            return
        ref_feats = embed_fn(ref_batch)
        cand_feats = embed_fn(cand_batch)
        for i, (slice_start, slice_end) in enumerate(cand_slices_batch):
            sims = (ref_feats[i : i + 1] @ cand_feats[slice_start:slice_end].T).squeeze(0)
            if int(torch.argmax(sims).item()) == gt_indices_batch[i]:
                correct += 1
        # Reset for next mini-batch
        ref_batch = []
        cand_batch = []
        cand_slices_batch = []
        gt_indices_batch = []

    total_len = len(examples) if hasattr(examples, "__len__") else None
    for ex in tqdm(examples, desc="Evaluating visual correspondence", total=total_len):
        try:
            # Parse fields from the known synthetic dataset schema.
            labels_raw = ex.get("labels_dict")
            ref_label_name = ex.get("ref_label")
            ref_pos_raw = ex.get("ref_label_pos")

            if labels_raw is None or ref_label_name is None or ref_pos_raw is None:
                skipped += 1
                continue

            labels_dict = json.loads(labels_raw) if isinstance(labels_raw, str) else labels_raw
            ref_pos = json.loads(ref_pos_raw) if isinstance(ref_pos_raw, str) else ref_pos_raw

            if not isinstance(labels_dict, dict):
                invalid += 1
                continue

            cand_labels = list(labels_dict.keys())
            cand_centers = [labels_dict[k] for k in cand_labels]
            if str(ref_label_name) not in cand_labels:
                skipped += 1
                continue

            gt_idx = cand_labels.index(str(ref_label_name))
            if not isinstance(ref_pos, (list, tuple)) or len(ref_pos) < 2:
                invalid += 1
                continue
            ref_center = (float(ref_pos[0]), float(ref_pos[1]))

            img_ref, img_cand = ex.get("image_1"), ex.get("image_2")
            if img_ref is None or img_cand is None:
                skipped += 1
                continue

            ref_patch = crop_and_pad(img_ref, ref_center, patch_size)
            cand_patches = [crop_and_pad(img_cand, c, patch_size) for c in cand_centers]
            if not cand_patches:
                skipped += 1
                continue

            slice_start = len(cand_batch)
            cand_batch.extend(cand_patches)
            slice_end = len(cand_batch)

            ref_batch.append(ref_patch)
            cand_slices_batch.append((slice_start, slice_end))
            gt_indices_batch.append(gt_idx)
            total += 1

            if len(ref_batch) >= batch_size:
                flush_batch()
        except Exception:
            invalid += 1

    # Flush any remaining examples.
    flush_batch()

    return {
        "accuracy": (correct / total) if total else 0.0,
        "total": total,
        "correct": correct,
        "skipped": skipped,
        "invalid": invalid,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["clip", "siglip"], default="clip")
    parser.add_argument("--patch_size", type=int, default=256)
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    subset = build_dataset(args.num_samples)

    if args.model == "clip":
        model = CLIPModel.from_pretrained(CLIP_NAME).to(device)
        processor = CLIPProcessor.from_pretrained(CLIP_NAME)
        feature_fn = lambda m, **inp: m.get_image_features(**inp)
    else:
        processor = AutoProcessor.from_pretrained(SIGLIP_NAME)
        model = SiglipVisionModel.from_pretrained(SIGLIP_NAME).to(device)
        feature_fn = lambda m, **inp: m(**inp).pooler_output
    model.eval()
    embed_fn = partial(embed_images, model=model, processor=processor, feature_fn=feature_fn)

    result = evaluate_visual_correspondence(
        subset,
        embed_fn,
        patch_size=args.patch_size,
        batch_size=args.batch_size,
    )
    print(
        f"\nResult: accuracy={result['accuracy']:.4f}, correct={result['correct']}, "
        f"total={result['total']}, skipped={result['skipped']}, invalid={result['invalid']}"
    )


if __name__ == "__main__":
    main()
