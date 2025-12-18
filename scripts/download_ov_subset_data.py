"""Download and convert a subset of LLaVA-OneVision-Data to images + conversations JSON.

Usage:
    python scripts/download_ov_subset_data.py \\
        --subset_name visual7w(cauldron,llava_format) \\
        --subset_json_name ov_visual7w_cauldron_llava_format.json \\
        --image_folder /workspace/data/LLaVA-OneVision-Data

Arguments:
    --subset_name: Config name passed to `load_dataset` for selecting the subset.
    --subset_json_name: Name of the output JSON file containing the converted samples.
    --image_folder: Root directory where JPEG images and the JSON file will be written.
"""

import os
from datasets import load_dataset
from tqdm import tqdm
import json
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset_name", type=str, required=True)
    parser.add_argument("--subset_json_name", type=str, required=True)
    parser.add_argument("--image_folder", type=str, required=True)
    args = parser.parse_args()

    data = load_dataset(
        "lmms-lab/LLaVA-OneVision-Data",
        args.subset_name,
        split="train",
    )

    converted_data = []
    for da in tqdm(data):
        json_data = {}
        json_data["id"] = da["id"]
        if da["image"] is not None:
            # Normalize to a .jpg path and ensure parent directories exist
            stem, _ = os.path.splitext(da["id"])
            json_data["image"] = f"{stem}.jpg"
            out_path = os.path.join(args.image_folder, json_data["image"])
            os.makedirs(os.path.dirname(out_path), exist_ok=True)

            img = da["image"]
            # JPEG only supports RGB; convert any other mode (RGBA, palette, grayscale, etc.) to RGB
            if img.mode != "RGB":
                img = img.convert("RGB")
            img.save(out_path, format="JPEG")
        json_data["conversations"] = da["conversations"]
        converted_data.append(json_data)

    with open(os.path.join(args.image_folder, args.subset_json_name), "w") as f:
        json.dump(converted_data, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    main()
