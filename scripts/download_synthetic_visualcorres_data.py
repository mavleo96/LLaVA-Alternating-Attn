"""Convert Synthetic-Visual-Correspondence-Data parquet shards into LLaVA-style JSON + JPEGs.

This script:
  1) Loads the synthetic visual correspondence dataset from its parquet files.
  2) Saves paired reference/candidate images to disk as JPEGs.
  3) Builds LLaVA-style instruction data where the human turn contains two image
     tokens plus a multiple-choice question (A–D), and the assistant turn is the
     ground-truth option letter from `ref_label`.
The resulting JSON file can be used for finetuning or evaluation-style prompts.
"""

import os
import json
import argparse
from datasets import load_dataset
from llava.constants import DEFAULT_IMAGE_TOKEN
from tqdm import tqdm


DATA_FILES = (
    "hf://datasets/mavleo96/"
    "Synthetic-Visual-Correspondence-Data/parquet_data/samples_*.parquet"
)


def main():
    parser = argparse.ArgumentParser(description="Convert synthetic visual correspondence parquet data to LLaVA JSON.")
    parser.add_argument(
        "--image_folder",
        type=str,
        default="/data/synthetic_visualcorres/images",
        help="Folder to save JPEG images for the synthetic visual correspondence dataset.",
    )
    args = parser.parse_args()

    # Load dataset (same parquet shards as the vision-encoder eval script).
    dataset = load_dataset("parquet", data_files=DATA_FILES, split="train")

    image_folder = args.image_folder
    os.makedirs(image_folder, exist_ok=True)

    image_prompt = f"Image 1: {DEFAULT_IMAGE_TOKEN}\n Image 2: {DEFAULT_IMAGE_TOKEN}\n "
    question_text = "Question: Which point is corresponding to the reference point?"
    detailed_prompt = (
        "Details: A point is circled on the first image, labeled with REF. We change the camera position or "
        "lighting and shoot the second image. You are given multiple red-circled points on the second image, "
        'choices of "A, B, C, D" are drawn beside each circle. Which point on the second image corresponds to '
        "the point in the first image? Select from the following options.\n(A) Point A\n(B) Point B\n(C) Point C\n(D) Point D"
    )
    directive = "Answer with the option’s letter from the given choices directly."

    converted_data = []
    for da in tqdm(dataset):
        json_data = {}
        json_data["id"] = da["id"]

        # Save images
        stem, _ = os.path.splitext(da["id"])
        out_path_1 = os.path.join(image_folder, f"{stem}_1.jpg")
        out_path_2 = os.path.join(image_folder, f"{stem}_2.jpg")
        img_1 = da["image_1"]
        img_2 = da["image_2"]
        if img_1.mode != "RGB":
            img_1 = img_1.convert("RGB")
        img_1.save(out_path_1, format="JPEG")
        if img_2.mode != "RGB":
            img_2 = img_2.convert("RGB")
        img_2.save(out_path_2, format="JPEG")
        json_data["image"] = [out_path_1, out_path_2]

        # Save conversations
        json_data["conversations"] = [
            {
                "from": "human",
                "value": image_prompt + "\n" + question_text + "\n" + detailed_prompt + "\n" + directive,
            },
            {"from": "gpt", "value": f"({da['ref_label']})"},
        ]

        # Save data
        converted_data.append(json_data)

    with open(os.path.join(os.path.dirname(image_folder), "synthetic_visualcorres_data.json"), "w") as f:
        json.dump(converted_data, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    main()

