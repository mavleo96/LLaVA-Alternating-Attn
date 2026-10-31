#!/usr/bin/env python3
"""
Layer-wise linear probes for BLINK Visual Correspondence.

A frozen LLaVA-OneVision Qwen2 model gets one bias-free linear head per decoder layer that reads the
last prompt token's hidden state and predicts 4 logits (A, B, C, D). Probes are trained on
Synthetic-Visual-Correspondence-Data and evaluated on the BLINK Visual_Correspondence val split.
"""

import argparse
import csv
import json
import os

import torch
import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import process_images, tokenizer_image_token
from llava.model.builder import load_pretrained_model


LETTERS = ["A", "B", "C", "D"]

QUESTION_PROMPT = "Which point is corresponding to the reference point?"
VISUAL_CORRESPONDENCE_DESCRIBE_PROMPT = (
    "A point is circled on the first image, labeled with REF. We change the camera position "
    "or lighting and shoot the second image. You are given multiple red-circled points on the "
    "second image, choices of \"A, B, C, D\" are drawn beside each circle. Which point on the "
    "second image corresponds to the point in the first image?"
)
QUESTION_DIRECTIVE = "Answer with the option’s letter from the given choices directly."


class LinearProbeWrapper(nn.Module):
    def __init__(self, model, output_dim: int):
        super().__init__()
        self.model = model
        self.n_layers = model.config.num_hidden_layers
        self.linear_probes = nn.ModuleList([
            nn.Linear(model.config.hidden_size, output_dim, bias=False, device=model.device, dtype=model.dtype)
            for _ in range(self.n_layers)
        ])

    @torch.no_grad()
    def extract(self, **inputs):
        hidden_states = self.model(**inputs, output_hidden_states=True).hidden_states
        # hidden_states[0] is the embedding output; hidden_states[i + 1] is the output of decoder layer i
        return [hidden_states[i + 1][:, -1, :] for i in range(self.n_layers)]

    def forward(self, **inputs):
        features = self.extract(**inputs)
        output = []
        for i in range(self.n_layers):
            output.append(self.linear_probes[i](features[i]))
        return torch.stack(output, dim=1)


def row_inputs(row, prompt_ids, wrapper, image_processor, device):
    images = [row["image_1"].convert("RGB"), row["image_2"].convert("RGB")]
    image_tensors = process_images(images, image_processor, wrapper.model.config)
    return {
        "input_ids": prompt_ids,
        "images": [img.to(dtype=wrapper.model.dtype, device=device) for img in image_tensors],
        "image_sizes": [img.size for img in images],
        "modalities": ["image", "image"],
    }


def train(args, wrapper, prompt_ids, image_processor):
    dataset = load_dataset(
        "mavleo96/Synthetic-Visual-Correspondence-Data", data_files="parquet_data/samples_*.parquet", split="train"
    )
    # collate keeps the PIL images: the default collate cannot stack them, and the wrapper takes one row
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, collate_fn=lambda rows: rows,
        generator=torch.Generator().manual_seed(args.seed),
    )
    optimizer = torch.optim.AdamW(wrapper.linear_probes.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    train_acc = None
    for epoch in range(args.epochs):
        correct = torch.zeros(wrapper.n_layers)
        for rows in tqdm(loader, desc=f"Train epoch {epoch}"):
            optimizer.zero_grad()
            for row in rows:
                label = LETTERS.index(row["ref_label"].strip("()"))
                logits = wrapper(**row_inputs(row, prompt_ids, wrapper, image_processor, args.device))  # (1, n_layers, 4)
                target = torch.full((wrapper.n_layers,), label, device=args.device)
                # Sum of per-layer losses, averaged over the batch
                loss = loss_fn(logits.float().flatten(0, 1), target) * wrapper.n_layers / len(rows)
                loss.backward()
                correct += (logits.detach().argmax(-1)[0].cpu() == label)
            optimizer.step()
        train_acc = (correct / len(dataset)).tolist()
        print(f"epoch {epoch} | mean layer acc {sum(train_acc) / len(train_acc):.4f}")
    torch.save(wrapper.linear_probes.state_dict(), os.path.join(args.out_dir, "linear_probes.pt"))
    return train_acc


def evaluate(args, wrapper, prompt_ids, image_processor):
    dataset = load_dataset("BLINK-Benchmark/BLINK", args.subtask, split="val")
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=lambda rows: rows)
    correct = torch.zeros(wrapper.n_layers)
    for rows in tqdm(loader, desc="BLINK val"):
        row = rows[0]
        logits = wrapper(**row_inputs(row, prompt_ids, wrapper, image_processor, args.device))
        label = LETTERS.index(row["answer"].strip("()"))
        correct += (logits.argmax(-1)[0].cpu() == label)
    return (correct / len(dataset)).tolist(), len(dataset)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="lmms-lab/llava-onevision-qwen2-0.5b-ov")
    parser.add_argument("--model_name", type=str, default="llava_qwen")
    parser.add_argument("--conv_template", type=str, default="qwen_2")
    parser.add_argument("--subtask", type=str, default="Visual_Correspondence")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--torch_dtype", type=str, default="bfloat16")
    parser.add_argument("--output_dir", type=str, default="results/linear_probe")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    args.out_dir = os.path.join(args.output_dir, os.path.basename(args.model_path.rstrip("/")))
    os.makedirs(args.out_dir, exist_ok=True)

    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path=args.model_path,
        model_base=None,
        model_name=args.model_name,
        device_map=args.device,
        torch_dtype=args.torch_dtype,
        attn_implementation="eager",
    )
    model.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    # Matches scripts/visualcorres_blink_eval.py: one 384x384 crop per image instead of anyres tiling.
    model.config.image_aspect_ratio = "nobase"

    # Visual Correspondence Prompt
    conv = conv_templates[args.conv_template].copy()
    conv.append_message(conv.roles[0], (
        f"Image 1: {DEFAULT_IMAGE_TOKEN}\nImage 2: {DEFAULT_IMAGE_TOKEN}\n"
        f"Question: {QUESTION_PROMPT}\n"
        f"Details: {VISUAL_CORRESPONDENCE_DESCRIBE_PROMPT}\n"
        f"{QUESTION_DIRECTIVE}\n"
    ))
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    prompt_ids = tokenizer_image_token(
        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).to(args.device)

    wrapper = LinearProbeWrapper(model, output_dim=len(LETTERS))
    train_acc = train(args, wrapper, prompt_ids, image_processor)
    blink_acc, n_blink = evaluate(args, wrapper, prompt_ids, image_processor)

    rows = [
        {"layer": layer, "blink_val_acc": blink_acc[layer], "synthetic_train_acc": train_acc[layer]}
        for layer in range(wrapper.n_layers)
    ]
    for r in rows:
        print(f"layer {r['layer']:2d} | BLINK val acc {r['blink_val_acc']:.4f} | synth train acc {r['synthetic_train_acc']:.4f}")

    with open(os.path.join(args.out_dir, "per_layer_accuracy.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with open(os.path.join(args.out_dir, "results.json"), "w") as f:
        json.dump({"args": vars(args), "num_blink_val": n_blink, "per_layer": rows}, f, indent=2)
    print(f"Results written to {args.out_dir}")


if __name__ == "__main__":
    main()
