#!/usr/bin/env python3
"""
Analysis Evaluation Script for LLaVA-Alternating-Attn

This script evaluates the model's performance on the Blink benchmark,
which tests visual correspondence between images.
"""

# conda create -n temp python=3.12
# conda activate temp
# pip install torch torchvision huggingface_hub transformers datasets accelerate scikit-learn tqdm numpy


import os
import torch
import re
import json
import argparse
import numpy as np
from sklearn.metrics import confusion_matrix
from transformers import AutoProcessor, AutoModelForImageTextToText
from tqdm import tqdm
from datasets import load_dataset
from collections import Counter

def extract_and_validate_answer(response, correct_answer):
    # Strip the response of (, ), " ", \n, \t, etc.
    response = response.strip("()\"\n\t")
    # Remove common prefixes and convert to uppercase
    response = response.replace("Answer:", "").replace("ANSWER:", "").replace("Point", "").strip().upper()
    
    # Extract just the letter if it exists
    letter_match = re.search(r'[A-D]', response)
    if letter_match:
        response = letter_match.group()
    else:
        response = ""
    
    response = f"({response})" if response else "()"

    correct = response == correct_answer
    return correct, response


def create_confusion_matrix(results):
    """Build a confusion matrix keyed by ground-truth then prediction using sklearn."""
    labels = sorted({r["answer"] for r in results} | {r["correct_answer"] for r in results})
    y_true = [r["correct_answer"] for r in results]
    y_pred = [r["answer"] for r in results]

    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    return {
        true_label: {pred_label: int(matrix[i][j]) for j, pred_label in enumerate(labels)}
        for i, true_label in enumerate(labels)
    }


def evaluate(model, processor, item, image_token_id):
    """Evaluate the model on a single item from the dataset."""
    question_text = "Question: " + item["question"]
    detailed_prompt = "Details: " + item["prompt"]
    question_directive = "Answer with the option’s letter from the given choices directly."
    prompt = (
        question_text + "\n" +
        detailed_prompt + "\n" +
        question_directive + "\n"
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": item["image_1"]},
                {"type": "image", "image": item["image_2"]},
                {"type": "text", "text": prompt}
            ]
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    ).to(model.device, dtype=torch.bfloat16)
    modality_ids = torch.where(inputs.input_ids == image_token_id, 1, 0)
    input_prompt = processor.decode(inputs["input_ids"][0])

    # Generate outputs
    input_ids_len = inputs["input_ids"].shape[1]
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            do_sample=False,
            pad_token_id=processor.tokenizer.eos_token_id,
            max_new_tokens=256,
            use_cache=True,
        )
        output2 = model(
            **inputs,
            output_attentions=True,
            return_dict=True,
        )

    # Decode only the generated tokens (after conditioning inputs)
    response = processor.decode(
        output[0, input_ids_len:],
        skip_special_tokens=True
    )

    # Process the attention matrices
    attention_matrices = output2.attentions
    if attention_matrices:
        a_layer_numpy_list = []
        for a_layer in attention_matrices:
            # average over batch and head dimensions
            a_layer_numpy_list.append(a_layer.type(torch.float16).mean(axis=(0, 1)).detach().cpu().numpy())
        current_attn = np.stack(a_layer_numpy_list, axis=0).astype(np.float32)

        # Sum the attention weight that output token has on the image tokens
        img_ids = np.argwhere(modality_ids.detach().cpu().numpy() == 1)
        current_visual_attn = current_attn[:, -1, img_ids[:, 1]].sum(axis=1)

    return response, input_prompt, current_visual_attn

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_checkpoint", type=str, required=True, help="Path to model")
    parser.add_argument("--results_output_path", type=str, required=True, help="Path to results output")
    parser.add_argument("--attn_output_path", type=str, required=True, help="Path to attention output")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use")
    args = parser.parse_args()

    # Load processor + model
    processor = AutoProcessor.from_pretrained(args.model_checkpoint)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_checkpoint,
        device_map=args.device,
        output_attentions=True,
        dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    # Get the image token and its ID
    if "<image>" in processor.tokenizer.special_tokens_map["additional_special_tokens"]:
        # LLaVA & SmolVLM models use <image> as the image token
        image_token = "<image>"
    elif "<IMG_CONTEXT>" in processor.tokenizer.special_tokens_map["additional_special_tokens"]:
        # InternVL models use <IMG_CONTEXT> as the image token
        image_token = "<IMG_CONTEXT>"
    elif "<|image_pad|>" in processor.tokenizer.special_tokens_map["additional_special_tokens"]:
        # QwenVL models use <|image_pad|> as the image token
        image_token = "<|image_pad|>"
    else:
        raise ValueError("No image token found")
    image_token_id = processor.tokenizer.convert_tokens_to_ids(image_token)

    # Load dataset
    dataset = load_dataset("BLINK-Benchmark/BLINK", "Visual_Correspondence", split="val")

    results = []
    attention_weight_sums = []
    for item in tqdm(dataset, desc="Evaluating Blink"):
        response, prompt, attn = evaluate(model, processor, item, image_token_id)

        correct, response_extracted = extract_and_validate_answer(response, item["answer"])
        results.append({
            "id": item["idx"],
            "prompt": prompt,
            "raw_answer": response,
            "answer": response_extracted,
            "correct_answer": item["answer"],
            "correct": correct,
        })
        attention_weight_sums.append(attn)

    # Create confusion matrix
    confusion_matrix = create_confusion_matrix(results)
    print(f"Confusion matrix: {confusion_matrix}")

    # Print final accuracy
    correct_count = sum(result["correct"] for result in results)
    total_count = len(results)
    accuracy = correct_count / total_count
    print(f"Final accuracy: {accuracy:.2f}")

    # Distribution of predicted answers vs correct answers
    predicted_answers = [result["answer"] for result in results]
    correct_answers = [result["correct_answer"] for result in results]
    print(f"Distribution of predicted answers vs correct answers:")
    predicted_answers = sorted(Counter(predicted_answers).items(), key=lambda x: x[0])
    correct_answers = sorted(Counter(correct_answers).items(), key=lambda x: x[0])
    print(f"Predicted answers: {predicted_answers}")
    print(f"Correct answers: {correct_answers}")

    # Stack the attention weight sums
    visual_attention_weight_sums = np.stack(attention_weight_sums, axis=0)

    final_results = {
        "model_path": args.model_checkpoint,
        "confusion_matrix": confusion_matrix,
        "accuracy": accuracy,
        "predicted_answers": predicted_answers,
        "correct_answers": correct_answers,
        "results": results,
        "prediction_distribution": predicted_answers,
        "correct_distribution": correct_answers,
    }
    final_attn_dict = {
        "model_path": args.model_checkpoint,
        "visual_attention_weight_sums": visual_attention_weight_sums,
    }

    os.makedirs(os.path.dirname(args.results_output_path), exist_ok=True)
    with open(args.results_output_path, "w") as f:
        json.dump(final_results, f)
    np.savez_compressed(args.attn_output_path, **final_attn_dict)

if __name__ == "__main__":
    main()