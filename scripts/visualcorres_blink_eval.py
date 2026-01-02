#!/usr/bin/env python3
"""
Blink Benchmark Evaluation Script for LLaVA-Alternating-Attn

This script evaluates the model's performance on the Blink benchmark,
which tests visual correspondence between images.
"""

import argparse
import json
import re
import os
import torch
from tqdm import tqdm
from datasets import load_dataset
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token, process_images
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from llava.utils import disable_torch_init
from peft import PeftModel

from collections import Counter

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.modules.module")
from sklearn.metrics import confusion_matrix


VISUAL_CORRESPONDENCE_DESCRIBE_PROMPT = (
    "A point is circled on the first image, labeled with REF. We change the camera position "
    "or lighting and shoot the second image. You are given multiple red-circled points on the "
    "second image, choices of \"A, B, C, D\" are drawn beside each circle."
)

SEMATIC_CORRESPONDENCE_DESCRIBE_PROMPT = (
    "Humans can find corresponding points for different objects in the same category. For instance, "
    "if there are images of two different cats, then the left ear tip of one cat corresponds to the "
    "left ear tip of the other cat, and the right front paw of one cat corresponds to the right "
    "front paw of the other cat. Given the following two images, a reference point is annotated "
    "on the first image, labeled with REF. You are given multiple red-circled points on the second image, "
    "choices of \"A, B, C, D\" are drawn beside each circle."
)

DESCRIBE_DIRECTIVE = (
    "Describe the reference point in first image and each of the red-circled points labeled with "
    "A, B, C, D in the second image separately."
)

QUESTION_DIRECTIVE = "Answer with the option’s letter from the given choices directly."

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


def evaluate(model, tokenizer, image_processor, item, conv_template, device):
    image_prompt = "".join(f"Image {i+1}: {DEFAULT_IMAGE_TOKEN}\n" for i in range(4) if item[f"image_{i+1}"] is not None)
    question_text = "Question: " + item["question"]
    detailed_prompt = "Details: " + item["prompt"]
    if conv_template == "manual":
        prompt = (
            image_prompt + "\n" +
            question_text + "\n" +
            detailed_prompt + "\n" +
            QUESTION_DIRECTIVE + "\n"
        )
    else:
        conv = conv_templates[conv_template].copy()
        conv.append_message(conv.roles[0], image_prompt + "\n" + question_text + "\n" + detailed_prompt + "\n" + QUESTION_DIRECTIVE)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
    input_ids = input_ids.unsqueeze(0).to(device)

    images = [
        item[f"image_{i+1}"].convert('RGB')
        for i in range(4) if item[f"image_{i+1}"] is not None
    ]
    image_tensors = process_images(images, image_processor, model.config)
    image_tensors = [_image.to(dtype=torch.float16, device=device) for _image in image_tensors]
    image_sizes = [image.size for image in images]

    attention_mask = torch.ones_like(input_ids)
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    output = model.generate(
        input_ids,
        attention_mask=attention_mask,
        pad_token_id=pad_token_id,
        images=image_tensors,
        image_sizes=image_sizes,
        modalities=["image"],
        do_sample=False,
        max_new_tokens=256,
        use_cache=True,
        output_attentions=False,
        return_dict_in_generate=True,
    )
    response = tokenizer.batch_decode(output.sequences, skip_special_tokens=True)[0]

    return response, prompt

def evaluate_with_query_expansion(model, tokenizer, image_processor, item, conv_template, subtask, device):
    image_prompt = "".join(f"Image {i+1}: {DEFAULT_IMAGE_TOKEN}\n" for i in range(4) if item[f"image_{i+1}"] is not None)
    question_text = "Question: " + item["question"]
    detailed_describe_prompt = {
        "Visual_Correspondence": VISUAL_CORRESPONDENCE_DESCRIBE_PROMPT,
        "Semantic_Correspondence": SEMATIC_CORRESPONDENCE_DESCRIBE_PROMPT
    }[subtask]

    # Describe the images
    if conv_template == "manual":
        describe_prompt = (
            image_prompt + "\n" +
            detailed_describe_prompt + "\n" +
            DESCRIBE_DIRECTIVE + "\n"
        )
    else:
        conv = conv_templates[conv_template].copy()
        conv.append_message(conv.roles[0], image_prompt + "\n" + detailed_describe_prompt + "\n" + DESCRIBE_DIRECTIVE)
        conv.append_message(conv.roles[1], None)
        describe_prompt = conv.get_prompt()

    describe_input_ids = tokenizer_image_token(describe_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
    describe_input_ids = describe_input_ids.unsqueeze(0).to(device)

    images = [
        item[f"image_{i+1}"].convert('RGB')
        for i in range(4) if item[f"image_{i+1}"] is not None
    ]
    image_tensors = process_images(images, image_processor, model.config)
    image_tensors = [_image.to(dtype=torch.float16, device=device) for _image in image_tensors]
    image_sizes = [image.size for image in images]

    describe_attention_mask = torch.ones_like(describe_input_ids)
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    describe_output = model.generate(
        describe_input_ids,
        attention_mask=describe_attention_mask,
        pad_token_id=pad_token_id,
        images=image_tensors,
        image_sizes=image_sizes,
        modalities=["image"],
        do_sample=False,
        max_new_tokens=1024,
        use_cache=True,
        output_attentions=False,
        return_dict_in_generate=True,
    )
    describe_response = tokenizer.batch_decode(describe_output.sequences, skip_special_tokens=True)[0]

    # Describe the images
    if conv_template == "manual":
        prompt = (
            describe_prompt + "\n" +
            describe_response + "\n" +
            question_text + "\n" +
            QUESTION_DIRECTIVE + "\n"
        )
    else:
        conv = conv_templates[conv_template].copy()
        conv.append_message(conv.roles[0], image_prompt + "\n" + describe_prompt + "\n" + DESCRIBE_DIRECTIVE + "\n" + describe_response + "\n" + question_text + "\n" + QUESTION_DIRECTIVE)
        conv.append_message(conv.roles[1], describe_response)
        conv.append_message(conv.roles[0], question_text + "\n" + QUESTION_DIRECTIVE)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
    input_ids = input_ids.unsqueeze(0).to(device)

    answer_attention_mask = torch.ones_like(input_ids)

    output = model.generate(
        input_ids,
        attention_mask=answer_attention_mask,
        pad_token_id=pad_token_id,
        images=image_tensors,
        image_sizes=image_sizes,
        modalities=["image"],
        do_sample=False,
        max_new_tokens=256,
        use_cache=True,
        output_attentions=False,
        return_dict_in_generate=True,
    )
    response = tokenizer.batch_decode(output.sequences, skip_special_tokens=True)[0]

    return response, prompt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to model")
    parser.add_argument("--model_name", type=str, default="llava_mistral", help="Model name")
    parser.add_argument("--lora_weights_path", type=str, default=None, help="Path to lora weights")
    parser.add_argument("--output_path", type=str, default="blink_results.json", help="Output path for results")
    parser.add_argument("--conv_template", type=str, default="mistral_instruct", help="Conversation template")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use")
    parser.add_argument("--load_8bit", action="store_true", help="Load model in 8bit")
    parser.add_argument("--load_4bit", action="store_true", help="Load model in 4bit")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples to evaluate")
    parser.add_argument("--subtask", type=str, default="Visual_Correspondence", help="BLINK subtask")
    parser.add_argument("--query_expansion", action="store_true", help="Use query expansion")
    args = parser.parse_args()
    
    # Disable torch init for faster loading
    disable_torch_init()

    if args.subtask not in ["Visual_Correspondence", "Semantic_Correspondence"]:
        raise ValueError(f"Subtask {args.subtask} not supported")
    
    # Load model
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path=args.model_path,
        model_base=None,
        model_name=args.model_name,
        load_8bit=args.load_8bit,
        load_4bit=args.load_4bit,
        device_map=args.device,
        torch_dtype="float16",
        attn_implementation="eager"
    )
    # Ensure padding is defined for generation
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    # Note: using anyres increase the memory usage and sometimes not suitable for multi-image setting.
    model.config.image_aspect_ratio = "nobase"
    if args.lora_weights_path:
        model = PeftModel.from_pretrained(model, args.lora_weights_path)
        model = model.merge_and_unload()
    
    # Load dataset
    dataset = load_dataset("BLINK-Benchmark/BLINK", args.subtask, split="val")
    if args.max_samples:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))
    
    results = []
    for item in tqdm(dataset, desc="Evaluating Blink"):
        if args.query_expansion:
            response, prompt = evaluate_with_query_expansion(model, tokenizer, image_processor, item, args.conv_template, args.subtask, args.device)
        else:
            response, prompt = evaluate(model, tokenizer, image_processor, item, args.conv_template, args.device)
    
        correct, response_extracted = extract_and_validate_answer(response, item["answer"])
        results.append({
            "id": item["idx"],
            "prompt": prompt,
            "raw_answer": response,
            "answer": response_extracted,
            "correct_answer": item["answer"],
            "correct": correct,
        })

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

    final_results = {
        "confusion_matrix": confusion_matrix,
        "accuracy": accuracy,
        "predicted_answers": predicted_answers,
        "correct_answers": correct_answers,
        "results": results,
        "prediction_distribution": predicted_answers,
        "correct_distribution": correct_answers,
    }

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(final_results, f)


if __name__ == "__main__":
    main()

