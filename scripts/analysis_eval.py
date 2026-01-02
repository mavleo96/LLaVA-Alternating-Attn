import os
import torch
import re
import json

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


def evaluate(model, processor, item):

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

    # Decode only the **generated** tokens (after conditioning inputs)
    response = processor.decode(
        output[0, input_ids_len:],
        skip_special_tokens=True
    )

    input_prompt = processor.decode(inputs["input_ids"][0])
    
    return response, input_prompt

def main():
    # Choose a model checkpoint
    # model_checkpoint = "OpenGVLab/InternVL3-8B-hf"
    model_checkpoint = "Qwen/Qwen2-VL-7B-Instruct"
    output_path = "results/qwen2-vl-7b-instruct-visualcorres.json"
    device = "cuda:0"

    # Load processor + model
    processor = AutoProcessor.from_pretrained(model_checkpoint)
    model = AutoModelForImageTextToText.from_pretrained(
        model_checkpoint,
        device_map=device,
        dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    # Load dataset
    dataset = load_dataset("BLINK-Benchmark/BLINK", "Visual_Correspondence", split="val")

    results = []
    for item in tqdm(dataset, desc="Evaluating Blink"):
        response, prompt = evaluate(model, processor, item)

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

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(final_results, f)

if __name__ == "__main__":
    main()