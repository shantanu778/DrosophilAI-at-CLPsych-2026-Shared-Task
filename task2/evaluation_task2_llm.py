#!/usr/bin/env python3
"""
Task 2 Evaluation — LLM approach
Usage:
    python evaluation_task2_llm.py --model_name ./phi4_task2-final --data_dir data/
"""

import argparse
import json
import re
import warnings
warnings.filterwarnings('ignore')
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from sklearn.metrics import (classification_report, f1_score,
                             precision_recall_fscore_support, confusion_matrix)

from dataset_task2_llm import (
    CLPsychTask2DataLoader, create_task2_dataset, df_to_training_format, LABEL_NAMES
)


def predict(instruction, input_text, model, tokenizer):
    messages = [
        {"role": "system", "content": instruction},
        {"role": "user",   "content": input_text}
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.1,
            top_p=0.9,
            do_sample=True,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    response = tokenizer.decode(
        outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True
    )
    # print(response)  # Debug: print raw model response
    return response


def parse_label(text):
    """Extract label from model output JSON"""
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group())
            label = obj.get('label', '').strip()
            if label in LABEL_NAMES:
                return label
        except:
            pass
    # Fallback: look for label keywords in raw text
    text_upper = text.upper()
    if 'SWITCH' in text_upper:
        return 'Switch'
    if 'ESCALATION' in text_upper:
        return 'Escalation'
    return 'Neither'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--data_dir",   type=str, default="data/")
    parser.add_argument("--max_history", type=int, default=1)
    parser.add_argument("--output_prefix", type=str, default=None)
    args = parser.parse_args()

    MODEL_PATH    = args.model_name
    OUTPUT_PREFIX = args.output_prefix or MODEL_PATH.strip("./").rstrip("/")

    # Load val data
    print("=" * 60)
    print("STEP 1 & 2: Loading Validation Data")
    print("=" * 60)

    val_loader = CLPsychTask2DataLoader(args.data_dir, split='val')
    val_df     = val_loader.load()
    val_loader.get_stats()

    val_inst_df = create_task2_dataset(val_df, max_history=args.max_history)
    val_data    = df_to_training_format(val_inst_df)
    print(f"\n✅ {len(val_data)} validation examples")

    # Load model
    print("\n" + "=" * 60)
    print("STEP 3: Loading Model")
    print("=" * 60)

    with open(f"{MODEL_PATH}/adapter_config.json") as f:
        base_model_name = json.load(f)["base_model_name_or_path"]
    print(f"Base model: {base_model_name}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.padding_side = "right"

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map="cuda",
        trust_remote_code=True,
    )
    if hasattr(base_model.config, 'sliding_window'):
        base_model.config.sliding_window = None
    try:
        for layer in base_model.model.layers:
            if hasattr(layer.self_attn, 'sliding_window'):
                layer.self_attn.sliding_window = None
    except AttributeError:
        pass

    model = PeftModel.from_pretrained(base_model, MODEL_PATH)
    model.eval()
    print("✅ Model loaded\n")

    # Generate predictions
    print("=" * 60)
    print("STEP 4: Generating Predictions")
    print("=" * 60)

    predictions = []
    y_true, y_pred_labels = [], []

    for item in tqdm(val_data, desc="Predicting"):
        try:
            response      = predict(item['instruction'], item['input'], model, tokenizer)
            pred_label    = parse_label(response)
            # print(f"GT: {item['label_str']} | Pred: {pred_label}")
            ground_truth  = item['label_str']

            predictions.append({
                'timeline_id':  item['timeline_id'],
                'post_id':      item['post_id'],
                'ground_truth': ground_truth,
                'prediction':   pred_label,
                'raw_response': response,
            })
            y_true.append(ground_truth)
            y_pred_labels.append(pred_label)

        except Exception as e:
            print(f"\n❌ Error: {e}")

    # Save predictions
    pred_file = f"task2_val_predictions_{OUTPUT_PREFIX}.json"
    with open(pred_file, 'w') as f:
        json.dump(predictions, f, indent=2)
    print(f"\n✅ Predictions saved to {pred_file}")

    # Metrics
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    print(classification_report(y_true, y_pred_labels,
                                zero_division=0))

    macro_f1 = f1_score(y_true, y_pred_labels, average='macro', zero_division=0)
    micro_f1 = f1_score(y_true, y_pred_labels, average='micro', zero_division=0)

    print(f"Macro-F1: {macro_f1:.4f}  |  Micro-F1: {micro_f1:.4f}")

    # Confusion matrix
    print("\nConfusion Matrix (rows=true, cols=pred):")
    cm = confusion_matrix(y_true, y_pred_labels)
    print(f"{'':>12}", end="")
    for l in LABEL_NAMES: print(f"{l:>12}", end="")
    print()
    for i, l in enumerate(LABEL_NAMES):
        print(f"{l:>12}", end="")
        for j in range(len(LABEL_NAMES)):
            print(f"{cm[i,j]:>12}", end="")
        print()

    # Per-class results
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred_labels,
        average=None, zero_division=0
    )
    results = {
        LABEL_NAMES[i]: {
            'precision': float(p[i]), 'recall': float(r[i]),
            'f1': float(f[i]), 'support': int(s[i])
        }
        for i in range(len(LABEL_NAMES))
    }
    results['macro_f1'] = float(macro_f1)
    results['micro_f1'] = float(micro_f1)

    results_file = f"task2_results_{OUTPUT_PREFIX}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Results saved to {results_file}")