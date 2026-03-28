#!/usr/bin/env python3
"""
Generate predictions on validation dataset and calculate metrics
"""

import json
import warnings
warnings.filterwarnings('ignore')
import argparse
import torch
from unsloth import FastLanguageModel
from tqdm import tqdm
import re
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, classification_report
from dataset import CLPsychDataLoader, create_instruction_dataset, df_to_training_format


parser = argparse.ArgumentParser()
parser.add_argument("--model_name", help="Exact location of Pretrained Models",
                type=str)
args = parser.parse_args()
MODELS = args.model_name

# ========== STEP 1: Load Data ==========
print("=" * 60)
print("STEP 1: Loading Data")
print("=" * 60)
val_loader = CLPsychDataLoader('tasks12/', split='val')
val_df = val_loader.load()
print("\n" + "=" * 60)
print("Validation Set Stats")
val_loader.verify_order()
val_loader.get_stats()
# ========== STEP 2: Create Instruction Dataset ==========
print("\n" + "=" * 60)
print("STEP 2: Creating Instruction Dataset")
print("=" * 60)

val_df = create_instruction_dataset(val_df)

val_data = df_to_training_format(val_df[:10])

# Create instruction dataset
# instruction_df = create_instruction_dataset(df)

# Verify first few examples
print("\n=== First 3 Examples ===")
for idx in range(min(3, len(val_df))):
    row = val_df.iloc[idx]
    print(f"\nExample {idx+1}:")
    print(f"Timeline: {row['timeline_id']}, Post: {row['post_index']}")
    print(f"Input: {row['input'][:100]}...")
    print(f"Output: {row['output'][:100]}...")

# ========== 1. Load Trained Model ==========
print("="*60)
print("Loading trained model...")
print("="*60)

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODELS,  # Your checkpoint directory
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)

FastLanguageModel.for_inference(model)
print("✅ Model loaded\n")

# ========== 2. Load Validation Data ==========
print("="*60)
print("Loading validation data...")
print("="*60)

# Load your val_data (should have instruction, input, output format)
# with open('val_data.json', 'r') as f:
#     val_data = json.load(f)

print(f"✅ Loaded {len(val_data)} validation examples\n")

# ========== 3. Generate Predictions ==========
print("="*60)
print("Generating predictions...")
print("="*60)

def predict_abcd(instruction, post_text, model, tokenizer):
    """Generate ABCD prediction - FIXED version"""
    
    messages = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": post_text}
    ]
    
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    # print(prompt)
    
    inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.1,
        top_p=0.9,
        do_sample=True,
        use_cache=False,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    response = tokenizer.decode(
        outputs[0][inputs['input_ids'].shape[1]:],
        skip_special_tokens=True
    )
    
    return response

def parse_json_output(text):
    """Extract JSON from model output"""
    json_match = re.search(r'\{.*\}', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except:
            return None
    return None

# Generate predictions
predictions = []

for item in tqdm(val_data, desc="Predicting"):
    # print(item)
    try:
    # Generate
        response = predict_abcd(
            item['instruction'],
            item['input'],
            model,
            tokenizer
        )
        print(response)
        # Parse
        prediction = parse_json_output(response)
        ground_truth = parse_json_output(item['output'])
        
        predictions.append({
            'timeline_id':item['timeline_id'],
            'post_id': item['post_id'],
            'prediction': prediction,
            'ground_truth': ground_truth,
            'raw_response': response
        })
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        predictions.append({
            'prediction': None,
            'ground_truth': None,
            'error': str(e)
        })

print(f"\n✅ Generated {len(predictions)} predictions\n")

# Save predictions
with open('val_predictions.json', 'w') as f:
    json.dump(predictions, f, indent=2)
print("✅ Predictions saved to val_predictions.json\n")

# ========== 4. Convert to Binary Matrices ==========
print("="*60)
print("Converting to binary format...")
print("="*60)

# Define all dimensions
dimensions = ['A', 'B-S', 'B-O', 'C-S', 'C-O', 'D']
categories = []
for dim in dimensions:
    categories.append(f"{dim}_adaptive")
    categories.append(f"{dim}_maladaptive")

n_categories = len(categories)

def evidence_to_binary(evidence_dict):
    """Convert evidence dict to binary vector"""
    vector = np.zeros(n_categories)
    
    if not evidence_dict:
        return vector
    
    # Adaptive state
    if 'adaptive-state' in evidence_dict:
        for dim, data in evidence_dict['adaptive-state'].items():
            if dim in dimensions and data:
                idx = categories.index(f"{dim}_adaptive")
                vector[idx] = 1
    
    # Maladaptive state
    if 'maladaptive-state' in evidence_dict:
        for dim, data in evidence_dict['maladaptive-state'].items():
            if dim in dimensions and data:
                idx = categories.index(f"{dim}_maladaptive")
                vector[idx] = 1
    
    return vector

# Convert all predictions and ground truths
y_true = []
y_pred = []

valid_count = 0
for pred in predictions:
    if pred['prediction'] and pred['ground_truth']:
        y_true.append(evidence_to_binary(pred['ground_truth']))
        y_pred.append(evidence_to_binary(pred['prediction']))
        valid_count += 1

y_true = np.array(y_true)
y_pred = np.array(y_pred)

print(f"Valid predictions: {valid_count}/{len(predictions)}")
print(f"Binary matrix shape: {y_true.shape}\n")

# ========== 5. Calculate Metrics ==========
print("="*60)
print("EVALUATION RESULTS")
print("="*60)

# Overall metrics (micro-averaged)
precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
    y_true, y_pred, average='micro', zero_division=0
)

print("\n--- Overall (Micro-Averaged) ---")
print(f"Precision: {precision_micro:.4f}")
print(f"Recall:    {recall_micro:.4f}")
print(f"F1 Score:  {f1_micro:.4f}")

# Macro-averaged (average across all dimensions)
precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
    y_true, y_pred, average='macro', zero_division=0
)

print("\n--- Macro-Averaged (All Dimensions) ---")
print(f"Precision: {precision_macro:.4f}")
print(f"Recall:    {recall_macro:.4f}")
print(f"F1 Score:  {f1_macro:.4f}")

# Per-dimension metrics
print("\n--- Per-Dimension Metrics ---")
precision_per_cat, recall_per_cat, f1_per_cat, support = precision_recall_fscore_support(
    y_true, y_pred, average=None, zero_division=0
)

for i, cat in enumerate(categories):
    if support[i] > 0:  # Only show categories that exist in ground truth
        print(f"\n{cat}:")
        print(f"  Precision: {precision_per_cat[i]:.4f}")
        print(f"  Recall:    {recall_per_cat[i]:.4f}")
        print(f"  F1:        {f1_per_cat[i]:.4f}")
        print(f"  Support:   {int(support[i])}")

# Exact match accuracy
from sklearn.metrics import accuracy_score
exact_match = accuracy_score(y_true, y_pred)
print(f"\n--- Exact Match Accuracy ---")
print(f"Accuracy: {exact_match:.4f}")

print("\n" + "="*60)

# ========== 6. Save Results ==========
results = {
    'micro': {
        'precision': float(precision_micro),
        'recall': float(recall_micro),
        'f1': float(f1_micro)
    },
    'macro': {
        'precision': float(precision_macro),
        'recall': float(recall_macro),
        'f1': float(f1_macro)
    },
    'per_dimension': {
        categories[i]: {
            'precision': float(precision_per_cat[i]),
            'recall': float(recall_per_cat[i]),
            'f1': float(f1_per_cat[i]),
            'support': int(support[i])
        }
        for i in range(len(categories))
    },
    'exact_match': float(exact_match)
}

with open('evaluation_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n✅ Results saved to evaluation_results.json")