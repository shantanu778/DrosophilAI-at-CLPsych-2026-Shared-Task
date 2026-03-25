#!/usr/bin/env python3
"""
Evaluate Llama-3 on presence rating task
"""

import json
import re
import torch
import numpy as np
from mlx_tune import FastLanguageModel, SFTTrainer, SFTConfig
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr

print("="*60)
print("Llama-3 Presence Rating - Evaluation")
print("="*60)

# ========== Load Model ==========
print("\nLoading model...")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="llama3_presence_lora",
    max_seq_length=1024,
    dtype=None,
    load_in_4bit=True,
)

FastLanguageModel.for_inference(model)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
print("✅ Model loaded")

# ========== Load Validation Data ==========
print("\nLoading validation data...")

with open('val_presence_instructions.json', 'r') as f:
    val_data = json.load(f)

print(f"Validation examples: {len(val_data)}")

# ========== Prediction Function ==========
def predict_presence(post_text, instruction, model, tokenizer):
    """Generate presence rating prediction"""
    
    messages = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": f"Post: {post_text}"}
    ]
    
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.1,
        top_p=0.9,
        do_sample=True,
        use_cache=True,
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


# ========== Generate Predictions ==========
print("\nGenerating predictions...")

predictions = []
adaptive_preds = []
adaptive_true = []
maladaptive_preds = []
maladaptive_true = []

for item in tqdm(val_data):
    try:
        # Extract post from input
        post_text = item['input'].replace('Post: ', '')
        
        # Predict
        response = predict_presence(post_text, item['instruction'], model, tokenizer)
        prediction = parse_json_output(response)
        
        # Parse ground truth
        ground_truth = json.loads(item['output'])
        
        if prediction:
            adaptive_preds.append(prediction.get('adaptive_score', 0))
            maladaptive_preds.append(prediction.get('maladaptive_score', 0))
        else:
            adaptive_preds.append(0)
            maladaptive_preds.append(0)
        
        adaptive_true.append(ground_truth['adaptive_score'])
        maladaptive_true.append(ground_truth['maladaptive_score'])
        
        predictions.append({
            'post_id': item.get('post_id'),
            'prediction': prediction,
            'ground_truth': ground_truth,
            'raw_response': response
        })
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        adaptive_preds.append(0)
        maladaptive_preds.append(0)
        adaptive_true.append(0)
        maladaptive_true.append(0)

# ========== Calculate Metrics ==========
print("\n" + "="*60)
print("EVALUATION RESULTS")
print("="*60)

# Adaptive metrics
adaptive_mae = mean_absolute_error(adaptive_true, adaptive_preds)
adaptive_rmse = np.sqrt(mean_squared_error(adaptive_true, adaptive_preds))
adaptive_corr, _ = pearsonr(adaptive_true, adaptive_preds)
adaptive_acc_1 = np.mean(np.abs(np.array(adaptive_true) - np.array(adaptive_preds)) <= 1)

print("\nAdaptive Self-State:")
print(f"  MAE:      {adaptive_mae:.4f}")
print(f"  RMSE:     {adaptive_rmse:.4f}")
print(f"  Corr:     {adaptive_corr:.4f}")
print(f"  Acc(±1):  {adaptive_acc_1:.4f}")

# Maladaptive metrics
maladaptive_mae = mean_absolute_error(maladaptive_true, maladaptive_preds)
maladaptive_rmse = np.sqrt(mean_squared_error(maladaptive_true, maladaptive_preds))
maladaptive_corr, _ = pearsonr(maladaptive_true, maladaptive_preds)
maladaptive_acc_1 = np.mean(np.abs(np.array(maladaptive_true) - np.array(maladaptive_preds)) <= 1)

print("\nMaladaptive Self-State:")
print(f"  MAE:      {maladaptive_mae:.4f}")
print(f"  RMSE:     {maladaptive_rmse:.4f}")
print(f"  Corr:     {maladaptive_corr:.4f}")
print(f"  Acc(±1):  {maladaptive_acc_1:.4f}")

# Overall
avg_mae = (adaptive_mae + maladaptive_mae) / 2
print(f"\nAverage MAE: {avg_mae:.4f}")

# ========== Save Results ==========
results = {
    'adaptive': {
        'mae': float(adaptive_mae),
        'rmse': float(adaptive_rmse),
        'correlation': float(adaptive_corr),
        'accuracy_within_1': float(adaptive_acc_1)
    },
    'maladaptive': {
        'mae': float(maladaptive_mae),
        'rmse': float(maladaptive_rmse),
        'correlation': float(maladaptive_corr),
        'accuracy_within_1': float(maladaptive_acc_1)
    },
    'average_mae': float(avg_mae)
}

with open('llama3_presence_results.json', 'w') as f:
    json.dump(results, f, indent=2)

# Save predictions
with open('llama3_presence_predictions.json', 'w') as f:
    json.dump(predictions, f, indent=2)

print("\n✅ Results saved!")
print("  - llama3_presence_results.json")
print("  - llama3_presence_predictions.json")