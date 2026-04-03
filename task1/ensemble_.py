import json
import numpy as np
from collections import Counter
import re
from sklearn.metrics import precision_recall_fscore_support, classification_report

# Load all predictions
with open('test_predictions_phi-final.json') as f:
    phi_preds = json.load(f)
    
with open('test_predictions_qwen-final.json') as f:
    qwen_preds = json.load(f)
    
with open('test_predictions_llama-final.json') as f:
    llama_preds = json.load(f)

def all_predictions(phi_pred, qwen_pred, llama_pred, state, dim):
    """Get all predictions for a given state and dimension"""
    categories = []
    
    if phi_pred and state in phi_pred and dim in phi_pred[state]:
        categories.append(phi_pred[state][dim]['subelement'])
    
    if qwen_pred and state in qwen_pred and dim in qwen_pred[state]:
        categories.append(qwen_pred[state][dim]['subelement'])
    
    if llama_pred and state in llama_pred and dim in llama_pred[state]:
        categories.append(llama_pred[state][dim]['subelement'])

    # print(f"All predictions for {state} {dim}: {categories}")
    set_categories = set(categories)
    if len(set_categories) > 1:
        if phi_pred and state in phi_pred and dim in phi_pred[state]:
            return phi_pred[state][dim]['subelement']
        elif qwen_pred and state in qwen_pred and dim in qwen_pred[state]:
            return qwen_pred[state][dim]['subelement']
        elif llama_pred and state in llama_pred and dim in llama_pred[state]:
            return llama_pred[state][dim]['subelement']

    return categories[0] if set_categories else 'unknown'

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


def get_best_category(phi_pred, qwen_pred, llama_pred, state, dim):
    """Get category from the best model or most common"""
    categories = []
    
    if phi_pred and state in phi_pred and dim in phi_pred[state]:
        categories.append(phi_pred[state][dim]['subelement'])
    
    if qwen_pred and state in qwen_pred and dim in qwen_pred[state]:
        categories.append(qwen_pred[state][dim]['subelement'])
    
    if llama_pred and state in llama_pred and dim in llama_pred[state]:
        categories.append(llama_pred[state][dim]['subelement'])
    
    if categories:
        # Return most common category
        return Counter(categories).most_common(1)[0][0]

# ========== Task 1.1: ABCD Classification (Majority Voting) ==========
def majority_vote_abcd(phi_pred, qwen_pred, llama_pred):
    """Majority voting for ABCD elements"""

    
    
    ensemble_pred = {
        'adaptive-state': {},
        'maladaptive-state': {}
    }
    
    dimensions = ['A', 'B-S', 'B-O', 'C-S', 'C-O', 'D']
    
    for state in ['adaptive-state', 'maladaptive-state']:
        for dim in dimensions:
            # Count votes
            # votes = []
            
            # if phi_pred and state in phi_pred and dim in phi_pred[state]:
            #     votes.append('present')
            # else:
            #     votes.append('absent')
            
            # if qwen_pred and state in qwen_pred and dim in qwen_pred[state]:
            #     votes.append('present')
            # else:
            #     votes.append('absent')
            
            # if llama_pred and state in llama_pred and dim in llama_pred[state]:
            #     votes.append('present')
            # else:
            #     votes.append('absent')
            
            # # Majority wins (at least 2 out of 3)
            # vote_count = Counter(votes)
            # if vote_count['present'] >= 2:
            # print(f"{state} {dim} classified as present by majority vote.")
                # Take category from best model or most common
            ensemble_pred[state][dim] = {
                'subelement': all_predictions(phi_pred, qwen_pred, llama_pred, state, dim)
            }
            # else:
            #     ensemble_pred[state][dim] = {
            #         'subelement': all_predictions(phi_pred, qwen_pred, llama_pred, state, dim)
            #     }
                # print(f"Warning: {state} {dim} classified as absent by majority vote.")
    
    return ensemble_pred

# ========== Task 1.2: Presence Rating (Averaging) ==========
def average_presence_scores(phi_preds, qwen_preds, llama_preds):
    """Average presence scores from 3 models"""
    if not phi_preds:
        # print("Warning: Missing adaptive presence scores in phi model.")
        phi_preds = {'adaptive-state': {'Presence': 0}, 'maladaptive-state': {'Presence': 0}}
    if not qwen_preds:
        # print("Warning: Missing adaptive presence scores in qwen model.")
        qwen_preds = {'adaptive-state': {'Presence': 0}, 'maladaptive-state': {'Presence': 0}}
    if not llama_preds:
        # print("Warning: Missing adaptive presence scores in llama model.")
        llama_preds = {'adaptive-state': {'Presence': 0}, 'maladaptive-state': {'Presence': 0}}
    

    # Simple average

    avg_adaptive = (phi_preds['adaptive-state']['Presence'] + qwen_preds['adaptive-state']['Presence'] + llama_preds['adaptive-state']['Presence']) / 3
    avg_maladaptive = (phi_preds['maladaptive-state']['Presence'] + qwen_preds['maladaptive-state']['Presence'] + llama_preds['maladaptive-state']['Presence']) / 3
    
    # Round to nearest integer (1-5)
    avg_adaptive = int(round(avg_adaptive))
    avg_maladaptive = int(round(avg_maladaptive))
    return {
        'adaptive-state': avg_adaptive,
        'maladaptive-state': avg_maladaptive
    }


# Apply ensemble
dimensions = ['A', 'B-S', 'B-O', 'C-S', 'C-O', 'D']
ensemble_predictions = []
for i in range(len(phi_preds)):
    # print(f"Processing sample {i}, {phi_preds[i]}, {qwen_preds[i]}, {llama_preds[i]}...")
    # print(phi_preds[i]['timeline_id'], phi_preds[i]['post_id'], "Processing sample", i)
    prediction = majority_vote_abcd(
            phi_preds[i]['prediction'],
            qwen_preds[i]['prediction'],
            llama_preds[i]['prediction']
        )
    presence_scores = average_presence_scores(
            phi_preds[i]['prediction'],
            qwen_preds[i]['prediction'],
            llama_preds[i]['prediction']
        )
    
    prediction['adaptive-state']['Presence'] = presence_scores['adaptive-state']
    prediction['maladaptive-state']['Presence'] = presence_scores['maladaptive-state']

    for dim in dimensions:
        if prediction['adaptive-state'][dim]['subelement'] == 'unknown':
            prediction['adaptive-state'].pop(dim)
        if prediction['maladaptive-state'][dim]['subelement'] == 'unknown':
            prediction['maladaptive-state'].pop(dim)

    
    d = {
        'timeline_id': phi_preds[i]['timeline_id'],
        'post_id': phi_preds[i]['post_id'],
        'prediction': prediction
    }
    ensemble_predictions.append(d)

# Save ensemble predictions
with open('test_predictions_ensemble.json', 'w') as f:
    json.dump(ensemble_predictions, f, indent=2)


# # Define all dimensions
dimensions = ['A', 'B-S', 'B-O', 'C-S', 'C-O', 'D']
categories = []
for dim in dimensions:
    categories.append(f"{dim}_adaptive")
    categories.append(f"{dim}_maladaptive")

n_categories = len(categories)



# # Convert all predictions and ground truths
# y_true = []
# y_pred = []

# valid_count = 0
# for idx, pred in enumerate(ensemble_predictions):
#     if pred['prediction'] and phi_preds[idx]['ground_truth']:
#         y_true.append(evidence_to_binary(phi_preds[idx]['ground_truth']))
#         y_pred.append(evidence_to_binary(pred['prediction']))
#         valid_count += 1

# y_true = np.array(y_true)
# y_pred = np.array(y_pred)

# print(f"Valid predictions: {valid_count}/{len(ensemble_predictions)}")
# print(f"Binary matrix shape: {y_true.shape}\n")

# # ========== 5. Calculate Metrics ==========
# print("="*60)
# print("EVALUATION RESULTS")
# print("="*60)

# # Overall metrics (micro-averaged)
# precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
#     y_true, y_pred, average='micro', zero_division=0
# )

# print("\n--- Overall (Micro-Averaged) ---")
# print(f"Precision: {precision_micro:.4f}")
# print(f"Recall:    {recall_micro:.4f}")
# print(f"F1 Score:  {f1_micro:.4f}")

# # Macro-averaged (average across all dimensions)
# precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
#     y_true, y_pred, average='macro', zero_division=0
# )

# print("\n--- Macro-Averaged (All Dimensions) ---")
# print(f"Precision: {precision_macro:.4f}")
# print(f"Recall:    {recall_macro:.4f}")
# print(f"F1 Score:  {f1_macro:.4f}")

# # Per-dimension metrics
# print("\n--- Per-Dimension Metrics ---")
# precision_per_cat, recall_per_cat, f1_per_cat, support = precision_recall_fscore_support(
#     y_true, y_pred, average=None, zero_division=0
# )

# for i, cat in enumerate(categories):
#     if support[i] > 0:  # Only show categories that exist in ground truth
#         print(f"\n{cat}:")
#         print(f"  Precision: {precision_per_cat[i]:.4f}")
#         print(f"  Recall:    {recall_per_cat[i]:.4f}")
#         print(f"  F1:        {f1_per_cat[i]:.4f}")
#         print(f"  Support:   {int(support[i])}")

# # Exact match accuracy
# from sklearn.metrics import accuracy_score
# exact_match = accuracy_score(y_true, y_pred)
# print(f"\n--- Exact Match Accuracy ---")
# print(f"Accuracy: {exact_match:.4f}")

# print("\n" + "="*60)

# # ========== 6. Save Results ==========
# results = {
#     'micro': {
#         'precision': float(precision_micro),
#         'recall': float(recall_micro),
#         'f1': float(f1_micro)
#     },
#     'macro': {
#         'precision': float(precision_macro),
#         'recall': float(recall_macro),
#         'f1': float(f1_macro)
#     },
#     'per_dimension': {
#         categories[i]: {
#             'precision': float(precision_per_cat[i]),
#             'recall': float(recall_per_cat[i]),
#             'f1': float(f1_per_cat[i]),
#             'support': int(support[i])
#         }
#         for i in range(len(categories))
#     },
#     'exact_match': float(exact_match)
# }

# with open(f'evaluation_results_{MODEL}.json', 'w') as f:
#     json.dump(results, f, indent=2)

# print("\n✅ Results saved to evaluation_results.json")