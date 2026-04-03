import json
from glob import glob
import os
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import LoraConfig, get_peft_model
import torch.nn.functional as F
import torch
import torch.nn as nn
from transformers import Trainer, TrainingArguments
from sklearn.metrics import f1_score


def load_task2_multilabel(json_files_dir):
    """Load JSON → multi-label DataFrame."""
    all_posts = []
    
    for json_file in glob(os.path.join(json_files_dir, "*.json")):
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        timeline_id = data["timeline_id"]
        for post in data["posts"]:
            all_posts.append({
                'timeline_id': timeline_id,
                'post_index': post['post_index'],
                'post_id': post['post_id'],
                'text': post['post'],
                'wellbeing': post.get('Well-being', 5),
                'switch': 1 if post.get('Switch') == 'S' else 0,      # Binary
                'escalation': 1 if post.get('Escalation') == 'E' else 0  # Binary
            })
    
    df = pd.DataFrame(all_posts).sort_values(['timeline_id', 'post_index'])
    
    # Multi-label: [switch, escalation]
    df['labels'] = list(zip(df['switch'], df['escalation']))
    
    print(f"Posts with BOTH: {(df['switch'] & df['escalation']).sum()}")
    return df

train_df = load_task2_multilabel('tasks12/train/')
val_df = load_task2_multilabel('tasks12/valid/')

print(f"Train shape: {train_df.head()}, Val shape: {val_df.head()}")



model_name = "mental/mental-roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Multi-label: 2 outputs (switch + escalation)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2,  # [switch, escalation]
    problem_type="multi_label_classification"  # Key change!
)

# LoRA
lora_config = LoraConfig(
    r=16, lora_alpha=32,
    target_modules=["query", "key", "value"],
    task_type="SEQ_CLS"
)
model = get_peft_model(model, lora_config)

def tokenize_multilabel_batched(examples):
    """✅ Batched tokenization + float labels"""
    tokenized = tokenizer(
        examples['text'], 
        truncation=True, 
        padding='max_length', 
        max_length=512,
        return_tensors='pt'  # Add this!
    )
    # ✅ Shape: (batch_size, 2) float32
    tokenized['labels'] = torch.tensor(examples['labels'], 
        dtype=torch.float32
    )
    return tokenized

# ✅ ONE LINE - batched processing
train_dataset = Dataset.from_pandas(train_df[['text', 'labels']]).map(
    tokenize_multilabel_batched, batched=True
)
val_dataset = Dataset.from_pandas(val_df[['text', 'labels']]).map(
    tokenize_multilabel_batched, batched=True
)
print(f"Tokenized Train: {train_dataset[0]}, Tokenized Val: {val_dataset[0]}")
def compute_multilabel_metrics(eval_pred):
    predictions, labels = eval_pred
    preds = (torch.sigmoid(torch.tensor(predictions)) > 0.5).numpy()
    labels = labels > 0.5  # Already torch.float → works!
    
    switch_f1 = f1_score(labels[:, 0], preds[:, 0])
    esc_f1 = f1_score(labels[:, 1], preds[:, 1])
    return {'switch_f1': switch_f1, 'escalation_f1': esc_f1, 'macro_f1': (switch_f1 + esc_f1)/2}

args = TrainingArguments(
    output_dir="./mentalroberta-multilabel-task2",
    num_train_epochs=10,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    warmup_steps=10,
    eval_strategy="steps",
    eval_steps=10,
    save_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    fp16=True,
    report_to="none",
    save_total_limit=2
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

trainer.train()


def predict_multilabel(model_path, texts):
    """Predict [switch_prob, escalation_prob] per post."""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    
    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
    
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.sigmoid(outputs.logits)  # [batch, 2]
    
    switch_preds = (probs[:, 0] > 0.5).long()
    esc_preds = (probs[:, 1] > 0.5).long()
    
    return {
        'switch': switch_preds.tolist(),
        'escalation': esc_preds.tolist(),
        'probs': probs.tolist()
    }

# Your example
texts = [
    "Thank you for suggestions on my hair! I love the outcome...",
    "Getting harder and harder... Feel worthless..."  # BOTH!
]

results = predict_multilabel("./mentalroberta-multilabel-task2", texts)
print("Post 1:", "0/0" if results['switch'][0] == 0 and results['escalation'][0] == 0 else "S/E?")
print("Post 2:", "S/E" if results['switch'][1] == 1 and results['escalation'][1] == 1 else "?")
