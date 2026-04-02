#!/usr/bin/env python3
"""
Task 2 — Approach 3: Fine-tune Phi-4 with full timeline context
to detect Switch / Escalation / Neither per post.

Usage:
    python train_task2_llm.py --model_name phi --epochs 10 --data_dir data/
"""

import argparse
from unsloth import FastLanguageModel
from trl import SFTTrainer
from datasets import Dataset
import torch
import json
import numpy as np
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from torch.utils.data import Dataset as TorchDataset
from sklearn.utils.class_weight import compute_class_weight

from dataset_task2_llm import (
    CLPsychTask2DataLoader, create_task2_dataset, df_to_training_format, LABEL_NAMES
)


def get_device():
    if torch.cuda.is_available():
        print(f"✅ Using CUDA GPU: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        return "cuda"
    print("⚠️  Using CPU")
    return "cpu"


def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for instruction, input_text, output in zip(instructions, inputs, outputs):
        text = (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"{instruction}<|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|>\n\n"
            f"{input_text}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
            f"{output}<|eot_id|>"
        )
        texts.append(text)
    return {"text": texts}


class TextDataset(TorchDataset):
    def __init__(self, hf_dataset, tokenizer, max_length=1024):
        self.data = []
        for text in hf_dataset["text"]:
            enc = tokenizer(
                text,
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt"
            )
            self.data.append({
                "input_ids":      enc["input_ids"].squeeze(),
                "attention_mask": enc["attention_mask"].squeeze(),
                "labels":         enc["input_ids"].squeeze(),
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


if __name__ == '__main__':

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device = get_device()

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True,
                        help="phi | qwen | llama")
    parser.add_argument("--epochs",    type=int,   default=5)
    parser.add_argument("--lr",        type=float, default=4e-4)
    parser.add_argument("--data_dir",  type=str,   default="data/")
    parser.add_argument("--max_history", type=int, default=1,
                        help="Max previous posts to include as context. None=all.")
    args = parser.parse_args()

    if args.model_name == 'phi':
        model_id   = "unsloth/phi-4-unsloth-bnb-4bit"
        output_tag = "phi4_task2"
    elif args.model_name == 'qwen':
        model_id   = "unsloth/Qwen2.5-14B-Instruct-bnb-4bit"
        output_tag = "qwen14B_task2"
    elif args.model_name == 'llama':
        model_id   = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
        output_tag = "llama8B_task2"
    else:
        model_id   = "unsloth/phi-4-unsloth-bnb-4bit"
        output_tag = "phi4_task2"

    # STEP 1: Load Data
    print("=" * 60)
    print("STEP 1: Loading Data")
    print("=" * 60)

    train_loader = CLPsychTask2DataLoader(args.data_dir, split='train')
    val_loader   = CLPsychTask2DataLoader(args.data_dir, split='val')
    train_df     = train_loader.load()
    val_df       = val_loader.load()
    train_loader.get_stats()
    val_loader.get_stats()

    # STEP 2: Create instruction dataset with timeline context
    print("\n" + "=" * 60)
    print("STEP 2: Creating Instruction Dataset with Timeline Context")
    print("=" * 60)

    train_inst_df = create_task2_dataset(train_df, max_history=args.max_history)
    val_inst_df   = create_task2_dataset(val_df,   max_history=args.max_history)

    train_data = df_to_training_format(train_inst_df)
    val_data   = df_to_training_format(val_inst_df)

    print("\n=== First 2 Examples ===")
    for i in range(min(2, len(train_data))):
        d = train_data[i]
        print(f"\nExample {i+1}: [{d['label_str']}]")
        print(f"Input (first 200 chars): {d['input'][:200]}...")
        print(f"Output: {d['output']}")

    with open('task2_train_sample.json', 'w') as f:
        json.dump(train_data[:3], f, indent=2)
    print("\nSaved sample to task2_train_sample.json")

    # Format as HuggingFace Dataset
    print("\nFormatting dataset...")
    train_hf = Dataset.from_list(train_data)
    val_hf   = Dataset.from_list(val_data)
    train_dataset = train_hf.map(formatting_prompts_func, batched=True)
    val_dataset   = val_hf.map(formatting_prompts_func, batched=True)
    print("✅ Datasets formatted")

    # STEP 4: Load Model
    print("\n" + "=" * 60)
    print(f"STEP 4: Loading {model_id}")
    print("=" * 60)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_id,
        max_seq_length=2048,   # longer — need space for timeline context
        dtype=torch.bfloat16,
        load_in_4bit=True,
    )
    model.config.pretraining_tp = 1

    # STEP 5: Configure LoRA
    print("\n" + "=" * 60)
    print("STEP 5: Configuring LoRA")
    print("=" * 60)

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=32,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )
    model.print_trainable_parameters()

    # STEP 6/7: Training Configuration
    print("\n" + "=" * 60)
    print("STEP 6/7: Configuring Training")
    print("=" * 60)

    training_args = TrainingArguments(
        output_dir=output_tag,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=2,   # smaller — inputs are longer
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,  # effective batch = 16
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        optim="adamw_8bit",
        weight_decay=0.01,
        max_grad_norm=0.3,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=True,
        tf32=True,
        dataloader_num_workers=0,
        dataloader_pin_memory=True,
        train_sampling_strategy="sequential",
        seed=42,
        report_to="none",
    )

    # Tokenize
    # print("\nTokenizing datasets...")
    # train_torch = TextDataset(train_hf, tokenizer, max_length=2048)
    # val_torch   = TextDataset(val_hf,   tokenizer, max_length=2048)
    # print(f"Train: {len(train_torch)} | Val: {len(val_torch)}")

    # STEP 8: Train
    print("\n" + "=" * 60)
    print("STEP 8: Starting Training")
    print("=" * 60)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        dataset_text_field="text",
        max_seq_length=2048,
        dataset_num_proc=2,
        packing=False,
        args=training_args,
    )
    trainer.train()

    # STEP 9: Save
    print("\n" + "=" * 60)
    print("STEP 9: Saving Model")
    print("=" * 60)

    save_path = f"./{output_tag}-final"
    trainer.save_model(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"\n✅ Training complete! Model saved to {save_path}")