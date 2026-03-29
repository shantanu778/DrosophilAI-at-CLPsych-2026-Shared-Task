#!/usr/bin/env python3
"""
Train Llama-3-8B for Task 1.2 (Presence Rating)
Using instruction tuning with Unsloth
"""

import warnings
warnings.filterwarnings('ignore')
import json
import torch
from mlx_tune import FastLanguageModel, SFTTrainer, SFTConfig
from datasets import Dataset
# from trl import SFTTrainer
from transformers import TrainingArguments
from dataset_v1 import prepare_instruction_dataset

# ========== Prepare Both Splits ==========
if __name__ == "__main__":
    train_data = prepare_instruction_dataset('tasks12/', split='train')
    val_data = prepare_instruction_dataset('tasks12/', split='val')
    
    # Show example
    print(f"\n{'='*60}")
    print("Example Instruction")
    print('='*60)
    print(f"\nInstruction: {train_data[0]['instruction'][:200]}...")
    print(f"\nInput: {train_data[0]['input'][:100]}...")
    print(f"\nOutput: {train_data[0]['output'][:300]}...")




    print("="*60)
    print("Llama-3-8B Instruction Tuning - Task 1.2")
    print("="*60)

    # ========== Load Instruction Data ==========
    print("\nLoading instruction data...")

    with open('train_presence_instructions.json', 'r') as f:
        train_data = json.load(f)

    with open('val_presence_instructions.json', 'r') as f:
        val_data = json.load(f)

    print(f"Train: {len(train_data)} examples")
    print(f"Val: {len(val_data)} examples")

    # ========== Load Model ==========
    print("\nLoading Llama-3-8B...")

    max_seq_length = 2048
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="mlx-community/Llama-3.2-3B-Instruct-4bit",
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )

    print("✅ Model loaded")

    # ========== Add LoRA ==========
    print("\nConfiguring LoRA...")

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    print("✅ LoRA configured")
    # model.print_trainable_parameters()

    # ========== Format Dataset ==========
    print("\nFormatting dataset...")

    def formatting_prompts_func(examples):
        """Format examples with Llama-3 chat template"""
        instructions = examples["instruction"]
        inputs = examples["input"]
        outputs = examples["output"]
        
        texts = []
        for instruction, input_text, output in zip(instructions, inputs, outputs):
            # Llama-3 chat format
            text = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

            {instruction}<|eot_id|><|start_header_id|>user<|end_header_id|>

            {input_text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

            {output}<|eot_id|>"""
            texts.append(text)
        
        return {"text": texts}

    # Convert to HF Dataset
    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)

    train_dataset = train_dataset.map(formatting_prompts_func, batched=True)
    val_dataset = val_dataset.map(formatting_prompts_func, batched=True)

    print(f"✅ Datasets formatted")

    # ========== Training Configuration ==========
    print("\nConfiguring training...")

    training_args = TrainingArguments(
        output_dir="llama3_presence_rating_v2",
        
        # Training schedule
        num_train_epochs=1,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        
        # Optimization
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        optim="adamw_8bit",
        weight_decay=0.01,
        
        # Logging & Evaluation
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        
        # Performance
        fp16=not torch.cuda.is_available(),
        bf16=torch.cuda.is_available(),
        
        # Other
        seed=42,
        report_to="none",
    )

    # ========== Initialize Trainer ==========
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=2,
        packing=False,
        args=training_args,
    )

    # ========== Train ==========
    print("\n" + "="*60)
    print("Starting Training")
    print("="*60 + "\n")

    try:
        trainer_stats = trainer.train()
        
        print("\n" + "="*60)
        print("Training Complete!")
        print("="*60)
        print(trainer_stats)
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

    # ========== Save Model ==========
    print("\nSaving model...")
    model.save_pretrained("llama3_presence_lora")
    tokenizer.save_pretrained("llama3_presence_lora")

    print("\n✅ Model saved to ./llama3_presence_lora")