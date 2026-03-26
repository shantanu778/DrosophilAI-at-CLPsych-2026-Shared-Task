from unsloth import FastLanguageModel 
from trl import SFTTrainer
import torch
import json
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)

from dataset import (CLPsychDataLoader, create_instruction_dataset, df_to_training_format,
                      ABCDInstructionDataset)


def get_device():
    """Detect best available device on Mac"""
    if torch.cuda.is_available():
        device = "cuda"
        print("✅ Using CUDA GPU")
    elif torch.backends.mps.is_available():
        device = "mps"
        print("✅ Using Apple Silicon MPS")
    else:
        device = "cpu"
        print("⚠️  Using CPU (this will be slow)")
    
    return device
device = get_device()

# login(token=hf_token)

# ========== STEP 1: Load Data ==========
print("=" * 60)
print("STEP 1: Loading Data")
print("=" * 60)

train_loader = CLPsychDataLoader('tasks12/', split='train')
val_loader = CLPsychDataLoader('tasks12/', split='val')
train_df = train_loader.load()
print("Training Set Stats")
val_df = val_loader.load()
train_loader.verify_order()
train_loader.get_stats()
print("\n" + "=" * 60)
print("Validation Set Stats")
val_loader.verify_order()
val_loader.get_stats()
# ========== STEP 2: Create Instruction Dataset ==========
print("\n" + "=" * 60)
print("STEP 2: Creating Instruction Dataset")
print("=" * 60)

train_df = create_instruction_dataset(train_df)
val_df = create_instruction_dataset(val_df)

# Create instruction dataset
# instruction_df = create_instruction_dataset(df)

# Verify first few examples
print("\n=== First 3 Examples ===")
for idx in range(min(3, len(train_df))):
    row = train_df.iloc[idx]
    print(f"\nExample {idx+1}:")
    print(f"Timeline: {row['timeline_id']}, Post: {row['post_index']}")
    print(f"Input: {row['input'][:100]}...")
    print(f"Output: {row['output'][:-100]}...")

# ========== STEP 3: Train/Val Split ==========
# print("\n" + "=" * 60)
# print("STEP 3: Train/Val Split")
# print("=" * 60)

# train_df, val_df = timeline_aware_split(instruction_df, test_size=0.15)

# Convert to list format
train_data = df_to_training_format(train_df)
val_data = df_to_training_format(val_df)
# Save for inspection
print("\n=== Saving sample data ===")
with open('train_sample.json', 'w') as f:
    json.dump(train_data[:5], f, indent=2)
print("Saved first 5 training examples to train_sample.json")

# ========== STEP 4: Load Model ==========
print("\n" + "=" * 60)
print("STEP 4: Loading Meta-Llama-3.1-8B")
print("=" * 60)

# model_name = "unsloth/Llama-3.3-70B-Instruct" 
model_name = "unsloth/Meta-Llama-3.1-8B-Instruct"

# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.bfloat16,
#     bnb_4bit_use_double_quant=True,
# )

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
    device_map="balanced",
)

# Load tokenizer
# tokenizer.pad_token = tokenizer.eos_token
# tokenizer.padding_side = "right"

# Load model WITHOUT BitsAndBytesConfig
# model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B", load_in_4bit=True)
# model = model.to(device)
# model.config.use_cache = False
model.config.pretraining_tp = 1

# ========== STEP 5: Setup LoRA ==========
print("\n" + "=" * 60)
print("STEP 5: Configuring LoRA")
print("=" * 60)

# model = prepare_model_for_kbit_training(model)
# LoRA configuration
# lora_config = LoraConfig(
#     r=8,  # Rank
#     lora_alpha=16,
#     target_modules=[
#         "q_proj",
#         "k_proj",
#         "v_proj",
#         "o_proj",
#         "gate_proj",
#         "up_proj",
#         "down_proj",
#     ],
#     lora_dropout=0.05,
#     bias="none",
#     task_type="CAUSAL_LM"
# )

model = FastLanguageModel.get_peft_model(  # Unsloth's version
    model,
    r=64,  # higher rank for 70B
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=128,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",  # VRAM saver
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)


# model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ========== STEP 6: Create Datasets ==========
print("\n" + "=" * 60)
print("STEP 6: Creating PyTorch Datasets")
print("=" * 60)

train_dataset = ABCDInstructionDataset(train_data, tokenizer, max_length=2048)
val_dataset = ABCDInstructionDataset(val_data, tokenizer, max_length=2048)

print(f"Train dataset: {len(train_dataset)} examples")
print(f"Val dataset: {len(val_dataset)} examples")

# ========== STEP 7: Training Configuration ==========
print("\n" + "=" * 60)
print("STEP 7: Configuring Training")
print("=" * 60)

# Add this right before training
print("\n" + "="*60)
print("FINAL DEVICE CHECK")
print("="*60)
print(f"Model device: {next(model.parameters()).device}")
print(f"Expected: mps")

# Quick speed test
# import time
# device = next(model.parameters()).device

# x = torch.randn(1000, 1000).to(device)
# start = time.time()
# for _ in range(100):
#     y = x @ x
# if device.type == "mps":
#     torch.mps.synchronize()
# elapsed = time.time() - start
# print(f"GPU matrix multiply (100 iterations): {elapsed:.2f}s")
# print("If > 5s, something is wrong")
# print("="*60 + "\n")

# training_args = TrainingArguments(
#     output_dir="./llama3-abcd-lora",
#     num_train_epochs=10,
#     per_device_train_batch_size=4,
#     per_device_eval_batch_size=4,
#     gradient_accumulation_steps=4,
#     learning_rate=5e-4,
#     lr_scheduler_type="cosine",
#     warmup_steps=0.03,  # 3% of total steps
#     logging_steps=10,
#     save_strategy="epoch",
#     eval_strategy="epoch",
#     bf16=True,
#     optim="paged_adamw_8bit",
#     max_grad_norm=0.3,
#     train_sampling_strategy="sequential",  # Ensure sequential sampling to preserve order
#     report_to="none",
#     save_total_limit=2,
# )

# data_collator = DataCollatorForLanguageModeling(
#     tokenizer=tokenizer,
#     mlm=False
# )
 # add this import at top

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    dataset_text_field="text",  # we'll add this
    max_seq_length=2048,
    dataset_num_proc=2,
    packing=False,  # preserves timeline order
    args=TrainingArguments(
        per_device_train_batch_size=2,  # small for 70B
        gradient_accumulation_steps=8, # effective batch=32
        per_device_eval_batch_size=2,
        output_dir="./llama3_8b_abcd",
        num_train_epochs=10,  # shorter first run
        load_best_model_at_end=True,     # Load best at end
        metric_for_best_model="eval_loss",
        greater_is_better=False,         # Lower loss = better  # Save after each eval
        learning_rate=2e-4,
        warmup_steps=10,
        logging_steps=5,
        save_strategy="steps",
        save_steps=20,
        eval_steps=20,
        eval_strategy="steps",
        bf16=True,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        max_grad_norm=1.0,
        report_to="none",
        ddp_find_unused_parameters=False,  # DDP essential
    ),
)

# ========== STEP 8: Train ==========
print("\n" + "=" * 60)
print("STEP 8: Starting Training")
print("=" * 60)

trainer.train()

# ========== STEP 9: Save Model ==========
print("\n" + "=" * 60)
print("STEP 9: Saving Model")
print("=" * 60)

trainer.save_model("./llama3_8B-abcd-lora-final")
tokenizer.save_pretrained("./llama3_8B-abcd-lora-final")

print("\n✅ Training complete! Model saved to ./llama3_70B-abcd-lora-final")