import argparse
from unsloth import FastLanguageModel 
from trl import SFTTrainer
from datasets import Dataset
import torch
import json
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)

from dataset import (CLPsychDataLoader, create_instruction_dataset, df_to_training_format,
                      ABCDInstructionDataset)


'''
Python task1.1/train.py --model_name phi/qwen/llama
'''

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




if __name__=='__main__':
    # login(token=hf_token)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", help="Available models Qwen, llama, phi",
                    type=str)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=4e-4)
    args = parser.parse_args()
    EPOCHS = args.epochs
    LR = args.lr

    if args.model_name =='qwen':
        model_name = "unsloth/Qwen3.5-9B-GGUF"
    elif args.model_name=='phi':
        model_name="unsloth/phi-4-bnb-4bit"
    elif args.model_name=='llama':
        model_name="unsloth/llama-3-8b-Instruct-bnb-4bit"
    elif args.model_name=='deepseek':
        model_name="unsloth/DeepSeek-R1-0528-Qwen3-8B-unsloth-bnb-4bit"
    else:
        assert 'This model in not available yet. we choose llama intead'
        model_name="unsloth/llama-3-8b-Instruct-bnb-4bit"
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

    # ========== Format Dataset ==========
    print("\nFormatting dataset...")

    # Convert to HF Dataset
    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)

    train_dataset = train_dataset.map(formatting_prompts_func, batched=True)
    val_dataset = val_dataset.map(formatting_prompts_func, batched=True)

    print(f"✅ Datasets formatted")
    # model_name = "unsloth/Llama-3.3-70B-Instruct" 
    model_name = model_name

    # ========== STEP 4: Load Model ==========
    print("\n" + "=" * 60)
    print(f"STEP 4: Loading {model_name} model")
    print("=" * 60)

    

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
        r=8,  # higher rank for 70B
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",  # VRAM saver
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )


    # model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # # ========== STEP 6: Create Datasets ==========
    # print("\n" + "=" * 60)
    # print("STEP 6: Creating PyTorch Datasets")
    # print("=" * 60)

    # train_dataset = ABCDInstructionDataset(train_data, tokenizer, max_length=2048)
    # val_dataset = ABCDInstructionDataset(val_data, tokenizer, max_length=2048)

    # print(f"Train dataset: {len(train_dataset)} examples")
    # print(f"Val dataset: {len(val_dataset)} examples")

    # # ========== STEP 7: Training Configuration ==========
    # print("\n" + "=" * 60)
    # print("STEP 7: Configuring Training")
    # print("=" * 60)

    # # Add this right before training
    # print("\n" + "="*60)
    # print("FINAL DEVICE CHECK")
    # print("="*60)
    # print(f"Model device: {next(model.parameters()).device}")
    # print(f"Expected: mps")

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
    # # add this import at top

    # trainer = Trainer(
    #     model=model,
    #     train_dataset=train_dataset,
    #     eval_dataset=val_dataset,
    #     args=TrainingArguments(
    #         per_device_train_batch_size=2,  # small for 70B
    #         gradient_accumulation_steps=8, # effective batch=32
    #         per_device_eval_batch_size=2,
    #         output_dir="./llama3_8b_abcd",
    #         num_train_epochs=5,  # shorter first run
    #         load_best_model_at_end=True,     # Load best at end
    #         metric_for_best_model="eval_loss",
    #         greater_is_better=False,         # Lower loss = better  # Save after each eval
    #         learning_rate=5e-4,
    #         warmup_steps=10,
    #         logging_steps=5,
    #         save_strategy="steps",
    #         save_steps=10,
    #         eval_steps=5,
    #         eval_strategy="steps",
    #         save_total_limit=2,
    #         bf16=True,
    #         optim="adamw_8bit",
    #         weight_decay=0.01,
    #         lr_scheduler_type="cosine",
    #         max_grad_norm=1.0,
    #         report_to="none",
    #         train_sampling_strategy="sequential",
    #     ),
    # )

    training_args = TrainingArguments(
        output_dir=args.model_name,
        
        # Training schedule
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,
        
        # Optimization
        learning_rate=LR,
        lr_scheduler_type="cosine",
        warmup_ratio=0.02,
        optim="adamw_8bit",
        weight_decay=0.01,
        
        # Logging & Evaluation
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,     # Load best at end
        metric_for_best_model="eval_loss",

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
        max_seq_length=2048,
        dataset_num_proc=2,
        packing=False,
        args=training_args,
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

    trainer.save_model(f"{args.model_name}-final")
    tokenizer.save_pretrained(f"{args.model_name}-final")

    print(f"\n✅ Training complete! Model saved to {args.model_name}-final")