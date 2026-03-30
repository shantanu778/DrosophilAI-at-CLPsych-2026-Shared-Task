import argparse
from xml.parsers.expat import model

from tqdm import tqdm
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





if __name__=='__main__':
    # login(token=hf_token)
    
    parser = argparse.ArgumentParser()
    # parser.add_argument("--model_name", help="Available models Qwen, llama, phi",
    #                 type=str)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=4e-4)
    args = parser.parse_args()
    EPOCHS = args.epochs
    LR = args.lr

    # if args.model_name =='qwen':
    #     model_name = "unsloth/Qwen3.5-9B-UD-Q4_K_XL"
    # elif args.model_name=='phi':
    #     model_name="unsloth/phi-4-bnb-4bit"
    # elif args.model_name=='llama':
    #     model_name="unsloth/llama-3-8b-Instruct-bnb-4bit"
    # elif args.model_name=='deepseek':
    #     model_name="unsloth/DeepSeek-R1-0528-Qwen3-8B-unsloth-bnb-4bit"
    # else:
    #     assert 'This model in not available yet. we choose llama intead'
    #     model_name="unsloth/llama-3-8b-Instruct-bnb-4bit"
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
    print(f"STEP 4: Loading model")
    print("=" * 60)

    

    # bnb_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_compute_dtype=torch.bfloat16,
    #     bnb_4bit_use_double_quant=True,
    # )

    model_l, tokenizer_l = FastLanguageModel.from_pretrained(
        model_name="unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
        device_map="balanced",
    )
    model_q, tokenizer_q = FastLanguageModel.from_pretrained(
        model_name="unsloth/Qwen3.5-0.8B",
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
        device_map="balanced",
    )
    model_p, tokenizer_p = FastLanguageModel.from_pretrained(
        model_name="unsloth/Phi-4-mini-instruct-bnb-4bit",      
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
    model_l.config.pretraining_tp = 1
    model_q.config.pretraining_tp = 1
    model_p.config.pretraining_tp = 1


    # ========== STEP 5: Setup LoRA ==========
    print("\n" + "=" * 60)
    print("STEP 5: Configuring LoRA")
    print("=" * 60)

   

    model_l = FastLanguageModel.get_peft_model(  # Unsloth's version
        model_l,
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
    model_q = FastLanguageModel.get_peft_model(  # Unsloth's version
        model_q,
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
    model_p = FastLanguageModel.get_peft_model(  # Unsloth's version
        model_p,
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
    model_l.print_trainable_parameters()
    model_q.print_trainable_parameters()
    model_p.print_trainable_parameters()

    # ========== STEP 6: Create Datasets ==========
    print("\n" + "=" * 60)
    print("STEP 6: Creating PyTorch Datasets")
    print("=" * 60)

    train_dataset = ABCDInstructionDataset(train_data, tokenizer_l, max_length=2048)
    val_dataset = ABCDInstructionDataset(val_data, tokenizer_l, max_length=2048)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=4)

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
    print(f"Model device: {next(model_l.parameters()).device}")
    # print(f"Expected: mps")


    # data_collator = DataCollatorForLanguageModeling(
    #     tokenizer=tokenizer,
    #     mlm=False
    # )
    # # add this import at top

    # ========== STEP 8: Train ==========
    print("\n" + "=" * 60)
    print("STEP 8: Starting Training")
    print("=" * 60)
    optimizer = torch.optim.AdamW(model_l.parameters(), lr=2e-4)
    total_loss = 0.0
    
    for epoch in range(EPOCHS):
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            input_ids = batch['input_ids'].to(model_l.device)
            attention_mask = batch['attention_mask'].to(model_l.device)
            labels = batch['labels'].to(model_l.device)
            
            output_llama = model_l(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            output_qwen = model_q(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            output_phi = model_p(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

            loss = output_llama.loss + output_qwen.loss + output_phi.loss // 3.0

            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_l.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
    
        train_loss = total_loss / len(train_loader)

        print(f"\n✅ Epoch {epoch} completed. Average Loss: {train_loss:.4f}")
    # ========== STEP 9: Save Model ==========
    print("\n" + "=" * 60)
    print("STEP 9: Saving Model")
    print("=" * 60)

    # trainer.save_model(f"{args.model_name}-final")
    # tokenizer.save_pretrained(f"{args.model_name}-final")

    # print(f"\n✅ Training complete! Model saved to {args.model_name}-final")