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
from dataset import get_taxonomy_string
import pandas as pd
import os
from glob import glob


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
            use_cache=True,
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

def df_to_training_format(df):
    """
    Convert DataFrame to list of dicts for training
    Order is already preserved in DataFrame
    """
    training_data = []
    
    for idx, row in df.iterrows():
        training_data.append({
            'timeline_id': row['timeline_id'],
            'post_id': row['post_id'],
            'instruction': row['instruction'],
            'input': row['input']
        })
    
    return training_data

def create_instruction_dataset(df):
    """
    Convert DataFrame to instruction-tuning format
    Maintains order from sorted DataFrame
    """
    
    instruction = """Analyze the social media post using the MIND framework. Identify ABCD self-state elements and output ONLY a JSON object.

                Dimensions: A (Affect), B-S (Behavior-Self), B-O (Behavior-Others), C-S (Cognition-Self), C-O (Cognition-Others), D (Desire).
                Rules:
                Each dimension may appear in adaptive-state, maladaptive-state, both, or neither.
                Include only dimensions detected. Evidence must be an exact quote (3-15 words).
                Presence score is an integer 1-5 based on intensity. Do not output NULL or None.

                Subelements:
                """ + get_taxonomy_string() + """

                Output ONLY valid JSON, no explanation: For example, if a post has evidence of contentment (A: Content, happy, joy, hopeful) and self care (B-S: Self care and improvement) in the adaptive state, and anxiety (A: Anxious/ fearful/ tense), self criticism (C-S: Self criticism), and expectation of unmet relatedness needs (D: Expectation that relatedness needs will not be met) in the maladaptive state, the output should be:
                {
                    "adaptive-state": {
                        "A": {"subelement": "(5) Content, happy, joy, hopeful", "highlighted_evidence": "exact quote"},
                        "B-S": {"subelement": "(1) Self care and improvement", "highlighted_evidence": "exact quote"},
                        "Presence": 3
                    },
                    "maladaptive-state": {
                        "A": {"subelement": "(2) Anxious/ fearful/ tense", "highlighted_evidence": "exact quote"},
                        "C-S": {"subelement": "(2) Self criticism", "highlighted_evidence": "exact quote"},
                        "D": {"subelement": "(2) Expectation that relatedness needs will not be met", "highlighted_evidence": "exact quote"},
                        "Presence": 4
                    }
                }"""

    dataset = []
    
    # Iterate through sorted DataFrame
    for idx, row in df.iterrows():
        # Skip posts without evidence

        dataset.append({
            'timeline_id': row['timeline_id'],
            'post_index': row['post_index'],
            'post_id': row['post_id'],
            'instruction': instruction,
            'input': f"Post: {row['text']}"
        })
    
    # Convert back to DataFrame to maintain order
    dataset_df = pd.DataFrame(dataset)
    dataset_df = dataset_df.sort_values(['timeline_id', 'post_index']).reset_index(drop=True)
    
    print(f"\nCreated {len(dataset_df)} instruction examples")
    print(f"From {dataset_df['timeline_id'].nunique()} timelines")
    
    return dataset_df


class CLPsychDataLoader:
    """Load CLPsych data with proper ordering"""
    
    def __init__(self, input_dir, split='train'):

        self.split = split
        if split == 'test':
            self.input_dir = os.path.join(input_dir, 'test')
        else:
            raise ValueError("Split must be one of 'train', 'val', or 'test'")
        self.df = None

    def load_clpsych_data(self, filepath):
        with open(filepath, 'r') as f:
            data = json.load(f)
            id, post = data['timeline_id'], data['posts']
        return id, post

    def load(self):
        """Load and parse JSON data into sorted DataFrame"""

        test_posts = []
        for file in glob(os.path.join(self.input_dir, '*.json')):
            # print(f"Loading {file}...")
            id, posts = self.load_clpsych_data(file)
            print(f"Loaded {id} with {len(posts)} posts.")
            for post in posts:
                # print(post)
                try:
                    assert 'post_id' in post
                    assert 'post' in post
                except AssertionError:
                    print(f"Timeline {id}, Post {post['post_id']} is missing required fields.")
                    continue
                test_posts.append({
                    'timeline_id': id,
                    'post_id': post['post_id'],
                    'post_index': post['post_index'],
                    'text': post['post'],
                })
        
        # Create DataFrame and sort by timeline_id and post_index
        self.df = pd.DataFrame(test_posts)
        self.df = self.df.sort_values(['timeline_id', 'post_index'])
        
        print(f"Loaded {len(self.df)} posts from {self.df['timeline_id'].nunique()} timelines")
        
        return self.df
    
    def verify_order(self):
        """Verify posts are in correct order within each timeline"""
        print("\n=== Verifying Post Order ===")
        issues = []
        
        for timeline_id in self.df['timeline_id'].unique():
            timeline_posts = self.df[self.df['timeline_id'] == timeline_id]
            indices = timeline_posts['post_index'].tolist()
            
            # Check if indices are in ascending order
            if indices != sorted(indices):
                issues.append(f"Timeline {timeline_id}: {indices}")
        
        if issues:
            print("❌ Order issues found:")
            for issue in issues:
                print(f"  {issue}")
            return False
        else:
            print("✅ All posts are in correct order")
            return True
    
    def get_stats(self):
        """Print dataset statistics"""
        print("\n=== Dataset Statistics ===")
        print(f"Total timelines: {self.df['timeline_id'].nunique()}")
        print(f"Total posts: {len(self.df)}")
        print(f"Avg posts per timeline: {len(self.df) / self.df['timeline_id'].nunique():.2f}")
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", help="Exact location of Pretrained Models",
                    type=str)
    args = parser.parse_args()
    MODELS = args.model_name

    # ========== STEP 1: Load Data ==========
    print("=" * 60)
    print("STEP 1: Loading Data")
    print("=" * 60)
    test_loader = CLPsychDataLoader('tasks12/', split='test')
    test_df = test_loader.load()
    print("\n" + "=" * 60)
    print("Test Set Stats")
    test_loader.verify_order()
    # ========== STEP 2: Create Instruction Dataset ==========
    print("\n" + "=" * 60)
    print("STEP 2: Creating Instruction Dataset")
    print("=" * 60)

    test_df = create_instruction_dataset(test_df)

    test_data = df_to_training_format(test_df[:10])

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

    print(f"✅ Loaded {len(test_data)} validation examples\n")

    # ========== 3. Generate Predictions ==========
    print("="*60)
    print("Generating predictions...")
    print("="*60)

    

    # Generate predictions
    predictions = []

    for item in tqdm(test_data, desc="Predicting"):
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
            
            predictions.append({
                'timeline_id':item['timeline_id'],
                'post_id': item['post_id'],
                'prediction': prediction
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
    with open('test_predictions.json', 'w') as f:
        json.dump(predictions, f, indent=2)
    print("✅ Predictions saved to test_predictions.json\n")