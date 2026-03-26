import json
import pandas as pd
import numpy as np
from glob import glob
import os   
from collections import defaultdict
from torch.utils.data import Dataset



TAXONOMY = {
    'A': {
        'adaptive': [
            ' (1) Calm/ laid back',
            ' (3) Sad, Emotional pain grieving',
            ' (5) Content, happy, joy, hopeful',
            ' (7) Vigor / energetic',
            ' (9) Justifiable anger/ assertive',
            ' anger, justifiable outrage',
            '(11) Proud',
            '(13) Feel loved, belong'
            ],
        'maladaptive': [
            '(2) Anxious/ fearful/ tense',
            '(4) Depressed, despair, hopeless',
            '(6) Mania',
            '(8) Apathic, don’t care, blunted',
            '(10) Angry (aggression), disgust contempt'
            '(12) Ashamed, guilty',
            '(14) Feel lonely'
            ]
          },
    'B-S': {
        'adaptive': [
            '(1) Self care and improvement'
            ],
        'maladaptive': [
            '(2) Self harm, neglect and avoidance'
            ]
    },
    'B-O': {
        'adaptive': 
            [
            '(1) Relating behavior',
            '(3) Autonomous or adaptive control behavior'
            ],
        'maladaptive': [
            '(2) Fight or flight behavior',
            '(4) Over controlled or controlling behavior'
            ]
    },
    'C-S': 
        {
        'adaptive': 
            [
            '(1) Self-acceptance and compassion'
            ],
          'maladaptive': 
              [
              '(2) Self criticism'
              ]
        },
    'C-O': 
        {
            'adaptive': 
            [
                '(1) Perception of the other as related',
                '(3) Perception of the other as facilitating autonomy needs'
            ],
              'maladaptive': 
              [
                  '(4) Perception of the other as blocking autonomy needs',
                  '(2) Perception of the other as detached or over attached'
                  ]
          },
    'D': 
    {
        'adaptive': 
            [
            '(5) Competence, self esteem, self-care',
            '(1) Relatedness',
            '(3) Autonomy and adaptive control'
            ],
      'maladaptive': 
          [
              '(6) Expectation that competence needs will not be met',
              '(4) Expectation that autonomy needs will not be met',
              '(2) Expectation that relatedness needs will not be met'
          ]
    }
}


def get_taxonomy_string():
    """Format taxonomy for prompt"""
    lines = []
    for dim, categories in TAXONOMY.items():
        lines.append(f"\n**{dim}**")
        lines.append("Adaptive:")
        for cat in categories['adaptive']:
            lines.append(f"  - {cat}")
        lines.append("Maladaptive:")
        for cat in categories['maladaptive']:
            lines.append(f"  - {cat}")
    return "\n".join(lines)



class CLPsychDataLoader:
    """Load CLPsych data with proper ordering"""
    
    def __init__(self, input_dir, split='train'):

        self.split = split
        if split == 'train':
            self.input_dir = os.path.join(input_dir, 'train')
        elif split == 'val':
            self.input_dir = os.path.join(input_dir, 'valid')
        elif split == 'test':
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

        train_posts = []
        for file in glob(os.path.join(self.input_dir, '*.json')):
            # print(f"Loading {file}...")
            id, posts = self.load_clpsych_data(file)
            print(f"Loaded {id} with {len(posts)} posts.")
            for post in posts:
                # print(post)
                try:
                    assert 'post_id' in post
                    assert 'post' in post
                    assert 'evidence' in post
                except AssertionError:
                    print(f"Timeline {id}, Post {post['post_id']} is missing required fields.")
                    continue
                train_posts.append({
                    'timeline_id': id,
                    'post_id': post['post_id'],
                    'post_index': post['post_index'],
                    'text': post['post'],
                    'well_being': post.get('Well-being', 0),
                    'is_switch': post.get('Switch', 0),
                    'is_escalation': post.get('Escalation', 0),
                    'evidence': post['evidence']
                })
        
        # Create DataFrame and sort by timeline_id and post_index
        self.df = pd.DataFrame(train_posts)
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
        
        # print(f"\nSwitch events: {self.df['is_switch'].sum()} ({self.df['is_switch'].mean()*100:.1f}%)")
        # print(f"Escalation events: {self.df['is_escalation'].sum()} ({self.df['is_escalation'].mean()*100:.1f}%)")
        
        # ABCD presence
        print("\n=== ABCD Element Presence ===")
        adaptive_counts = defaultdict(int)
        maladaptive_counts = defaultdict(int)
        
        for _, row in self.df.iterrows():
            evidence = row['evidence']
            
            # Adaptive
            if 'adaptive-state' in evidence:
                for dim in ['A', 'B-S', 'B-O', 'C-S', 'C-O', 'D']:
                    if dim in evidence['adaptive-state'] and evidence['adaptive-state'][dim].get('Category'):
                        adaptive_counts[dim] += 1
            
            # Maladaptive
            if 'maladaptive-state' in evidence:
                for dim in ['A', 'B-S', 'B-O', 'C-S', 'C-O', 'D']:
                    if dim in evidence['maladaptive-state'] and evidence['maladaptive-state'][dim].get('Category'):
                        maladaptive_counts[dim] += 1
        
        print("\nAdaptive:")
        for dim in ['A', 'B-S', 'B-O', 'C-S', 'C-O', 'D']:
            print(f"  {dim}: {adaptive_counts[dim]}")
        
        print("\nMaladaptive:")
        for dim in ['A', 'B-S', 'B-O', 'C-S', 'C-O', 'D']:
            print(f"  {dim}: {maladaptive_counts[dim]}")

# Load data
# train_loader = CLPsychDataLoader('tasks12/', split='train')
# val_loader = CLPsychDataLoader('tasks12/', split='val')
# # test_loader = CLPsychDataLoader('tasks12/', split='test')
# df = val_loader.load()
# val_loader.verify_order()
# val_loader.get_stats()


def format_evidence_as_json(evidence):
    """Convert evidence dict to clean JSON string"""
    output = {
        'adaptive-state': {},
        'maladaptive-state': {}
    }
    
    # Process adaptive state
    if 'adaptive-state' in evidence:
        for dim, data in evidence['adaptive-state'].items():
            if dim == 'Presence':
                # print(data)
                output['adaptive-state']['Presence'] = data
            if dim != 'Presence' and isinstance(data, dict):
                if 'Category' in data and data['Category']:
                    output['adaptive-state'][dim] = {
                        'Category': data['Category'],
                        'highlighted_evidence': data.get('highlighted_evidence', '')
                    }
    
    # Process maladaptive state
    if 'maladaptive-state' in evidence:
        for dim, data in evidence['maladaptive-state'].items():
            if dim == 'Presence':
                output['maladaptive-state']['Presence'] = data
            if dim != 'Presence' and isinstance(data, dict):
                if 'Category' in data and data['Category']:
                    output['maladaptive-state'][dim] = {
                        'Category': data['Category'],
                        'highlighted_evidence': data.get('highlighted_evidence', '')
                    }
    
    return json.dumps(output, indent=2)


def df_to_training_format(df):
    """
    Convert DataFrame to list of dicts for training
    Order is already preserved in DataFrame
    """
    training_data = []
    
    for idx, row in df.iterrows():
        training_data.append({
            'instruction': row['instruction'],
            'input': row['input'],
            'output': row['output']
        })
    
    return training_data

# Convert to training format
# train_data = df_to_training_format(train_df)
# val_data = df_to_training_format(val_df)

# print(f"\nTrain data: {len(train_data)} examples")
# print(f"Val data: {len(val_data)} examples")



def create_instruction_dataset(df):
    """
    Convert DataFrame to instruction-tuning format
    Maintains order from sorted DataFrame
    """
    
    instruction = """You are an expert in mental health text analysis using the MIND framework. Analyze the social media post and identify ABCD (Affect, Behavior, Cognition, Desire) elements and their subelements. Analyze the post and rate the presence of adaptive and maladaptive self-states based on subelements on a scale of 1-5.

        For each post, identify:
        - Adaptive self-state elements (supporting well-being)
        - Maladaptive self-state elements (negative patterns)

        ABCD Dimensions:
        - A: Affect (Emotional tone or mood)
        - B-S: Behavior toward Self (Actions or tendencies directed inward)
        - B-O: Behavior toward Others (Actions or tendencies directed outward)
        - C-S: Cognition toward Self (C-S) (Beliefs, interpretations, and appraisals about self)
        - C-O: Cognition toward Others (C-O) (Beliefs, interpretations, and appraisals about others)
        - D: Desire (Motivations, needs, wishes, and expectations)

        Subelements:
        """ + get_taxonomy_string() + """

        Output in JSON format. For example,

        [
            {
                "timeline_id": "91b6a42835",
                "post_id": "28641e5b6d",
                "adaptive-state": {
                "Presence": 5,
                "B-S": { "subelements": 1 },
                "B-O": { "subelements": 1 },
                "C-S": { "subelements": 1 },
                "D": { "subelements": 3 }
                },
                "maladaptive-state": {
                "Presence": 2,
                "C-S": { "subelements": 2 }
                }
            },
            {
                "timeline_id": "306d938d4b",
                "post_id": "308b0c2c6c",
                "adaptive-state": {
                "Presence": 2,
                "B-S": { "subelements": 1 },
                "B-O": { "subelements": 1 }
                },
                "maladaptive-state": {
                "Presence": 2,
                "A": { "subelements": 2 }
                }
            }
        ]


    
        Rules:
        1. Each dimension can appear in adaptive, maladaptive, both, or neither
        2. Evidence must be exact quote from the post (3-15 words)
        3. Only include dimensions you detect in the post
        4. Multiple dimensions can be present simultaneously
        5. Brief analysis of adaptive elements
        6. Adaptive and Maladaptive presence score (1-5) based on the subelement, Only consider integar value
        7. Overall reasoning"""

    dataset = []
    
    # Iterate through sorted DataFrame
    for idx, row in df.iterrows():
        # Skip posts without evidence
        if not row.get('evidence'):
            continue
        
        # Check if there's actual content
        has_content = False
        evidence = row['evidence']
        
        for state in ['adaptive-state', 'maladaptive-state']:
            if state in evidence:
                for dim, data in evidence[state].items():
                    if dim != 'Presence' and isinstance(data, dict):
                        if data.get('Category'):
                            has_content = True
                            break
        
        if not has_content:
            continue
        
        dataset.append({
            'timeline_id': row['timeline_id'],
            'post_index': row['post_index'],
            'post_id': row['post_id'],
            'instruction': instruction,
            'input': f"Post: {row['text']}",
            'output': format_evidence_as_json(row['evidence'])
        })
    
    # Convert back to DataFrame to maintain order
    dataset_df = pd.DataFrame(dataset)
    dataset_df = dataset_df.sort_values(['timeline_id', 'post_index']).reset_index(drop=True)
    
    print(f"\nCreated {len(dataset_df)} instruction examples")
    print(f"From {dataset_df['timeline_id'].nunique()} timelines")
    
    return dataset_df






class ABCDInstructionDataset(Dataset):
    """
    Dataset for ABCD instruction tuning
    Preserves order from input list
    """
    
    def __init__(self, data, tokenizer, max_length=2048):
        """
        Args:
            data: List of dicts with 'instruction', 'input', 'output'
                  (already in correct order)
            tokenizer: Hugging Face tokenizer
            max_length: Maximum sequence length
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """Get item at index (maintains order)"""
        item = self.data[idx]
        
        # Format as Llama-3 chat format
        messages = [
            {"role": "system", "content": item['instruction']},
            {"role": "user", "content": item['input']},
            {"role": "assistant", "content": item['output']}
        ]
        
        # Apply chat template
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        
        # Tokenize
        encodings = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        return {
            'input_ids': encodings['input_ids'].squeeze(),
            'attention_mask': encodings['attention_mask'].squeeze(),
            'labels': encodings['input_ids'].squeeze()
        }

# Note: We'll create the actual dataset after loading the model
# to avoid loading tokenizer twice


if __name__=='__main__':
    train_loader = CLPsychDataLoader('..tasks12/', split='train')
    val_loader = CLPsychDataLoader('..tasks12/', split='val')
    train_df = train_loader.load()
    print("Training Set Stats")
    val_df = val_loader.load()
    train_loader.verify_order()
    train_loader.get_stats()
    print("\n" + "=" * 60)
    print("Validation Set Stats")
    val_loader.verify_order()
    val_loader.get_stats()