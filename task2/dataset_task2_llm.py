import json
import pandas as pd
import numpy as np
from glob import glob
import os
from collections import defaultdict

DIMENSIONS = ['A', 'B-S', 'B-O', 'C-S', 'C-O', 'D']
LABEL_NAMES = ['Neither', 'Switch', 'Escalation']


def get_label(post):
    """Switch='S', Escalation='E', no change='0'"""
    switch_val = str(post.get('Switch', '0')).strip()
    escal_val  = str(post.get('Escalation', '0')).strip()
    if switch_val == 'S':
        return 1, 'Switch'
    elif escal_val == 'E':
        return 2, 'Escalation'
    return 0, 'Neither'


class CLPsychTask2DataLoader:
    """Load CLPsych data for Task 2 LLM approach"""

    def __init__(self, input_dir, split='train'):
        self.split = split
        if split == 'train':
            self.input_dir = os.path.join(input_dir, 'train')
        elif split == 'val':
            self.input_dir = os.path.join(input_dir, 'valid')
        elif split == 'test':
            self.input_dir = os.path.join(input_dir, 'test')
        else:
            raise ValueError("Split must be 'train', 'val', or 'test'")
        self.df = None

    def load(self):
        posts = []
        for file in glob(os.path.join(self.input_dir, '*.json')):
            with open(file, 'r') as f:
                data = json.load(f)
            tid = data['timeline_id']
            print(f"Loaded {tid} with {len(data['posts'])} posts.")
            for post in data['posts']:
                try:
                    assert 'post_id' in post
                    assert 'post' in post
                except AssertionError:
                    continue
                label_int, label_str = get_label(post)
                posts.append({
                    'timeline_id':  tid,
                    'post_id':      post['post_id'],
                    'post_index':   post['post_index'],
                    'text':         post['post'],
                    'label':        label_int,
                    'label_str':    label_str,
                    'well_being':   post.get('Well-being', 0),
                    'is_switch':    1 if str(post.get('Switch','0')).strip()=='S' else 0,
                    'is_escalation':1 if str(post.get('Escalation','0')).strip()=='E' else 0,
                })

        self.df = pd.DataFrame(posts)
        self.df = self.df.sort_values(['timeline_id', 'post_index']).reset_index(drop=True)
        print(f"Loaded {len(self.df)} posts from {self.df['timeline_id'].nunique()} timelines")
        return self.df

    def get_stats(self):
        print("\n=== Task 2 Label Distribution ===")
        total = len(self.df)
        for label in LABEL_NAMES:
            n = (self.df['label_str'] == label).sum()
            print(f"  {label}: {n} ({n/total*100:.1f}%)")


def build_timeline_context(df, current_idx, current_post_text, max_history=None):
    """
    Build context string from all previous posts in the timeline.
    max_history=None means use ALL previous posts.
    """
    history = df.iloc[:current_idx]

    if max_history is not None:
        history = history.tail(max_history)

    if len(history) == 0:
        return f"[This is the first post in the timeline]\n\nCurrent post:\n{current_post_text}"

    history_parts = []
    for _, row in history.iterrows():
        history_parts.append(f"Post {int(row['post_index'])}: {row['text']}")

    history_str = "\n\n".join(history_parts)
    return f"Previous posts in timeline:\n{history_str}\n\nCurrent post (Post {current_idx + 1}):\n{current_post_text}"


INSTRUCTION = """You are an expert in mental health trajectory analysis. 
Analyze a social media user's post history and identify if the CURRENT post represents a moment of change in their mental health trajectory.

Definitions:
- Switch (S): A DRASTIC, sudden change in the user's mood or mental state compared to their previous posts. A clear turning point.
- Escalation (E): A GRADUAL worsening or intensification of a negative mental health pattern across posts. Slow deterioration.
- Neither (O): No significant change — the post continues the same pattern as before.

You will be given the user's previous posts to understand the user's mental health trajectory, then classify the CURRENT post as Switch, Escalation, or Neither.
Consider the dataset has a strong class imbalance, of Neither: 65.7%, Switch: 17.1%, and Escalation: 17.1%

Output ONLY one of these three labels as a JSON object:
{
  "label": "Switch" or "Escalation" or "Neither",
  "reasoning": "A brief explanation (1-2 sentences) of why you chose this label."
}
"""


def create_task2_dataset(df, max_history=None):
    """
    Create instruction-tuning dataset for Task 2.
    Each example includes full timeline history as context.
    """
    dataset = []

    for tid, group in df.groupby('timeline_id'):
        group = group.sort_values('post_index').reset_index(drop=True)

        for i, row in group.iterrows():
            local_idx = group.index.get_loc(i)
            context = build_timeline_context(
                group, local_idx, row['text'], max_history=max_history
            )
            d = {
                'timeline_id': tid,
                'post_id':     row['post_id'],
                'post_index':  row['post_index'],
                'label':       row['label'],
                'label_str':   row['label_str'],
                'instruction': INSTRUCTION,
                'input':       context,
                'output':      json.dumps({
                    "label":       row['label_str'],
                    "reasoning": f"A brief explanation (1-2 sentences) of why you chose this label."
                }, indent=2),
            }


            dataset.append(d)

    dataset_df = pd.DataFrame(dataset)
    dataset_df = dataset_df.sort_values(['timeline_id', 'post_index']).reset_index(drop=True)
    print(f"\nCreated {len(dataset_df)} Task 2 examples from {dataset_df['timeline_id'].nunique()} timelines")

    # Print label distribution
    total = len(dataset_df)
    for label in LABEL_NAMES:
        n = (dataset_df['label_str'] == label).sum()
        print(f"  {label}: {n} ({n/total*100:.1f}%)")

    return dataset_df


def df_to_training_format(df):
    return [
        {
            'timeline_id': r['timeline_id'],
            'post_id':     r['post_id'],
            'label':       r['label'],
            'label_str':   r['label_str'],
            'instruction': r['instruction'],
            'input':       r['input'],
            'output':      r['output'],
        }
        for _, r in df.iterrows()
    ]