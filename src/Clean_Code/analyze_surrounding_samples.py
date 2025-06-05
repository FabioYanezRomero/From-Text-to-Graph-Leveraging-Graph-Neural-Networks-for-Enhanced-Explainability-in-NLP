#!/usr/bin/env python3
"""
Analyze samples surrounding the problematic index in AG News dataset.
"""

from datasets import load_dataset
import pandas as pd

def print_sample_info(dataset, idx, title):
    sample = dataset[idx]
    print(f"\n{title} (Index {idx}):")
    print(f"Label: {sample['label']}")
    print(f"Text length: {len(sample['text'])} chars, {len(sample['text'].split())} words")
    print(f"Text: {sample['text']}")
    
    # Print character codes of non-alphanumeric characters
    special_chars = [(i, c, ord(c)) for i, c in enumerate(sample['text']) 
                    if not c.isalnum() and not c.isspace()]
    if special_chars:
        print("\nSpecial characters (position, char, unicode):")
        for pos, char, code in special_chars:
            print(f"  {pos:3d}: '{char}' (U+{code:04X})")

# Load the dataset
dataset = load_dataset("ag_news", split="train")

# Analyze samples around the problematic index
problem_idx = 39659
window = 2  # Number of samples before and after to analyze

print(f"Analyzing samples around index {problem_idx} in AG News dataset\n" + "="*70)

# Print previous samples
for i in range(problem_idx - window, problem_idx):
    print_sample_info(dataset, i, f"Previous Sample {i - (problem_idx - window) + 1}")

# Print the problematic sample
print_sample_info(dataset, problem_idx, "\nPROBLEMATIC SAMPLE")

# Print next samples
for i in range(problem_idx + 1, problem_idx + window + 1):
    print_sample_info(dataset, i, f"\nNext Sample {i - problem_idx}")

# Print memory usage info
print("\n" + "="*70)
print("Memory Analysis:")
print(f"Total samples in dataset: {len(dataset)}")
print(f"Average text length: {sum(len(x['text']) for x in dataset)/len(dataset):.1f} chars")
print(f"Max text length: {max(len(x['text']) for x in dataset)} chars")
