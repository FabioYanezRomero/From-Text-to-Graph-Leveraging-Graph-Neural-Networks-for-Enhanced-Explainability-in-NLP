#!/usr/bin/env python3
"""
Analyze problematic sample in AG News dataset that causes embedding generation to fail.
"""

from datasets import load_dataset
import pandas as pd

# Load the AG News dataset
dataset = load_dataset("ag_news", split="train")

# The sample where the error occurs
problem_idx = 39659
window = 5  # Number of samples to show before and after

# Get the problematic range
start_idx = max(0, problem_idx - window)
end_idx = min(len(dataset), problem_idx + window + 1)

# Create a list to store sample information
samples = []
for i in range(start_idx, end_idx):
    sample = dataset[i]
    samples.append({
        'index': i,
        'text': sample['text'],
        'label': sample['label'],
        'text_length': len(sample['text']),
        'words': len(sample['text'].split()),
        'has_special_chars': any(not c.isalnum() and not c.isspace() for c in sample['text'])
    })

# Convert to DataFrame for better visualization
df = pd.DataFrame(samples)

# Print analysis
print("\n=== Analysis of Samples Around Problematic Index ===")
print(f"Problematic index: {problem_idx}")
print(f"Dataset size: {len(dataset)}")
print("\nSamples around the problematic index:")
print(df[['index', 'text_length', 'words', 'label', 'has_special_chars']])

# Print full text of the problematic sample
print("\n=== Problematic Sample (Full Text) ===")
print(f"Index: {problem_idx}")
print(f"Label: {dataset[problem_idx]['label']}")
print(f"Text: {dataset[problem_idx]['text']}")

# Check for any extremely long samples
print("\n=== Length Analysis ===")
print(f"Max text length in dataset: {max(len(x) for x in dataset['text'])}")
print(f"Average text length: {sum(len(x) for x in dataset['text']) / len(dataset):.2f}")
print(f"Number of samples > 1000 chars: {sum(1 for x in dataset['text'] if len(x) > 1000)}")

print("\n=== Special Characters in Problematic Sample ===")
problem_text = dataset[problem_idx]['text']
print(''.join(c if not c.isalnum() and not c.isspace() else ' ' for c in problem_text).strip())
