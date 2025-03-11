from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Any, Optional, Union, Tuple


""" 
With this dataloader we are going to generate the different graphs from sentences based on the format of 
each dataset and the configuration of the graphs we want to generate.
"""

class GenericTextDataset(Dataset):
    """
    Generic dataset class for handling various text dataset formats.
    This class can process datasets with single sentences, sentence pairs, or multiple sentences.
    """
    
    def __init__(self, data: List[Dict[str, Any]], 
                 sentence_keys: Optional[Union[str, List[str]]] = None,
                 label_key: str = 'label'):
        """
        Initialize the dataset with data items and configuration.
        
        Args:
            data (List[Dict]): List of dictionaries containing sentence(s) and labels
            sentence_keys (str or List[str], optional): 
                - If str: Key for accessing a single sentence
                - If List[str]: Keys for accessing multiple sentences (e.g., ['sentence1', 'sentence2'])
                - If None: Auto-detect keys containing "sentence" in their name
            label_key (str): Key to access the label in the data dictionaries
        """
        self.data = data
        self.label_key = label_key
        
        # Auto-detect sentence keys if not provided
        if sentence_keys is None:
            if len(data) > 0:
                sample = data[0]
                # Find all keys that might contain sentences
                possible_keys = [key for key in sample.keys() 
                                if 'sentence' in key.lower() or 'text' in key.lower()]
                if possible_keys:
                    self.sentence_keys = sorted(possible_keys)  # Sort to ensure consistent order
                else:
                    raise ValueError("Could not auto-detect sentence keys. Please specify sentence_keys.")
            else:
                raise ValueError("Empty dataset provided and no sentence_keys specified.")
        else:
            self.sentence_keys = [sentence_keys] if isinstance(sentence_keys, str) else sentence_keys
            
    def __len__(self):
        """
        Return the number of items in the dataset.
        """
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Return the sentence(s) and label at the given index.
        
        Returns:
            tuple: (sentence(s), label) where sentence(s) can be:
                  - A single sentence string if only one sentence key is specified
                  - A tuple of sentence strings if multiple sentence keys are specified
        """
        # Get sentences
        if len(self.sentence_keys) == 1:
            sentences = self.data[idx][self.sentence_keys[0]]
        else:
            sentences = tuple(self.data[idx][key] for key in self.sentence_keys)
        
        # Get label
        label = self.data[idx][self.label_key]
        
        return sentences, label


# Legacy dataset classes for backward compatibility
class TextPairDataset(GenericTextDataset):
    """Dataset class for sentence pairs (backward compatibility)"""
    def __init__(self, sentences: list, label_key: str = 'gold_label'):
        super().__init__(data=sentences, 
                         sentence_keys=['sentence1', 'sentence2'],
                         label_key=label_key)
    
    def __getitem__(self, idx):
        """Override to maintain the exact same return format as before"""
        sentences, label = super().__getitem__(idx)
        return sentences[0], sentences[1], label


class SNLIDataset(TextPairDataset):
    """Legacy dataset class for SNLI data"""
    def __init__(self, sentences: list):
        super().__init__(sentences, label_key='gold_label')


class RTEDataset(TextPairDataset):
    """Legacy dataset class for RTE data"""
    def __init__(self, sentences: list):
        super().__init__(sentences, label_key='label')


class SciTailDataset(TextPairDataset):
    """Legacy dataset class for SciTail data"""
    def __init__(self, sentences: list):
        super().__init__(sentences, label_key='gold_label')


# Example usage classes for different dataset formats
class SingleSentenceDataset(GenericTextDataset):
    """Dataset class for single sentence classification tasks (e.g., SST-2)"""
    def __init__(self, data: list, sentence_key: str = 'sentence', label_key: str = 'label'):
        super().__init__(data=data, sentence_keys=sentence_key, label_key=label_key)


class MultiSentenceDataset(GenericTextDataset):
    """Dataset class for tasks with multiple sentences (e.g., multi-document summarization)"""
    def __init__(self, data: list, sentence_keys: List[str], label_key: str = 'label'):
        super().__init__(data=data, sentence_keys=sentence_keys, label_key=label_key)