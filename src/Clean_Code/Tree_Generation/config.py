"""
Configuration module for Tree Generation

This module contains configuration settings for the constituency tree generation process.
"""

# Only constituency trees will be generated
GRAPH_TYPES = ["constituency"]

# Default model for constituency parsing
DEFAULT_MODELS = {
    "constituency": "crf-con-roberta-en"  # RoBERTa-based constituency parser
}

# Available models for constituency parsing
AVAILABLE_MODELS = {
    "constituency": [
        "crf-con-roberta-en",  # RoBERTa-based constituency parser
        "crf-con-en"           # Standard LSTM-based constituency parser
    ]
}

# Default datasets
DEFAULT_DATASETS = ["SetFit/ag_news"]    # "stanfordnlp/sst2", "SetFit/ag_news"

# Default subsets
DEFAULT_SUBSETS = ["train", "test", "validation"]

# Default batch size for processing
DEFAULT_BATCH_SIZE = 256

# Default device
DEFAULT_DEVICE = "cuda:0"

# Default directories
DEFAULT_OUTPUT_DIR = "/app/src/Clean_Code/output/text_trees"
DEFAULT_DATA_DIR = "/app/src/Clean_Code/output/text_trees"
