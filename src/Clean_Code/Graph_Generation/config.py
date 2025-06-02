"""
Configuration module for Graph Generation

This module contains configuration settings for the graph generation process.
"""

# Only constituency graphs will be generated
GRAPH_TYPES = ["constituency"]

# Default model for constituency parsing
DEFAULT_MODELS = {
    "constituency": "crf-con-en"  # Standard LSTM-based constituency parser
}

# Available models (only constituency)
AVAILABLE_MODELS = {
    "constituency": [
        "crf-con-roberta-en",  # RoBERTa-based constituency parser
        "crf-con-en"           # Standard LSTM-based constituency parser
    ]
}

# Default datasets
DEFAULT_DATASETS = ["stanfordnlp/sst2", "SetFit/ag_news"]

# Default subsets
DEFAULT_SUBSETS = ["train", "test", "validation"]

# Default batch size for processing
DEFAULT_BATCH_SIZE = 256

# Default device
DEFAULT_DEVICE = "cuda:0"

# Default directories
DEFAULT_OUTPUT_DIR = "/app/src/Clean_Code/output/text_graphs"
DEFAULT_DATA_DIR = "/app/src/Clean_Code/output/text_graphs"
