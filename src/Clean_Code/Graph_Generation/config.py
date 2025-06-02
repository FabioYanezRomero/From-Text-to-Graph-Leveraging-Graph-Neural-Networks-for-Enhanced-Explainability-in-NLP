"""
Configuration module for Graph Generation

This module contains configuration settings for the graph generation process.
"""

# Available graph types
GRAPH_TYPES = ["syntactic", "semantic", "constituency"]

# Default models for each graph type
DEFAULT_MODELS = {
    "syntactic": "dep-biaffine-roberta-en",
    "semantic": "sdp-vi-roberta-en",
    "constituency": "con-crf-roberta-en"
}

# Available models for each graph type
AVAILABLE_MODELS = {
    "syntactic": [
        "dep-biaffine-en",
        "dep-biaffine-roberta-en",
        "dep-crf2o-en",
        "dep-crf2o-roberta-en"
    ],
    "semantic": [
        "sdp-biaffine-en",
        "sdp-vi-en",
        "sdp-vi-roberta-en"
    ],
    "constituency": [
        "con-crf-en",
        "con-crf-roberta-en",
        "con-biaffine-en",
        "con-biaffine-roberta-en"
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

# Default output directory
DEFAULT_OUTPUT_DIR = "/app/processed_data"

# Default data directory
DEFAULT_DATA_DIR = "/app/data/text_graphs"
