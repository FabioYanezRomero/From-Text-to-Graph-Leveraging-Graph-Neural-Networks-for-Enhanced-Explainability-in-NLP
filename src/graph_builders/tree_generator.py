#!/usr/bin/env python3
"""
Tree Generator Module

This module provides functionality to generate constituency trees from text datasets.
It leverages the Graph_Generation package to create constituency parse trees.
"""

import os
import argparse
import torch
import inspect
from tqdm import tqdm
from datasets import load_dataset
from torch.utils.data import DataLoader
import pickle as pkl
import importlib
import os as _os
import pkgutil
from .registry import GENERATORS


def _auto_discover_builders():
    """Import all modules in this package except known non-builder files.

    This triggers registration decorators in modules like constituency.py, syntactic.py,
    or any new builder added in the future.
    """
    pkg_dir = _os.path.dirname(__file__)
    # When executed as a module (__name__ == "__main__"), __package__ holds the package name
    pkg_name = __package__ or 'src.graph_builders'
    exclude = {"__init__", "__main__", "tree_generator", "base_generator", "registry", "semantic"}
    for m in pkgutil.iter_modules([pkg_dir]):
        if m.ispkg:
            continue
        if m.name in exclude:
            continue
        modname = f"{pkg_name}.{m.name}"
        try:
            importlib.import_module(modname)
        except ModuleNotFoundError as e:
            print(f"[warn] Skipping builder module '{modname}' due to missing dependency: {e}")
        except Exception as e:
            print(f"[warn] Skipping builder module '{modname}' due to import error: {e}")


def _parse_graph_type(graph_type: str):
    """Parse graph type strings like:
    - "constituency", "syntactic"
    - "window.word.k5", "window.token.k3"
    - "ngrams.word.n3", "skipgrams.word.k2", "skipgrams.token.k2"
    Returns (name_for_registry, kwargs_dict)
    """
    parts = graph_type.split('.')
    base = parts[0]
    unit = None
    params = {}
    for p in parts[1:]:
        if p in ("word", "token"):
            unit = p
        elif p and p[0].isalpha():
            # like k5, n3
            key = p[0]
            val = p[1:]
            if val.isdigit():
                params[key] = int(val)
            else:
                params[key] = val
    if base in ("constituency", "syntactic"):
        return base, {}
    # registry keys are base.unit (default unit=word)
    if unit is None:
        unit = 'word'
    return f"{base}.{unit}", params


def build_trees(graph_type, dataset_name, subset, batch_size, device, output_dir, model_name=None):
    """
    Build trees from a dataset
    
    Args:
        graph_type (str): Type of graph to generate (constituency, syntactic, semantic)
        dataset_name (str): Name of the dataset
        subset (str): Subset of the dataset (train, test, validation)
        batch_size (int, optional): Batch size for processing. Defaults to DEFAULT_BATCH_SIZE
        device (str, optional): Device to run on. Defaults to DEFAULT_DEVICE
    """
    # Create appropriate tree generator via registry
    _auto_discover_builders()
    reg_name, params = _parse_graph_type(graph_type)
    try:
        GenCls = GENERATORS.get(reg_name)
    except KeyError as e:
        available = ", ".join(sorted(GENERATORS.names()))
        raise NotImplementedError(f"Graph type '{graph_type}' is not supported. Available: {available}") from e
    # Instantiate with supported kwargs (device, k, n, model_name, etc.)
    init_params = {k: v for k, v in params.items()}
    sig = inspect.signature(GenCls.__init__)
    if 'device' in sig.parameters:
        init_params['device'] = device
    if model_name is not None and 'model_name' in sig.parameters:
        init_params['model_name'] = model_name
    generator = GenCls(**init_params)
    # Load dataset
    instance = load_dataset(dataset_name, split=subset)
    instance.set_format(type='torch')
    dataloader = DataLoader(dataset=instance, batch_size=batch_size, shuffle=False)
    
    # Create output directory
    output_path = os.path.join(output_dir, dataset_name, subset, graph_type)
    os.makedirs(output_path, exist_ok=True)
    
    # Process dataset
    iterator = 0
    for batch in tqdm(dataloader, desc=f"Processing {dataset_name}/{subset} {graph_type} trees"):
        # Extract sentences and labels
        try:
            sentences = batch['sentence']
        except:
            sentences = batch['text']
        
        labels = batch['label']
        
        # Generate trees
        trees = generator.get_graph(sentences)
        processed_data = [(trees, labels)]
        
        # Save trees
        with open(f"{output_path}/{iterator}.pkl", 'wb') as f:
            pkl.dump(processed_data, f)
        iterator += 1


def process_dataset(graph_type, dataset, subsets, batch_size, device, output_dir, model_name=None):
    """
    Process multiple datasets to generate constituency trees
    
    Args:
        graph_type (str): Type of graph to generate (constituency, syntactic, semantic)
        dataset (str): Dataset to process
        subsets (list, optional): List of subsets to process. If None, uses DEFAULT_SUBSETS
        batch_size (int, optional): Batch size for processing. Defaults to DEFAULT_BATCH_SIZE
        device (str, optional): Device to run on. Defaults to DEFAULT_DEVICE
    """
    
    # Process the dataset
    print(f"Processing dataset: {dataset}")
        
    # Create data directory
    os.makedirs(output_dir, exist_ok=True)
        
    # Handle each subset according to the given dataset
    subsets = subsets_handler(dataset, subsets)
    for subset in subsets:
        print(f"Processing subset: {subset}")
        
        # Clear GPU memory
        if device.startswith('cuda'):
            torch.cuda.empty_cache()
            
        # Build trees
        build_trees(
                graph_type=graph_type,
                dataset_name=dataset,
                subset=subset,
                batch_size=batch_size,
                device=device,
                output_dir=output_dir,
                model_name=model_name,
            )
        
        # Clear GPU memory again
        if device.startswith('cuda'):
            torch.cuda.empty_cache()


def subsets_handler(dataset, subsets):
    if dataset == "SetFit/ag_news":
        if "validation" in subsets:
            subsets.remove("validation")
    elif dataset == "stanfordnlp/sst2":
        if "test" in subsets:
            subsets.remove("test")
    return subsets


def main(args):
    # Process datasets
    process_dataset(
        graph_type=args.graph_type,
        dataset=args.dataset,
        subsets=args.subsets,
        batch_size=args.batch_size,
        device=args.device,
        output_dir=args.output_dir,
        model_name=args.model_name,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph_type", type=str, default="constituency")
    parser.add_argument("--dataset", type=str, default="SetFit/ag_news")
    parser.add_argument("--subsets", nargs='+', type=str, default=["train", "validation", "test"])
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda:0")
    import os as _os
    _base = _os.environ.get('GRAPHTEXT_OUTPUT_DIR', 'outputs')
    parser.add_argument("--output_dir", type=str, default=f"{_base}/graphs")
    parser.add_argument("--model_name", type=str, required=True, help="HF model or checkpoint path for tokenization/embeddings where required")
    args = parser.parse_args()
    main(args)
