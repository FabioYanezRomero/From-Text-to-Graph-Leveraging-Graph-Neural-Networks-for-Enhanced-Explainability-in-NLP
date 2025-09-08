"""
Syntactic Tree Generator Module

This module provides functionality to create syntactic trees from sentences.
It leverages the Stanza NLP library to generate syntactic trees and converts them
into directed graphs where nodes represent words and edges represent syntactic relations.
"""

from typing import List, Dict, Union, Any, Optional
import os
import networkx as nx
import torch
import torch.cuda
import stanza
from .base_generator import BaseTreeGenerator
import concurrent.futures



class SyntacticTreeGenerator(BaseTreeGenerator):
    """
    Creates syntactic trees from sentences.

    This class processes sentences using a Stanza syntactic parser and converts
    the parsed trees into directed graphs. Each node in the graph represents
    either a syntactic phrase or a word from the sentence.

    Attributes:
        model (str): Name or configuration of the syntactic parser model.
        property (str): Property type, always set to 'syntactic'.
        nlp: The loaded Stanza pipeline.
        device (str): The device to run the parser on (CPU or CUDA).
    """

    def __init__(self, device: str = 'cuda:0'):
        """
        Initialize the syntactic parser with Stanza.

        Args:
            device (str, optional): Device to run the parser on. Defaults to 'cuda:0'.

        Raises:
            RuntimeError: If the specified device is not available.
        """
        super().__init__(property='syntactic', device=device)
        
        # Verify device availability for better error handling
        if device.startswith('cuda') and not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. Please use 'cpu' instead or check CUDA installation.")
        
        self.device = device
        if device.startswith('cuda'):
            torch.cuda.set_device(device)
        
        # Set up Stanza configuration
        stanza_device = 'gpu' if device.startswith('cuda') else 'cpu'

        # Download the combined_charlm model (best accuracy)
        stanza.download('en', package='combined_charlm', processors={
            'tokenize': 'default',
            "lemma": "combined_nocharlm",
            'pos': 'default',
            'depparse': 'default'  # Uses BERT for best accuracy
        })
        
        self.nlp = stanza.Pipeline(
            lang='en',
            processors='tokenize,pos,lemma,depparse',  
            package={'depparse': 'combined_charlm'},
            use_gpu=stanza_device == 'gpu',
            download_method=stanza.DownloadMethod.NONE,  # We already downloaded the models
            tokenize_pretokenized=False,
            tokenize_no_ssplit=True,
            pos_batch_size=1000,
            depparse_batch_size=1000,
            depparse_pretagged=True,  # Use POS tags from the POS tagger
        )
        print("Stanza pipeline with combined_charlm model initialized successfully.")
        

    def _parse(self, sentences: List[str]):
        """
        Parse sentences using the Stanza syntactic parser, respecting the batch size.

        Args:
            sentences (List[str]): List of sentences to parse.

        Returns:
            List: List of parsed syntactic trees, one per input sentence.
        """
        results = []
        for s in sentences:
            results.append(self.nlp(str(s)))
        return results

    def _build_graph(self, parsed_results: stanza.Document) -> List[nx.DiGraph]:
        """
        Build graphs from the parsed data in parallel across sentences.
        """
        def build_graph_for_sentence(sentence):
            graph = nx.DiGraph()
            for token in sentence.tokens:
                word = token.words[0]
                graph.add_node(word.id, text=word.text, pos=word.upos)
                if word.head != 0:
                    graph.add_edge(word.head, word.id, deprel=word.deprel)
            return graph

        with concurrent.futures.ThreadPoolExecutor() as executor:
            graphs = list(executor.map(build_graph_for_sentence, parsed_results.sentences))
        return graphs

    def get_graph(self, sentences: List[str], ids: Optional[List[str]] = None) -> List[nx.DiGraph]:
        """
        Generate syntactic dependency graphs for a list of sentences.

        Args:
            sentences (List[str]): List of sentences to process.
            ids (List[str], optional): Not used, for interface compatibility.
    
        Returns:
            List[nx.DiGraph]: List of syntactic dependency graphs.
        """
        parsed_batches = self._parse(sentences)
        graphs = []
        for parsed in parsed_batches:
            graphs.extend(self._build_graph(parsed))
        return graphs