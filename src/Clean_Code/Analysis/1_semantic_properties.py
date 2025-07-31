import pickle as pkl
import numpy as np
from tqdm import tqdm
from utils import *
from arguments import *
import spacy
import os

args = arguments()


DATASET_LABELS = {"ag-news": {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"},
                  "sst2": {0: "Negative", 1: "Positive"}}

dataset = "ag-news"
nlp = spacy.load("en_core_web_sm")
stop_words = nlp.Defaults.stop_words
punctuations = nlp.Defaults.tokenizer_exceptions.keys()
symbols = {'!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~', '', ' '}

    
"""
predictions: sample list of predictions obtained from the GNN
"""

total_embeddings = {}
Summary = {}
iterator = 0
folder = os.listdir(f"/usrvol/experiments/explainability_results/{dataset}")
for i in tqdm(range(len(folder)), desc="Processing files: "):
    with open(f"/usrvol/experiments/explainability_results/{dataset}/results_{i}.pkl", "rb") as f:
        explanations = pkl.load(f)
        
    for explanation_id, explanation in tqdm(explanations.items()):
        Summary[iterator] = {}
        prediction = explanation[5]
        label = explanation[6]
        subgraph = explanation[1]['subgraph']
        graph = to_networkx(explanation[3])
        features = explanation[3].x

        for key, value in explanation[3].dict_nodes[0].items():
            graph.nodes[key]['label'] = value
        #nx.set_node_attributes(graph, explanation[2].dict_nodes, name='label')
        self_loops = list(nx.selfloop_edges(graph))
        graph.remove_edges_from(self_loops)
        important_words, unimportant_words, important_sequence, word_embeddings, special_embeddings, important_graph, unimportant_graph, depth = semantic_features(graph, subgraph, features)
        total_embeddings.update(word_embeddings)
        total_embeddings.update(special_embeddings)
            
        Summary[iterator]['important_words'] = important_words
        Summary[iterator]['unimportant_words'] = unimportant_words
        Summary[iterator]['important_sequence'] = important_sequence
        Summary[iterator]['important_graph'] = important_graph
        Summary[iterator]['unimportant_graph'] = unimportant_graph
        Summary[iterator]['depth'] = depth
        Summary[iterator]['prediction'] = prediction
        Summary[iterator]['label'] = label                                  

        iterator += 1

for key, value in Summary.items():
    Summary[key]['important_words'] = [word for word in value['important_words'] if word not in stop_words and word not in punctuations and word not in symbols]
    Summary[key]['unimportant_words'] = [word for word in value['unimportant_words'] if word not in stop_words and word not in punctuations and word not in symbols]
  
with open(f"/usrvol/experiments/explainability_results/total_embeddings_{dataset}.pkl", "wb") as f:
    pkl.dump(total_embeddings, f)
    
with open(f"/usrvol/experiments/explainability_results/summary_{dataset}.pkl", "wb") as f:
    pkl.dump(Summary, f)



