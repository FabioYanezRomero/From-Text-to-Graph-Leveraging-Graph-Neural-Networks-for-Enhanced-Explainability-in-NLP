import os
import math
import copy
import torch
import numpy as np
import networkx as nx
from rdkit import Chem
from torch import Tensor
from textwrap import wrap
from functools import partial
from collections import Counter
from typing import List, Tuple, Dict
from torch_geometric.data import Batch, Data
from torch_geometric.utils import to_networkx
from typing import Callable, Union, Optional
import matplotlib.pyplot as plt
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.datasets import MoleculeNet
from torch_geometric.utils import remove_self_loops
""" from adapted_shapley import GnnNetsGC2valueFunc, GnnNetsNC2valueFunc, \
    gnn_score, mc_shapley, l_shapley, mc_l_shapley, NC_mc_l_shapley, sparsity """
from torch_geometric.data import Dataset, DataLoader
import torch.nn.functional as F


def get_elements_except_one(lst, exclude_index):
    return lst[:exclude_index] + lst[exclude_index+1:]


def sparsity(coalition: list, data: Data, subgraph_building_method='zero_filling'):
    if subgraph_building_method == 'zero_filling':
        return 1.0 - len(coalition) / data.num_nodes

    elif subgraph_building_method == 'split':
        row, col = data.edge_index
        node_mask = torch.zeros(data.x.shape[0])
        node_mask[coalition] = 1.0
        edge_mask = (node_mask[row] == 1) & (node_mask[col] == 1)
        return 1.0 - (edge_mask.sum() / edge_mask.shape[0]).item()

def gnn_score(coalition: list, data: Data, value_func: str,
              subgraph_building_method='zero_filling') -> torch.Tensor:
    """ the value of subgraph with selected nodes """
    num_nodes = data.num_nodes
    subgraph_build_func = get_graph_build_func(subgraph_building_method)
    mask = torch.zeros(num_nodes).type(torch.float32).to(data.x.device)
    mask[coalition] = 1.0
    ret_x, ret_edge_index = subgraph_build_func(data.x, data.edge_index, mask)
    mask_data = Data(x=ret_x, edge_index=ret_edge_index)
    mask_data = Batch.from_data_list([mask_data])
    score = value_func(mask_data, data.outer_graphs)
    # get the score of predicted class for graph or specific node idx
    return score.item()

def graph_build_split(X, edge_index, node_mask):
    """ subgraph building through spliting the selected nodes from the original graph """
    ret_X = X
    row, col = edge_index
    edge_mask = (node_mask[row] == 1) & (node_mask[col] == 1)
    ret_edge_index = edge_index[:, edge_mask]
    return ret_X, ret_edge_index

def graph_build_zero_filling(X, edge_index, node_mask):
    """ subgraph building through masking the unselected nodes with zero features """
    ret_X = X * node_mask.unsqueeze(1)
    return ret_X, edge_index

def get_graph_build_func(build_method):
    if build_method.lower() == 'zero_filling':
        return graph_build_zero_filling
    elif build_method.lower() == 'split':
        return graph_build_split
    else:
        raise NotImplementedError

def mc_l_shapley(coalition: list, data: Data, local_radius: int,
                 value_func: str, outer_graphs, subgraph_building_method='zero_filling',
                 sample_num=1000) -> float:
    """ monte carlo sampling approximation of the l_shapley value """
    graph = to_networkx(data)
    num_nodes = graph.number_of_nodes()
    subgraph_build_func = get_graph_build_func(subgraph_building_method)

    local_region = copy.copy(coalition)
    for k in range(int(np.round(local_radius)) - 1):
        k_neiborhoood = []
        for node in local_region:
            k_neiborhoood += list(graph.neighbors(node))
        local_region += k_neiborhoood
        local_region = list(set(local_region))

    coalition_placeholder = num_nodes
    set_exclude_masks = []
    set_include_masks = []
    for example_idx in range(sample_num):
        subset_nodes_from = [node for node in local_region if node not in coalition]
        random_nodes_permutation = np.array(subset_nodes_from + [coalition_placeholder])
        random_nodes_permutation = np.random.permutation(random_nodes_permutation)
        split_idx = np.where(random_nodes_permutation == coalition_placeholder)[0][0]
        selected_nodes = random_nodes_permutation[:split_idx]
        set_exclude_mask = np.ones(num_nodes)
        set_exclude_mask[local_region] = 0.0
        set_exclude_mask[selected_nodes] = 1.0
        set_include_mask = set_exclude_mask.copy()
        set_include_mask[coalition] = 1.0

        set_exclude_masks.append(set_exclude_mask)
        set_include_masks.append(set_include_mask)

    exclude_mask = np.stack(set_exclude_masks, axis=0)
    include_mask = np.stack(set_include_masks, axis=0)
    data.outer_graphs = outer_graphs
    marginal_contributions = \
        marginal_contribution(data, exclude_mask, include_mask, value_func, subgraph_build_func)

    mc_l_shapley_value = (marginal_contributions).mean().item()
    return mc_l_shapley_value

class MarginalSubgraphDataset(Dataset):
    def __init__(self, data, exclude_mask, include_mask, subgraph_build_func):
        self.num_nodes = data.num_nodes
        self.X = data.x
        self.edge_index = data.edge_index
        self.device = self.X.device
        self.outer_graphs =data.outer_graphs
        self.label = data.y
        self.exclude_mask = torch.tensor(exclude_mask).type(torch.float32).to(self.device)
        self.include_mask = torch.tensor(include_mask).type(torch.float32).to(self.device)
        self.subgraph_build_func = subgraph_build_func
        

    def __len__(self):
        return self.exclude_mask.shape[0]

    def __getitem__(self, idx):
        exclude_graph_X, exclude_graph_edge_index = self.subgraph_build_func(self.X, self.edge_index, self.exclude_mask[idx])
        include_graph_X, include_graph_edge_index = self.subgraph_build_func(self.X, self.edge_index, self.include_mask[idx])
        exclude_data = Data(x=exclude_graph_X, edge_index=exclude_graph_edge_index)
        include_data = Data(x=include_graph_X, edge_index=include_graph_edge_index)
        outer_data = []
        for i in range(len(self.outer_graphs[0])):
            outer_data.append([Data(x=self.outer_graphs[0][i], edge_index=self.outer_graphs[1][i], edge_attr=self.outer_graphs[2][i], dict_nodes=self.outer_graphs[3][i], batch=self.outer_graphs[4][i]), self.outer_graphs[-1]])
        return exclude_data, include_data, outer_data

def marginal_contribution(data: Data, exclude_mask: np.ndarray, include_mask: np.ndarray,
                          value_func, subgraph_build_func):
    """ Calculate the marginal value for each pair. Here exclude_mask and include_mask are node mask. """
    marginal_subgraph_dataset = MarginalSubgraphDataset(data, exclude_mask, include_mask, subgraph_build_func)
    dataloader = DataLoader(marginal_subgraph_dataset, batch_size=256, shuffle=False, num_workers=0)

    marginal_contribution_list = []

    for exclude_data, include_data, outer_graphs in dataloader:
        exclude_values = value_func(exclude_data, outer_graphs)
        include_values = value_func(include_data, outer_graphs)
        margin_values = include_values - exclude_values
        marginal_contribution_list.append(margin_values)

    marginal_contributions = torch.cat(marginal_contribution_list, dim=0)
    return marginal_contributions


def GnnNetsGC2valueFunc(gnnNets, target_class):
    def value_func(batch, outer_batch):
        with torch.no_grad():
            try:
                specific_graph = outer_batch[0][1][0].item()
                number_of_graphs = len(outer_batch)+1
            except:
                specific_graph = outer_batch[-1]
                number_of_graphs = len(outer_batch[0])+1
            
            xs = []
            edge_indexes = []
            batches = []
            j = 0
            for i in range(number_of_graphs):
                if i == specific_graph:
                    xs.append(batch.x)
                    edge_indexes.append(batch.edge_index)
                    batches.append(batch.batch)
                else:
                    try:
                        xs.append(outer_batch[0][0].x)
                        edge_indexes.append(outer_batch[0][0].edge_index)
                        batches.append(outer_batch[0][0].batch)
                        outer_batch = outer_batch[1:]
                    except:
                        xs.append(outer_batch[0][j])
                        edge_indexes.append(outer_batch[1][j])
                        batches.append(outer_batch[4][j])
                        j += 1

            inputs = []
            for i in range(len(xs)):
                inputs.append(xs[i])
                inputs.append(edge_indexes[i])
                inputs.append(batches[i])
            
            logits = gnnNets(*inputs)
            probs = F.softmax(logits, dim=-1)
            score = probs[:, target_class]
        return score
    return value_func


def GnnNetsNC2valueFunc(gnnNets_NC, node_idx, target_class):
    def value_func(data):
        with torch.no_grad():
            logits = gnnNets_NC(data=data)
            probs = F.softmax(logits, dim=-1)
            # select the corresponding node prob through the node idx on all the sampling graphs
            batch_size = data.batch.max() + 1
            probs = probs.reshape(batch_size, -1, probs.shape[-1])
            score = probs[:, node_idx, target_class]
            return score
    return value_func


# Don't modify
def find_closest_node_result(results, max_nodes):
    """ return the highest reward tree_node with its subgraph is smaller than max_nodes """
    results = sorted(results, key=lambda x: len(x.coalition))

    result_node = results[0]
    for result_idx in range(len(results)):
        x = results[result_idx]
        if len(x.coalition) <= max_nodes and x.P > result_node.P:
            result_node = x
    return result_node


# Don't modify
def reward_func(reward_method, value_func, node_idx=None,
                local_radius=[4], sample_num=[100],
                subgraph_building_method='zero_filling'):
    if reward_method.lower() == 'gnn_score':
        return partial(gnn_score,
                       value_func=value_func,
                       subgraph_building_method=subgraph_building_method)

    elif reward_method.lower() == 'mc_shapley':
        return partial(mc_shapley,
                       value_func=value_func,
                       subgraph_building_method=subgraph_building_method,
                       sample_num=sample_num)

    elif reward_method.lower() == 'l_shapley':
        return partial(l_shapley,
                       local_radius=local_radius,
                       value_func=value_func,
                       subgraph_building_method=subgraph_building_method)

    elif reward_method.lower() == 'mc_l_shapley':                                                       # aquí
        return partial(mc_l_shapley,
                       local_radius=local_radius,
                       value_func=value_func,
                       subgraph_building_method=subgraph_building_method,
                       sample_num=sample_num)

    elif reward_method.lower() == 'nc_mc_l_shapley':
        assert node_idx is not None, " Wrong node idx input "
        return partial(NC_mc_l_shapley,
                       node_idx=node_idx,
                       local_radius=local_radius,
                       value_func=value_func,
                       subgraph_building_method=subgraph_building_method,
                       sample_num=sample_num)

    else:
        raise NotImplementedError

# devuelve el subconjunto de nodos centrales de cada subgrafo, así como la máscara de estos nodos y aristas para el grafo original 
# y también los nuevos índices de estos elementos si es que relabel_nodes = True
def k_hop_subgraph_with_default_whole_graph(
        edge_index, node_idx=None, num_hops=3, relabel_nodes=False,
        num_nodes=None, flow='source_to_target'):
    r"""Computes the :math:`k`-hop subgraph of :obj:`edge_index` around node
    :attr:`node_idx`.
    It returns (1) the nodes involved in the subgraph, (2) the filtered
    :obj:`edge_index` connectivity, (3) the mapping from node indices in
    :obj:`node_idx` to their new location, and (4) the edge mask indicating
    which edges were preserved.
    Args:
        node_idx (int, list, tuple or :obj:`torch.Tensor`): The central
            node(s).
        num_hops: (int): The number of hops :math:`k`.
        edge_index (LongTensor): The edge indices.
        relabel_nodes (bool, optional): If set to :obj:`True`, the resulting
            :obj:`edge_index` will be relabeled to hold consecutive indices
            starting from zero. (default: :obj:`False`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)
        flow (string, optional): The flow direction of :math:`k`-hop
            aggregation (:obj:`"source_to_target"` or
            :obj:`"target_to_source"`). (default: :obj:`"source_to_target"`)
    :rtype: (:class:`LongTensor`, :class:`LongTensor`, :class:`LongTensor`,
             :class:`BoolTensor`)
    """

    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    assert flow in ['source_to_target', 'target_to_source']
    if flow == 'target_to_source':
        row, col = edge_index
    else:
        col, row = edge_index  # edge_index 0 to 1, col: source, row: target

    node_mask = row.new_empty(num_nodes, dtype=torch.bool)
    edge_mask = row.new_empty(row.size(0), dtype=torch.bool)

    inv = None

    # Genera un subconjunto de nodos para tomarlos como nodos centrales, empieza por el nodo 0 
    # y añade nodos hasta que el subconjunto permanece constante
    if node_idx is None:
        subsets = torch.tensor([0])
        cur_subsets = subsets
        while 1:
            node_mask.fill_(False)
            node_mask[subsets] = True
            torch.index_select(node_mask, 0, row, out=edge_mask)
            subsets = torch.cat([subsets, col[edge_mask]]).unique()
            if not cur_subsets.equal(subsets):
                cur_subsets = subsets
            else:
                subset = subsets
                break
    
    # Parte del nodo que se ha dado como argumento para obtener un subconjunto de nodos centrales
    else:
        if isinstance(node_idx, (int, list, tuple)):
            node_idx = torch.tensor([node_idx], device=row.device, dtype=torch.int64).flatten()
        elif isinstance(node_idx, torch.Tensor) and len(node_idx.shape) == 0:
            node_idx = torch.tensor([node_idx])
        else:
            node_idx = node_idx.to(row.device)

        subsets = [node_idx]
        for _ in range(num_hops):
            node_mask.fill_(False)
            node_mask[subsets[-1]] = True
            torch.index_select(node_mask, 0, row, out=edge_mask)
            subsets.append(col[edge_mask])
        subset, inv = torch.cat(subsets).unique(return_inverse=True)
        inv = inv[:node_idx.numel()]

    node_mask.fill_(False)
    node_mask[subset] = True
    
    # bitwise AND, if 1 in node_mask[row] and node_mask[col] then 1
    edge_mask = node_mask[row] & node_mask[col]

    # only select the columns where edge_mask == True
    edge_index = edge_index[:, edge_mask]

    if relabel_nodes:
        node_idx = row.new_full((num_nodes,), -1)
        node_idx[subset] = torch.arange(subset.size(0), device=row.device)
        edge_index = node_idx[edge_index]

    return subset, edge_index, inv, edge_mask  # subset: key new node idx; value original node idx

# Here we get the score of the children that later will be used to obtain the Shapley values
def compute_scores(score_func, children, outer_graphs):
    results = []
    for child in children:
        if child.P == 0:
            score = score_func(child.coalition, child.data, outer_graphs=outer_graphs)
        else:
            score = child.P
        results.append(score)
    return results


class PlotUtils(object):
    def __init__(self, dataset_name, is_show=True):
        self.dataset_name = dataset_name
        self.is_show = is_show

    def plot(self, graph, nodelist, figname, title_sentence=None, **kwargs):
        """ plot function for different dataset """
        if self.dataset_name.lower() in ['ba_2motifs', 'ba_lrp']:
            self.plot_ba2motifs(graph, nodelist, title_sentence=title_sentence, figname=figname)
        elif self.dataset_name.lower() in ['mutag'] + list(MoleculeNet.names.keys()):
            x = kwargs.get('x')
            self.plot_molecule(graph, nodelist, x, title_sentence=title_sentence, figname=figname)
        elif self.dataset_name.lower() in ['ba_shapes', 'ba_community', 'tree_grid', 'tree_cycle']:
            y = kwargs.get('y')
            node_idx = kwargs.get('node_idx')
            self.plot_bashapes(graph, nodelist, y, node_idx, title_sentence=title_sentence, figname=figname)
        elif self.dataset_name.lower() in ['graph_sst2', 'graph_sst5', 'twitter']:
            words = kwargs.get('words')
            self.plot_sentence(graph, nodelist, words=words, title_sentence=title_sentence, figname=figname)
        else:
            raise NotImplementedError

    def plot_subgraph(self,
                      graph,
                      nodelist,
                      colors: Union[None, str, List[str]] = '#FFA500',
                      labels=None,
                      edge_color='gray',
                      edgelist=None,
                      subgraph_edge_color='black',
                      title_sentence=None,
                      figname=None):

        if edgelist is None:
            edgelist = [(n_frm, n_to) for (n_frm, n_to) in graph.edges()
                        if n_frm in nodelist and n_to in nodelist]
        pos = nx.kamada_kawai_layout(graph)
        pos_nodelist = {k: v for k, v in pos.items() if k in nodelist}

        nx.draw_networkx_nodes(graph, pos,
                               nodelist=list(graph.nodes()),
                               node_color=colors,
                               node_size=300)

        nx.draw_networkx_edges(graph, pos, width=3, edge_color=edge_color, arrows=False)

        nx.draw_networkx_edges(graph, pos=pos_nodelist,
                               edgelist=edgelist, width=6,
                               edge_color=subgraph_edge_color,
                               arrows=False)

        if labels is not None:
            nx.draw_networkx_labels(graph, pos, labels)

        plt.axis('off')
        if title_sentence is not None:
            plt.title('\n'.join(wrap(title_sentence, width=60)))
        if figname is not None:
            plt.savefig(figname)
        if self.is_show:
            plt.show()
        if figname is not None:
            plt.close()

    def plot_subgraph_with_nodes(self,
                                 graph,
                                 nodelist,
                                 node_idx,
                                 colors='#FFA500',
                                 labels=None,
                                 edge_color='gray',
                                 edgelist=None,
                                 subgraph_edge_color='black',
                                 title_sentence=None,
                                 figname=None):
        node_idx = int(node_idx)
        if edgelist is None:
            edgelist = [(n_frm, n_to) for (n_frm, n_to) in graph.edges()
                        if n_frm in nodelist and n_to in nodelist]

        pos = nx.kamada_kawai_layout(graph)  # calculate according to graph.nodes()
        pos_nodelist = {k: v for k, v in pos.items() if k in nodelist}

        nx.draw_networkx_nodes(graph, pos,
                               nodelist=list(graph.nodes()),
                               node_color=colors,
                               node_size=300)
        if isinstance(colors, list):
            list_indices = int(np.where(np.array(graph.nodes()) == node_idx)[0])
            node_idx_color = colors[list_indices]
        else:
            node_idx_color = colors

        nx.draw_networkx_nodes(graph, pos=pos,
                               nodelist=[node_idx],
                               node_color=node_idx_color,
                               node_size=600)

        nx.draw_networkx_edges(graph, pos, width=3, edge_color=edge_color, arrows=False)

        nx.draw_networkx_edges(graph, pos=pos_nodelist,
                               edgelist=edgelist, width=3,
                               edge_color=subgraph_edge_color,
                               arrows=False)

        if labels is not None:
            nx.draw_networkx_labels(graph, pos, labels)

        plt.axis('off')
        if title_sentence is not None:
            plt.title('\n'.join(wrap(title_sentence, width=60)))

        if figname is not None:
            plt.savefig(figname)
        if self.is_show:
            plt.show()
        if figname is not None:
            plt.close()

    def plot_sentence(self, graph, nodelist, words, edgelist=None, title_sentence=None, figname=None):
        pos = nx.kamada_kawai_layout(graph)
        words_dict = {i: words[i] for i in graph.nodes}
        if nodelist is not None:
            pos_coalition = {k: v for k, v in pos.items() if k in nodelist}
            nx.draw_networkx_nodes(graph, pos_coalition,
                                   nodelist=nodelist,
                                   node_color='yellow',
                                   node_shape='o',
                                   node_size=500)
            if edgelist is None:
                edgelist = [(n_frm, n_to) for (n_frm, n_to) in graph.edges()
                            if n_frm in nodelist and n_to in nodelist]
                nx.draw_networkx_edges(graph, pos=pos_coalition, edgelist=edgelist, width=5, edge_color='yellow', arrows=True)

        nx.draw_networkx_nodes(graph, pos, nodelist=list(graph.nodes()), node_size=300)

        nx.draw_networkx_edges(graph, pos, width=2, edge_color='grey')
        nx.draw_networkx_labels(graph, pos, words_dict)

        plt.axis('off')
        plt.title('\n'.join(wrap(' '.join(words), width=50)))
        if title_sentence is not None:
            string = '\n'.join(wrap(' '.join(words), width=50))
            string += '\n'.join(wrap(title_sentence, width=60))
            plt.title(string)
        if figname is not None:
            plt.savefig(figname)
        if self.is_show:
            plt.show()
        if figname is not None:
            plt.close()

    def plot_ba2motifs(self,
                       graph,
                       nodelist,
                       edgelist=None,
                       title_sentence=None,
                       figname=None):
        return self.plot_subgraph(graph, nodelist,
                                  edgelist=edgelist,
                                  title_sentence=title_sentence,
                                  figname=figname)

    def plot_molecule(self,
                      graph,
                      nodelist,
                      x,
                      edgelist=None,
                      title_sentence=None,
                      figname=None):
        # collect the text information and node color
        if self.dataset_name == 'mutag':
            node_dict = {0: 'C', 1: 'N', 2: 'O', 3: 'F', 4: 'I', 5: 'Cl', 6: 'Br'}
            node_idxs = {k: int(v) for k, v in enumerate(np.where(x.cpu().numpy() == 1)[1])}
            node_labels = {k: node_dict[v] for k, v in node_idxs.items()}
            node_color = ['#E49D1C', '#4970C6', '#FF5357', '#29A329', 'brown', 'darkslategray', '#F0EA00']
            colors = [node_color[v % len(node_color)] for k, v in node_idxs.items()]

        elif self.dataset_name in MoleculeNet.names.keys():
            element_idxs = {k: int(v) for k, v in enumerate(x[:, 0])}
            node_idxs = element_idxs
            node_labels = {k: Chem.PeriodicTable.GetElementSymbol(Chem.GetPeriodicTable(), int(v))
                           for k, v in element_idxs.items()}
            node_color = ['#29A329', 'lime', '#F0EA00',  'maroon', 'brown', '#E49D1C', '#4970C6', '#FF5357']
            colors = [node_color[(v - 1) % len(node_color)] for k, v in node_idxs.items()]
        else:
            raise NotImplementedError

        self.plot_subgraph(graph, nodelist,
                           colors=colors,
                           labels=node_labels,
                           edgelist=edgelist,
                           edge_color='gray',
                           subgraph_edge_color='black',
                           title_sentence=title_sentence,
                           figname=figname)

    def plot_bashapes(self,
                      graph,
                      nodelist,
                      y,
                      node_idx,
                      edgelist=None,
                      title_sentence=None,
                      figname=None):
        node_idxs = {k: int(v) for k, v in enumerate(y.reshape(-1).tolist())}
        node_color = ['#FFA500', '#4970C6', '#FE0000', 'green']
        colors = [node_color[v % len(node_color)] for k, v in node_idxs.items()]
        self.plot_subgraph_with_nodes(graph,
                                      nodelist,
                                      node_idx,
                                      colors,
                                      edgelist=edgelist,
                                      title_sentence=title_sentence,
                                      figname=figname,
                                      subgraph_edge_color='black')


class MCTSNode(object):
    def __init__(self, coalition: list = None, data: Data = None, ori_graph: nx.Graph = None,
                 c_puct: float = 10.0, W: float = 0, N: int = 0, P: float = 0,
                 load_dict: Optional[Dict] = None, device='cpu'):
        self.data = data
        self.coalition = coalition
        self.ori_graph = ori_graph
        self.device = device
        self.c_puct = c_puct
        self.children = []
        self.W = W  # sum of node value
        self.N = N  # times of arrival
        self.P = P  # property score (reward)
        if load_dict is not None:
            self.load_info(load_dict)

    def Q(self):
        return self.W / self.N if self.N > 0 else 0

    def U(self, n):
        return self.c_puct * self.P * math.sqrt(n) / (1 + self.N)

    @property
    def info(self):
        info_dict = {
            'data': self.data.to('cpu'),
            'coalition': self.coalition,
            'ori_graph': self.ori_graph,
            'W': self.W,
            'N': self.N,
            'P': self.P
        }
        return info_dict

    def load_info(self, info_dict):
        self.W = info_dict['W']
        self.N = info_dict['N']
        self.P = info_dict['P']
        self.coalition = info_dict['coalition']
        self.ori_graph = info_dict['ori_graph']
        self.data = info_dict['data'].to(self.device)
        self.children = []
        return self


class MCTS(object):
    r"""
    Monte Carlo Tree Search Method.
    
    Args:
        X (:obj:`torch.Tensor`): Input node features
        edge_index (:obj:`torch.Tensor`): The edge indices.
        num_hops (:obj:`int`): The number of hops :math:`k`.
        n_rollout (:obj:`int`): The number of sequence to build the monte carlo tree.
        min_atoms (:obj:`int`): The number of atoms for the subgraph in the monte carlo tree leaf node.
        c_puct (:obj:`float`): The hyper-parameter to encourage exploration while searching.
        expand_atoms (:obj:`int`): The number of children to expand.
        high2low (:obj:`bool`): Whether to expand children tree node from high degree nodes to low degree nodes.
        node_idx (:obj:`int`): The target node index to extract the neighborhood.
        score_func (:obj:`Callable`): The reward function for tree node, such as mc_shapely and mc_l_shapely.
    """
    def __init__(self, X: torch.Tensor, edge_index: torch.Tensor, edge_attr:torch.Tensor, dict_node:dict, num_hops,
                 n_rollout: list[int] = [10], min_atoms: list[int] = [3], c_puct: list[float] = [10.0],
                 expand_atoms: list[int] = [14], high2low: bool = False,
                 node_idx= None, score_func = None, device='cpu'):

        self.X = X
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.dict_node = dict_node
        self.device = device
        self.num_hops = num_hops
        self.data = Data(x=self.X, edge_index=self.edge_index, edge_attr=self.edge_attr, dict_node=self.dict_node)
        graph_data = Data(x=self.X, edge_index=remove_self_loops(self.edge_index)[0], edge_attr=self.edge_attr)
        self.graph = to_networkx(graph_data, to_undirected=True)
        self.data = Batch.from_data_list([self.data])
        self.num_nodes = self.graph.number_of_nodes()
        self.score_func = score_func
        self.n_rollout = n_rollout
        self.min_atoms = min_atoms
        self.c_puct = c_puct
        self.expand_atoms = expand_atoms
        self.high2low = high2low
        self.new_node_idx = None

        # extract the sub-graph and change the node indices.
        if node_idx is not None:                                                                        # aquí no
            if isinstance(node_idx, Tensor):
                node_idx = node_idx.item()

            self.ori_node_idx = node_idx
            self.ori_graph = copy.copy(self.graph)
            x, edge_index, subset, edge_mask, kwargs = \
                self.__subgraph__(node_idx, self.X, self.edge_index, self.num_hops)
            self.data = Batch.from_data_list([Data(x=x, edge_index=edge_index)])
            self.graph = self.ori_graph.subgraph(subset.tolist())
            mapping = {int(v): k for k, v in enumerate(subset)}
            self.graph = nx.relabel_nodes(self.graph, mapping)
            self.new_node_idx = torch.where(subset == self.ori_node_idx)[0].item()
            self.num_nodes = self.graph.number_of_nodes()
            self.subset = subset

        self.root_coalition = sorted([node for node in range(self.num_nodes)])
        self.MCTSNodeClass = partial(MCTSNode, data=self.data, ori_graph=self.graph,
                                     c_puct=self.c_puct, device=self.device)
        self.root = self.MCTSNodeClass(self.root_coalition)
        self.state_map = {str(self.root.coalition): self.root}

    def set_score_func(self, score_func):
        self.score_func = score_func

    # Get the specific subgraph using k_hop_subgraph_with_default_whole_graph
    @staticmethod
    def __subgraph__(node_idx, x, edge_index, num_hops, **kwargs):
        num_nodes, num_edges = x.size(0), edge_index.size(1)
        subset, edge_index, _, edge_mask = k_hop_subgraph_with_default_whole_graph(
            edge_index, node_idx, num_hops, relabel_nodes=True, num_nodes=num_nodes)

        # Get the elements of x specified in the subset, as well as all kwargs in the subset index corresponding to nodes and edges
        x = x[subset]
        for key, item in kwargs.items():
            if torch.is_tensor(item) and item.size(0) == num_nodes:
                item = item[subset]
            elif torch.is_tensor(item) and item.size(0) == num_edges:
                item = item[edge_mask]
            kwargs[key] = item

        return x, edge_index, subset, edge_mask, kwargs

    def mcts_rollout(self, tree_node, outer_graphs):
        cur_graph_coalition = tree_node.coalition
        if len(cur_graph_coalition) <= self.min_atoms:
            return tree_node.P

        # Expand if this node has never been visited
        if len(tree_node.children) == 0:
            node_degree_list = list(self.graph.subgraph(cur_graph_coalition).degree)
            node_degree_list = sorted(node_degree_list, key=lambda x: x[1], reverse=self.high2low)
            all_nodes = [x[0] for x in node_degree_list]

            if self.new_node_idx:
                expand_nodes = [node for node in all_nodes if node != self.new_node_idx]
            else:
                expand_nodes = all_nodes

            if len(all_nodes) > self.expand_atoms:
                expand_nodes = expand_nodes[:self.expand_atoms]

            for each_node in expand_nodes:
                # for each node, pruning it and get the remaining sub-graph
                # here we check the resulting sub-graphs and only keep the largest one
                subgraph_coalition = [node for node in all_nodes if node != each_node]

                subgraphs = [self.graph.subgraph(c)
                             for c in nx.connected_components(self.graph.subgraph(subgraph_coalition))]

                if self.new_node_idx:
                    for sub in subgraphs:
                        if self.new_node_idx in list(sub.nodes()):
                            main_sub = sub
                else:
                    main_sub = subgraphs[0]

                    for sub in subgraphs:
                        if sub.number_of_nodes() > main_sub.number_of_nodes():
                            main_sub = sub

                new_graph_coalition = sorted(list(main_sub.nodes()))

                # check the state map and merge the same sub-graph
                find_same = False
                for old_graph_node in self.state_map.values():
                    if Counter(old_graph_node.coalition) == Counter(new_graph_coalition):
                        new_node = old_graph_node
                        find_same = True

                if not find_same:
                    new_node = self.MCTSNodeClass(new_graph_coalition)
                    self.state_map[str(new_graph_coalition)] = new_node

                find_same_child = False
                for cur_child in tree_node.children:
                    if Counter(cur_child.coalition) == Counter(new_graph_coalition):
                        find_same_child = True

                if not find_same_child:
                    tree_node.children.append(new_node)

            scores = compute_scores(self.score_func, tree_node.children, outer_graphs=outer_graphs)
            for child, score in zip(tree_node.children, scores):
                child.P = score

        sum_count = sum([c.N for c in tree_node.children])
        selected_node = max(tree_node.children, key=lambda x: x.Q() + x.U(sum_count))
        v = self.mcts_rollout(selected_node, outer_graphs)
        selected_node.W += v
        selected_node.N += 1
        return v

    def mcts(self, verbose=True, outer_graphs=None):
        if verbose:
            print(f"The nodes in graph is {self.graph.number_of_nodes()}")
        for rollout_idx in range(self.n_rollout):
            self.mcts_rollout(self.root, outer_graphs)
            if verbose:
                print(f"At the {rollout_idx} rollout, {len(self.state_map)} states that have been explored.")

        explanations = [node for _, node in self.state_map.items()]
        explanations = sorted(explanations, key=lambda x: x.P, reverse=True)
        return explanations


class SubgraphX(object):
    r"""
    The implementation of paper
    `On Explainability of Graph Neural Networks via Subgraph Explorations <https://arxiv.org/abs/2102.05152>`_.
    
    Args:
        model (:obj:`torch.nn.Module`): The target model prepared to explain
        num_classes(:obj:`int`): Number of classes for the datasets
        num_hops(:obj:`int`, :obj:`None`): The number of hops to extract neighborhood of target node
          (default: :obj:`None`)
        explain_graph(:obj:`bool`): Whether to explain graph classification model (default: :obj:`True`)
        rollout(:obj:`int`): Number of iteration to get the prediction
        min_atoms(:obj:`int`): Number of atoms of the leaf node in search tree
        c_puct(:obj:`float`): The hyperparameter which encourages the exploration
        expand_atoms(:obj:`int`): The number of atoms to expand
          when extend the child nodes in the search tree
        high2low(:obj:`bool`): Whether to expand children nodes from high degree to low degree when
          extend the child nodes in the search tree (default: :obj:`False`)
        local_radius(:obj:`int`): Number of local radius to calculate :obj:`l_shapley`, :obj:`mc_l_shapley`
        sample_num(:obj:`int`): Sampling time of monte carlo sampling approximation for
          :obj:`mc_shapley`, :obj:`mc_l_shapley` (default: :obj:`mc_l_shapley`)
        reward_method(:obj:`str`): The command string to select the
        subgraph_building_method(:obj:`str`): The command string for different subgraph building method,
          such as :obj:`zero_filling`, :obj:`split` (default: :obj:`zero_filling`)
        save_dir(:obj:`str`, :obj:`None`): Root directory to save the explanation results (default: :obj:`None`)
        filename(:obj:`str`): The filename of results
        vis(:obj:`bool`): Whether to show the visualization (default: :obj:`True`)
    Example:
        >>> # For graph classification task
        >>> subgraphx = SubgraphX(model=model, num_classes=2)
        >>> _, explanation_results, related_preds = subgraphx(x, edge_index)
    """
    def __init__(self, model, num_classes: int, device, num_hops: list[int] = [1], verbose: bool = False,
                 explain_graph: bool = True, rollout: list[int] = [20], min_atoms: list[int] = [5], c_puct: list[float] = [10.0],
                 expand_atoms: list[int]=[5], high2low=False, local_radius: list[int]= [4], sample_num: list[int]=[10], reward_method='mc_l_shapley',
                 subgraph_building_method='zero_filling', save_dir: Optional[str] = None,
                 filename: str = 'example', vis: bool = True):

        self.model = model
        self.model.eval()
        self.device = device
        self.model.to(self.device)
        self.num_classes = num_classes
        self.num_hops = self.update_num_hops(num_hops)
        self.explain_graph = explain_graph
        self.verbose = verbose

        # mcts hyper-parameters
        self.rollout = rollout
        self.min_atoms = min_atoms
        self.c_puct = c_puct
        self.expand_atoms = expand_atoms
        self.high2low = high2low

        # reward function hyper-parameters
        self.local_radius = local_radius
        self.sample_num = sample_num
        self.reward_method = reward_method
        self.subgraph_building_method = subgraph_building_method

        # saving and visualization
        self.vis = vis
        self.save_dir = save_dir
        self.filename = filename
        self.save = True if self.save_dir is not None else False

    def update_num_hops(self, num_hops):
        if num_hops is not None:
            return num_hops

        k = 0
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                k += 1
        return k

    def get_reward_func(self, value_func, local_radius, sample_num, node_idx=None):
        if self.explain_graph:                                                                              # Entramos aquí
            node_idx = None
        else:
            assert node_idx is not None
        return reward_func(reward_method=self.reward_method,
                           value_func=value_func,
                           node_idx=node_idx,
                           local_radius=local_radius,
                           sample_num=sample_num,
                           subgraph_building_method=self.subgraph_building_method)

    def get_mcts_class(self, x_list, edge_index_list, edge_attr_list, dict_node_list, node_idx_list, score_funcs):
        if self.explain_graph:
            node_idx_list = [None]*len(x_list)
        else:
            assert not None in node_idx_list
        assert len(x_list) == len(edge_index_list)

        mcts_list = []

        # Ensure self.num_hops is an iterable (wrap it in a list if it's not)
        if isinstance(self.num_hops, int):
            num_hops_iter = [self.num_hops] * len(x_list)
        else:
            num_hops_iter = self.num_hops

        for x, edge_index, edge_attr, dict_node, node_idx, num_hops, n_rollout, min_atoms, c_puct, expand_atoms, score_func in zip(x_list, edge_index_list, edge_attr_list, dict_node_list, node_idx_list, num_hops_iter, self.rollout, self.min_atoms, self.c_puct, self.expand_atoms, score_funcs):
            mcts_list.append(MCTS(x, edge_index,
                                  edge_attr, dict_node,
                                    node_idx=node_idx,
                                  device=self.device,
                                  score_func=score_func,
                                  num_hops=num_hops,
                                  n_rollout=n_rollout,
                                  min_atoms=min_atoms,
                                  c_puct=c_puct,
                                  expand_atoms=expand_atoms,
                                  high2low=self.high2low))
        
        return mcts_list

    def visualization(self, results: list,
                      max_nodes: int, plot_utils: PlotUtils, words: Optional[list] = None,
                      y: Optional[Tensor] = None, title_sentence: Optional[str] = None,
                      vis_name: Optional[str] = None):
        if self.save:
            if vis_name is None:
                vis_name = f"{self.filename}.png"
        else:
            vis_name = None
        tree_node_x = find_closest_node_result(results, max_nodes=max_nodes)
        if self.explain_graph:
            if words is not None:
                plot_utils.plot(tree_node_x.ori_graph,
                                tree_node_x.coalition,
                                words=words,
                                title_sentence=title_sentence,
                                figname=vis_name)
            else:
                plot_utils.plot(tree_node_x.ori_graph,
                                tree_node_x.coalition,
                                x=tree_node_x.data.x,
                                title_sentence=title_sentence,
                                figname=vis_name)
        else:
            subset = self.mcts_state_map.subset
            subgraph_y = y[subset].to('cpu')
            subgraph_y = torch.tensor([subgraph_y[node].item()
                                       for node in tree_node_x.ori_graph.nodes()])
            plot_utils.plot(tree_node_x.ori_graph,
                            tree_node_x.coalition,
                            node_idx=self.mcts_state_map.new_node_idx,
                            title_sentence=title_sentence,
                            y=subgraph_y,
                            figname=vis_name)

    def read_from_MCTSInfo_list(self, MCTSInfo_list):
        if isinstance(MCTSInfo_list[0], dict):
            ret_list = [MCTSNode(device=self.device).load_info(node_info) for node_info in MCTSInfo_list]
        elif isinstance(MCTSInfo_list[0][0], dict):
            ret_list = []
            for single_label_MCTSInfo_list in MCTSInfo_list:
                single_label_ret_list = [MCTSNode(device=self.device).load_info(node_info) for node_info in single_label_MCTSInfo_list]
                ret_list.append(single_label_ret_list)
        return ret_list

    def write_from_MCTSNode_list(self, MCTSNode_list):
        if isinstance(MCTSNode_list[0], MCTSNode):
            ret_list = [node.info for node in MCTSNode_list]
        elif isinstance(MCTSNode_list[0][0], MCTSNode):
            ret_list = []
            for single_label_MCTSNode_list in MCTSNode_list:
                single_label_ret_list = [node.info for node in single_label_MCTSNode_list]
                ret_list.append(single_label_ret_list)
        return ret_list

    def explain(self, x_list: list[Tensor], edge_index_list: list[Tensor], 
                batch_list: list[Tensor], edge_attr_list: list[Tensor],  
                dict_node_list: list[dict], label: int,
                max_nodes: list[int] = [5], node_idx_list = None,
                saved_MCTSInfo_list: Optional[List[List]] = None):
        #probs = self.model(x, edge_index).squeeze().softmax(dim=-1)

        if node_idx_list is None:
            node_idx_list = [None]*len(x_list)

        x1 = x_list[0]
        x2 = x_list[1]
        edge_index1 = edge_index_list[0]
        edge_index2 = edge_index_list[1]
        batch1 = batch_list[0]
        batch2 = batch_list[1]

        probs = self.model(x1=x1, x2=x2, edge_index1=edge_index1, edge_index2=edge_index2, batch1=batch1, batch2=batch2).squeeze().softmax(dim=-1)
        if self.explain_graph:                                                                                                  # aquí
            if saved_MCTSInfo_list:
                results_list = self.read_from_MCTSInfo_list(saved_MCTSInfo_list)
            # ENTRAMOS AQUI
            if not saved_MCTSInfo_list:
                value_func = GnnNetsGC2valueFunc(self.model, target_class=label)
                payoff_funcs = [self.get_reward_func(value_func, local_radius=self.local_radius[i], sample_num=self.sample_num[i]) for i in range(len(self.local_radius))]
                self.mcts_state_map = self.get_mcts_class(x_list=x_list, edge_index_list=edge_index_list, 
                                                          edge_attr_list=edge_attr_list, dict_node_list=dict_node_list, 
                                                          node_idx_list=node_idx_list, score_funcs=payoff_funcs)
                results_list = []
                for i in range(len(self.mcts_state_map)):
                    outer_graphs = [get_elements_except_one(x_list, i), get_elements_except_one(edge_index_list, i), 
                                    get_elements_except_one(edge_attr_list, i), get_elements_except_one(dict_node_list, i), 
                                    get_elements_except_one(batch_list, i), i]
                    
                    results_list.append(self.mcts_state_map[i].mcts(verbose=self.verbose, outer_graphs=outer_graphs))
                

            # l sharply score
            value_func = GnnNetsGC2valueFunc(self.model, target_class=label)
            tree_node_x_list = []
            for results in range(len(results_list)):
                tree_node_x = find_closest_node_result(results_list[i], max_nodes=max_nodes[i])
                tree_node_x_list.append(tree_node_x)

        else:                                                                                       # aquí no
            if saved_MCTSInfo_list:
                results_list = self.read_from_MCTSInfo_list(saved_MCTSInfo_list)

            self.mcts_state_map = self.get_mcts_class(x_list=x_list, edge_index_list=edge_index_list, 
                                                      node_idx_list=node_idx_list, edge_attr_list=edge_attr_list,
                                                      dict_node_list=dict_node_list ,score_funcs=None)
            for i in range(len(self.mcts_state_map)):
                self.new_node_idx = self.mcts_state_map[i].new_node_idx
            
            
            if not saved_MCTSInfo_list:
                for i in range(len(self.mcts_state_map)):
                    # mcts will extract the subgraph and relabel the nodes
                    value_func = GnnNetsNC2valueFunc(self.model,node_idx=self.mcts_state_map[i].new_node_idx,
                                             target_class=label)
                    payoff_func = self.get_reward_func(value_func,
                                                   node_idx=self.mcts_state_map[i].new_node_idx,
                                                   local_radius=self.local_radius[i],
                                                   sample_num=self.sample_num[i])
                    self.mcts_state_map[i].set_score_func(payoff_func)
                    results = self.mcts_state_map[i].mcts(verbose=self.verbose)

                    tree_node_x = find_closest_node_result(results, max_nodes=max_nodes[i])
                
        related_pred_list = [None]*len(self.mcts_state_map)
        for i in range(len(self.mcts_state_map)):
            # keep the important structure
            masked_node_list = [node for node in range(tree_node_x_list[i].data.x.shape[0])
                                if node in tree_node_x_list[i].coalition]

            # remove the important structure, for node_classification,
            # remain the node_idx when remove the important structure
            maskout_node_list = [node for node in range(tree_node_x_list[i].data.x.shape[0])
                                if node not in tree_node_x_list[i].coalition]
            if not self.explain_graph:
                maskout_node_list += [self.new_node_idx]

            masked_score = gnn_score(masked_node_list,
                                    tree_node_x_list[i].data,
                                    value_func=value_func,
                                    subgraph_building_method=self.subgraph_building_method)

            maskout_score = gnn_score(maskout_node_list,
                                    tree_node_x_list[i].data,
                                    value_func=value_func,
                                    subgraph_building_method=self.subgraph_building_method)

            sparsity_score = sparsity(masked_node_list, tree_node_x.data,
                                    subgraph_building_method=self.subgraph_building_method)
            results_list[i] = self.write_from_MCTSNode_list(results_list[i])
            related_pred = {'masked': masked_score,
                            'maskout': maskout_score,
                            'origin': probs[node_idx_list[i], label].item(),
                            'sparsity': sparsity_score}
            related_pred_list[i] = related_pred

        return results_list, related_pred_list

    def __call__(self, x_list: list[Tensor], edge_index_list: list[Tensor], batch_list, **kwargs)\
            -> Tuple[None, List, List[Dict]]:
        r""" explain the GNN behavior for the graph using SubgraphX method
        Args:
            x (:obj:`torch.Tensor`): Node feature matrix with shape
              :obj:`[num_nodes, dim_node_feature]`
            edge_index (:obj:`torch.Tensor`): Graph connectivity in COO format
              with shape :obj:`[2, num_edges]`
            kwargs(:obj:`Dict`):
              The additional parameters
                - node_idx (:obj:`int`, :obj:`None`): The target node index when explain node classification task
                - max_nodes (:obj:`int`, :obj:`None`): The number of nodes in the final explanation results
        :rtype: (:obj:`None`, List[torch.Tensor], List[Dict])
        """
        node_idx_list = kwargs.get('node_idx_list')
        max_nodes = kwargs.get('max_nodes')   # default max subgraph size

        # collect all the class index
        labels = tuple(label for label in range(self.num_classes))
        ex_labels = tuple(torch.tensor([label]).to(self.device) for label in labels)

        related_preds = []
        explanation_results = []
        saved_results = None
        if self.save:
            if os.path.isfile(os.path.join(self.save_dir, f"{self.filename}.pt")):
                saved_results = torch.load(os.path.join(self.save_dir, f"{self.filename}.pt"))

        for label_idx, label in enumerate(ex_labels):
            results, related_pred = self.explain(x_list, edge_index_list,
                                                 batch_list, label=label,
                                                 max_nodes=max_nodes,
                                                 node_idx_list=node_idx_list,
                                                 saved_MCTSInfo_list=saved_results)
            related_preds.append(related_pred)
            explanation_results.append(results)

        if self.save:
            torch.save(explanation_results,
                       os.path.join(self.save_dir, f"{self.filename}.pt"))

        return None, explanation_results, related_preds