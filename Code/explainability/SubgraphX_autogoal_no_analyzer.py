import torch
import numpy as np
import os
from tqdm import tqdm
from architecture_GNNs import *
from dataloader import *
from adapted_subgraphX import *
from torch_geometric.nn import MessagePassing
from torch.utils.data import Subset
import pickle as pkl
from analysis_general import GraphAnalyzer_general
from datetime import datetime

state_dict = torch.load('/usrvol/results/Trained_GCNCONV2/model14.pt')


"""HIPERPARÁMETROS A OPTIMIZAR"""

num_hops = 1                                                #int
rollout = 50                                                #int
min_atoms = 2                                               #int
c_puct =    10                                              #int
expand_atoms =  2                                           #int
local_radius = 1                                            #int
sample_num = 5                                              #int
max_nodes = 15                                              #int


class subgraphX_autogoal_no_analyzer():
    def __init__(
            self, 
            model, 
            state_dict=state_dict,  
            num_hops=num_hops, # int [1-3]
            rollout=rollout,    # int [1-100]
            min_atoms=min_atoms, # int [1-10]
            c_puct=c_puct,          #float [5-15]
            ):
        self.model = model
        self.state_dict = state_dict
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.num_hops = num_hops
        self.rollout = rollout
        self.min_atoms = min_atoms
        self.c_puct = c_puct
        self.expand_atoms = expand_atoms
        self.local_radius = local_radius
        self.sample_num = sample_num
        self.max_nodes = max_nodes


    def load_model(self):
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    def load_dataset(self):
        dataset = Dataset_GNN(root=args['root_test_data_path'], files_path=args['raw_test_data_path'])
        return dataset    

    def explain(self, dataset, num_hops, rollout, min_atoms, c_puct, 
                expand_atoms, local_radius, sample_num, max_nodes):
        explainer = SubgraphX(self.model, num_classes=4, device=self.device, 
                              num_hops=num_hops, rollout=rollout, min_atoms=min_atoms, 
                              c_puct=c_puct, expand_atoms=expand_atoms, local_radius=local_radius, 
                              sample_num=sample_num, save_dir='/usrvol/explainability_results/')
        masked_list = []
        maskout_list = []
        sparsity_list = []
        for i, file in enumerate(range(len((dataset)))):
            loader = MyDataLoader(dataset[file], batch_size=args['batch_size'], shuffle=False)
            for j, batch in tqdm(enumerate(loader), total=len(loader), leave=True):
                batch1, batch2, label = batch
                
                # Graphs input information for subgraphX
                x_list = [batch1.x.to(device), batch2.x.to(device)]
                edge_index_list = [batch1.edge_index.to(device), batch2.edge_index.to(device)]
                edge_attr_list = [batch1.edge_attr.to(device), batch2.edge_attr.to(device)]
                dict_nodes_list = [batch1.dict_nodes, batch2.dict_nodes]
                batch_list = [batch1.batch.to(device), batch2.batch.to(device)]
                label = int(label.item())

                results_list, related_pred_list = explainer.explain(x_list=x_list, edge_index_list=edge_index_list, 
                                                                        batch_list=batch_list,edge_attr_list=edge_attr_list, 
                                                                        dict_node_list=dict_nodes_list, label=label,
                                                                        max_nodes=max_nodes)
                masked_list += [related_pred_list[i]['masked'] for i in range(len(related_pred_list))]
                maskout_list += [related_pred_list[i]['maskout'] for i in range(len(related_pred_list))]
                sparsity_list += [related_pred_list[i]['sparsity'] for i in range(len(related_pred_list))]
            
            break


        #Estos tres valores los queremos maximizar

        masked_score = np.mean(masked_list)
        maskout_score = np.mean(maskout_list)
        sparsity_score = np.mean(sparsity_list)

        print(f"Masked score: {masked_score}")
        print(f"Maskout score: {maskout_score}")
        print(f"Sparsity score: {sparsity_score}")

        return masked_score, maskout_score, sparsity_score

    def save_results(self, masked_score, maskout_score, sparsity_score):
        results = {'masked_score': masked_score, 'maskout_score': maskout_score, 'sparsity_score': sparsity_score}
        current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        with open(f"/usrvol/explainability_results/results{current_datetime}.pkl", 'wb') as f:
            pkl.dump(results, f)

    def main(self):
        self.load_model()
        dataset = self.load_dataset()
        masked_score, maskout_score, sparsity_score = self.explain(dataset, num_hops, rollout, min_atoms, c_puct, 
                                                                    expand_atoms, local_radius, sample_num, max_nodes)
        self.save_results(masked_score, maskout_score, sparsity_score)

        return masked_score, maskout_score, sparsity_score


model = GNN_classifier(size=args['size'], num_layers=args['num_layers'], dropout=args['dropout'], module= args['module'], layer_norm=args['layer_norm'], residual=args['residual'], pooling=args['pooling'], lin_transform=args['lin_transform'])
optimizer = subgraphX_autogoal_no_analyzer(model)
masked_score, maskout_score, sparsity_score = optimizer.main()

print(f"Masked score: {masked_score}")
print(f"Maskout score: {maskout_score}")
print(f"Sparsity score: {sparsity_score}")

""" device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = GNN_classifier(size=args['size'], num_layers=args['num_layers'], 
                       dropout=args['dropout'], module= args['module'], 
                       layer_norm=args['layer_norm'], residual=args['residual'], 
                       pooling=args['pooling'], lin_transform=args['lin_transform'])
state_dict = torch.load('/usrvol/results/Trained_GCNCONV2/model14.pt')

model.load_state_dict(state_dict)
model.to(device)
model.eval()

# MODIFICAR RUTA
dataset = Dataset_GNN(root=args['root_test_data_path'], files_path=args['raw_test_data_path'])
explainer = SubgraphX(model, num_classes=4, device=device, num_hops=num_hops, rollout=rollout, min_atoms=min_atoms, c_puct=c_puct,
                                    expand_atoms=expand_atoms, local_radius=local_radius, sample_num=sample_num, 
                                    save_dir='/usrvol/explainability_results/')


masked_list = []
maskout_list = []
sparsity_list = []
for i, file  in enumerate(range(len((dataset)))):
    loader = MyDataLoader(dataset[file], batch_size=args['batch_size'], shuffle=False)
    
    for j, batch in tqdm(enumerate(loader), total=len(loader), leave=True):
        batch1, batch2, label = batch
        
        # Graphs input information for subgraphX
        x_list = [batch1.x.to(device), batch2.x.to(device)]
        edge_index_list = [batch1.edge_index.to(device), batch2.edge_index.to(device)]
        edge_attr_list = [batch1.edge_attr.to(device), batch2.edge_attr.to(device)]
        dict_nodes_list = [batch1.dict_nodes, batch2.dict_nodes]
        batch_list = [batch1.batch.to(device), batch2.batch.to(device)]
        logits = model(batch1.x, batch1.edge_index, batch1.batch, batch2.x, batch2.edge_index, batch2.batch)
        prediction = logits.argmax(dim=-1).item()
        label = int(label.item())
        batches = [batch1, batch2]

        results_list, related_pred_list = explainer.explain(x_list=x_list, edge_index_list=edge_index_list, 
                                                                batch_list=batch_list,edge_attr_list=edge_attr_list, 
                                                                dict_node_list=dict_nodes_list, label=label,
                                                                max_nodes=max_nodes)
        masked_list += [related_pred_list[i]['masked'] for i in range(len(related_pred_list))]
        maskout_list += [related_pred_list[i]['maskout'] for i in range(len(related_pred_list))]
        sparsity_list += [related_pred_list[i]['sparsity'] for i in range(len(related_pred_list))]
    
    break


#Estos tres valores los queremos maximizar

masked_score = np.mean(masked_list)
maskout_score = np.mean(maskout_list)
sparsity_score = np.mean(sparsity_list)

print(f"Masked score: {masked_score}")
print(f"Maskout score: {maskout_score}")
print(f"Sparsity score: {sparsity_score}")
 """