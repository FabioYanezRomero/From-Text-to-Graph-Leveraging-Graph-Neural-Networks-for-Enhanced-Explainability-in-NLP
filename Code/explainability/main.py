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

subset_indices = list(range(100))
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = GNN_classifier(size=args['size'], num_layers=args['num_layers'], dropout=args['dropout'], module= args['module'], layer_norm=args['layer_norm'], residual=args['residual'], pooling=args['pooling'], lin_transform=args['lin_transform'])
state_dict = torch.load('/usrvol/results/Trained_GCNCONV2/model14.pt')

model.load_state_dict(state_dict)

dataset = Dataset_GNN(root=args['root_test_data_path'], files_path=args['raw_test_data_path'])
explainer = SubgraphX(model, num_classes=4, device=device, save_dir='/usrvol/explainability_results/')


total_results = []
total_related_pred = []
identifier = []

wrong_results = []
wrong_related_pred = []
wrong_identifier = []

for i, file  in enumerate(range(len((dataset)))):
    loader = MyDataLoader(dataset[file], batch_size=args['batch_size'], shuffle=False)
    
    for j, batch in tqdm(enumerate(loader), total=len(loader), leave=True):
        batch1, batch2, label = batch
        x_list = [batch1.x.to(device), batch2.x.to(device)]
        edge_index_list = [batch1.edge_index, batch2.edge_index]
        edge_attr_list = [batch1.edge_attr, batch2.edge_attr]
        dict_nodes_list = [batch1.dict_nodes, batch2.dict_nodes]
        batch_list = [batch1.batch, batch2.batch]
        logits = model(batch1.x, batch1.edge_index, batch1.batch, batch2.x, batch2.edge_index, batch2.batch)
        prediction = logits.argmax(dim=-1).item()
        if prediction != label.item():
            try:
                # Pasar aquí los datos de edge_attr_list y dict_nodes_list para que los devuelva en los resultados
                results_list, related_pred_list = explainer.explain(x_list=x_list, edge_index_list=edge_index_list, 
                                                                    batch_list=batch_list,edge_attr_list=edge_attr_list, 
                                                                    dict_node_list=dict_nodes_list, label=label)
                wrong_results.append(results_list)
                wrong_related_pred.append(related_pred_list)
                wrong_identifier.append(j)
            except:
                print(f"Error on batch {j} in file {i} for a wrong prediction")

        else:
            try:
                results_list, related_pred_list = explainer.explain(x_list=x_list, edge_index_list=edge_index_list, 
                                                                    batch_list=batch_list,edge_attr_list=edge_attr_list, 
                                                                    dict_node_list=dict_nodes_list, label=label)
                total_results.append(results_list)
                total_related_pred.append(related_pred_list)
                identifier.append(j)
            except:
                print(f"Error on batch {j} in file {i} for a right prediction")


    with open(f"/usrvol/explainability_results/results_{i}.pkl", 'wb') as f:
        pkl.dump(total_results, f)

    with open(f"/usrvol/explainability_results/related_pred_{i}.pkl", 'wb') as f:
        pkl.dump(total_related_pred, f)

    with open(f"/usrvol/explainability_results/identifier_{i}.pkl", 'wb') as f:
        pkl.dump(identifier, f)

    with open(f"/usrvol/explainability_results/wrong_results_{i}.pkl", 'wb') as f:
        pkl.dump(wrong_results, f)

    with open(f"/usrvol/explainability_results/wrong_related_pred_{i}.pkl", 'wb') as f:
        pkl.dump(wrong_related_pred, f)

    with open(f"/usrvol/explainability_results/wrong_identifier_{i}.pkl", 'wb') as f:
        pkl.dump(wrong_identifier, f)

    break

print("done!")