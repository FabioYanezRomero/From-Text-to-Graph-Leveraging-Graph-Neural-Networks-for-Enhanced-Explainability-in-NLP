import os
import argparse
import pickle as pkl
import json
import glob
from tqdm import tqdm
import torch
from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset
from torch_geometric.data import Data


def load_sentences(dataset_name, split):
    ds = load_dataset(dataset_name, split=split)
    if 'sentence' in ds.column_names:
        return ds['sentence']
    elif 'text' in ds.column_names:
        return ds['text']
    else:
        raise ValueError("No suitable sentence/text column found in dataset.")


def main():

    
    for split in ['train', 'validation', 'test']:
        dataset_name = 'setfit/ag_news'
        model_name = 'bert-base-uncased'
        output_dir = f"/app/src/Clean_Code/output/gnn_embeddings/fully_connected/{dataset_name}/{split}"
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        batch_size = 128
        parser = argparse.ArgumentParser(description="Generate constituency or syntactic graphs with node embeddings for GNN training.")
        parser.add_argument('--dataset_name', type=str, default=dataset_name)
        parser.add_argument('--split', type=str, default=split)
        parser.add_argument('--output_dir', type=str, default=output_dir)
        parser.add_argument('--model_name', type=str, default=model_name)
        parser.add_argument('--device', type=str, default=device)
        parser.add_argument('--batch_size', type=int, default=batch_size, help='Batch size for processing graphs and sentences')
        args = parser.parse_args()
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        
        # Find the best model checkpoint based on weighted F1 score
        finetuned_models_dir = f"/app/src/Clean_Code/output/finetuned_llms/{args.dataset_name}"
        best_epoch = -1
        best_f1 = -1.0
        best_checkpoint = None
        
        # For ag-news, the model files are directly in the dataset directory
        model_dir = finetuned_models_dir
        
        # Check if we have model files directly in this directory
        model_files = glob.glob(f"{model_dir}/model_epoch_*.pt") or glob.glob(f"{model_dir}/model_final.pt")
        
        # If no model files found, try to find model directories (for other datasets)
        if not model_files:
            model_dirs = glob.glob(f"{finetuned_models_dir}/*")
            if model_dirs:
                # Use the most recent model directory
                model_dir = max(model_dirs, key=os.path.getmtime)
                model_files = glob.glob(f"{model_dir}/model_epoch_*.pt") or glob.glob(f"{model_dir}/model_final.pt")
        
        if not model_files:
            raise FileNotFoundError(f"No model files found in {finetuned_models_dir} or its subdirectories")
        
        # First, try to find the best model based on classification reports
        report_files = glob.glob(f"{model_dir}/classification_report_test_epoch*.json")
        
        if report_files:  # If we have classification reports, use them to find the best model
            for report_file in report_files:
                try:
                    with open(report_file, 'r') as f:
                        report = json.load(f)
                        f1 = report.get('weighted avg', {}).get('f1-score', -1)
                        if f1 > best_f1:
                            best_f1 = f1
                            # Extract epoch number from filename
                            epoch = int(report_file.split('_')[-1].split('.')[0][5:])
                            best_epoch = epoch
                            best_checkpoint = os.path.join(model_dir, f'model_epoch_{epoch}.pt')
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"Warning: Could not parse {report_file}: {e}")
                    continue
        
        # If no best checkpoint found via reports, try to find any model checkpoint
        if best_checkpoint is None or not os.path.exists(best_checkpoint):
            # Try to find the final model first
            final_model_path = os.path.join(model_dir, 'model_final.pt')
            if os.path.exists(final_model_path):
                best_checkpoint = final_model_path
                best_epoch = 'final'
            else:
                # Fall back to any model_epoch_X.pt file
                model_files = glob.glob(f"{model_dir}/model_epoch_*.pt")
                if model_files:
                    # Sort by epoch number and take the last one
                    model_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
                    best_checkpoint = model_files[-1]
                    best_epoch = int(best_checkpoint.split('_')[-1].split('.')[0])
        
        if best_checkpoint is None or not os.path.exists(best_checkpoint):
            raise FileNotFoundError(f"Could not find a valid model checkpoint in {model_dir}")
        
        print(f"Found model checkpoint: {best_checkpoint} (epoch: {best_epoch}, F1: {best_f1:.4f} if available)")
        
        print(f"Loading best model from {best_checkpoint} (epoch {best_epoch}, F1={best_f1:.4f})")
        
        # First load the base model, then load the state dict
        model = AutoModel.from_pretrained(args.model_name).to(args.device)
        checkpoint = torch.load(best_checkpoint, map_location=args.device)
        
        # Handle state dict key mismatch - remove 'bert.' prefix and ignore classifier weights
        bert_state_dict = {}
        for key, value in checkpoint.items():
            if key.startswith('bert.'):
                new_key = key[5:]  # Remove 'bert.' prefix
                bert_state_dict[new_key] = value
        
        # Load only the BERT model weights, ignoring classifier weights
        model.load_state_dict(bert_state_dict, strict=False)
        model = model.to(args.device)

        os.makedirs(args.output_dir, exist_ok=True)
        try:
            dataset = load_dataset(args.dataset_name, split=args.split)
        except Exception as e:
            print(f"Failed to load dataset {args.dataset_name} split {args.split}: {e}")
            continue
        
        batch_size = args.batch_size
        
        # Determine the text field to use
        text_field = 'sentence' if 'sentence' in dataset.column_names else 'text'
        
        # Clean and create output directory
        if os.path.exists(args.output_dir):
            import shutil
            shutil.rmtree(args.output_dir)
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Define the output file path
        output_file = os.path.join(args.output_dir, f'{split}_graphs_with_embeddings.pkl')
        
        # Process in batches
        for batch_idx in tqdm(range(0, len(dataset), batch_size), desc='Processing batches'):
            batch = dataset[batch_idx:batch_idx + batch_size]
            texts = batch[text_field]
            labels = batch['label']
            
            # Tokenize the batch
            inputs = tokenizer(
                texts,
                return_tensors='pt',
                return_offsets_mapping=True,
                truncation=True,
                padding=True,
                return_attention_mask=True
            )
            
            # Move to device and get model outputs
            # Remove offset_mapping as it's not expected by the model
            model_inputs = {k: v.to(args.device) for k, v in inputs.items() if k != 'offset_mapping'}
            with torch.no_grad():
                outputs = model(**model_inputs)
            
            batch_graphs = []
            # Process each sample in the batch
            for j in range(len(texts)):
                # Get the actual tokens (excluding padding)
                attention_mask = model_inputs['attention_mask'][j].bool()
                hidden = outputs.last_hidden_state[j, attention_mask]  # [seq_len, hidden_size]
                
                # Create fully connected graph for this sample
                seq_len = hidden.size(0)
                edge_index = torch.stack([
                    torch.arange(seq_len, device=args.device),
                    torch.arange(seq_len, device=args.device)
                ], dim=0)
                
                # Create graph data
                data = Data(
                    x=hidden.cpu(),
                    edge_index=edge_index.cpu(),
                    y=torch.tensor(labels[j], dtype=torch.long).unsqueeze(0)
                )
                batch_graphs.append(data)
            
            # Save this batch to disk (append to existing file)
            mode = 'ab' if os.path.exists(output_file) and batch_idx > 0 else 'wb'
            with open(output_file, mode) as f:
                for graph in batch_graphs:
                    pkl.dump(graph, f)
            
            # Clear memory
            del batch_graphs, inputs, outputs
            torch.cuda.empty_cache()
        
        print(f"All graphs saved to {output_file}")

if __name__ == "__main__":
    main()
