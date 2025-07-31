import os
import argparse
import pickle as pkl
import json
import glob
from tqdm import tqdm
import torch
from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset


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
        dataset_name = 'stanfordnlp/sst2'
        model_name = 'bert-base-uncased'
        output_dir = f"/app/src/Clean_Code/output/gnn_embeddings/fully_connected/{dataset_name}/{split}"
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        batch_size = 128
        
        # Parse command line arguments
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
        
        # Find all model directories for this dataset
        model_dirs = glob.glob(f"{finetuned_models_dir}/*")
        if not model_dirs:
            raise FileNotFoundError(f"No model directories found in {finetuned_models_dir}")
            
        # Use the most recent model directory
        model_dir = max(model_dirs, key=os.path.getmtime)
        
        # Find all classification report files
        report_files = glob.glob(f"{model_dir}/classification_report_test_epoch*.json")
        
        for report_file in report_files:
            try:
                with open(report_file, 'r') as f:
                    report = json.load(f)
                    f1 = report.get('weighted avg', {}).get('f1-score', -1)
                    if f1 > best_f1:
                        best_f1 = f1
                        epoch = int(os.path.basename(report_file).split('_')[-1].replace('.json', '').replace('epoch', ''))
                        best_epoch = epoch
                        # Try both possible checkpoint filename patterns
                        checkpoint_path = os.path.join(model_dir, f'model_epoch_{epoch}.pt')
                        if not os.path.exists(checkpoint_path):
                            checkpoint_path = os.path.join(model_dir, f'model_final.pt')
                        best_checkpoint = checkpoint_path
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: Could not parse {report_file}: {e}")
                continue
        
        if best_checkpoint is None or not os.path.exists(best_checkpoint):
            raise FileNotFoundError(f"Could not find a valid model checkpoint in {model_dir}")
        
        print(f"Loading best model from {best_checkpoint} (epoch {best_epoch}, F1={best_f1:.4f})")
        
        # First load the base model, then load the state dict
        model = AutoModel.from_pretrained(args.model_name).to(args.device)
        checkpoint = torch.load(best_checkpoint, map_location=args.device)
        model.load_state_dict(checkpoint)
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
            inputs = {k: v.to(args.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)
            
            batch_graphs = []
            # Process each sample in the batch
            for j in range(len(texts)):
                # Get the actual tokens (excluding padding)
                attention_mask = inputs['attention_mask'][j].bool()
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
