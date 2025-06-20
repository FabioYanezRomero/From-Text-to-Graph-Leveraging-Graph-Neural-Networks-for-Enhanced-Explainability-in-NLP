import argparse
import os
from pathlib import Path
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
import shap
import matplotlib.pyplot as plt
import json
from metrics import ForwardPassCounter, measure_time, measure_memory, profile_hardware, estimate_flops


def parse_args():
    parser = argparse.ArgumentParser(description="LLM Explainability with SHAP on SST-2")
    parser.add_argument("--model_path", type=str, default=None, help="Path to fine-tuned BERT model directory (overrides --dataset)")
    parser.add_argument("--dataset", type=str, default="stanfordnlp/sst2", help="Dataset name (used to find best checkpoint if --model_path not given)")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save SHAP outputs")
    parser.add_argument("--num_samples", type=int, default=50, help="Number of validation samples to explain")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use: 'cpu' or 'cuda'")
    return parser.parse_args()


def find_best_checkpoint(dataset_name):
    import glob, json
    import numpy as np
    # Map dataset name to checkpoint root
    ds_path = dataset_name.replace('/', os.sep)
    base_dir = os.path.join(os.path.dirname(__file__), '..', 'output', 'finetuned_llms', ds_path)
    # Find all subdirs (runs)
    runs = [os.path.join(base_dir, d) for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    best_f1 = -1
    best_model_path = None
    best_epoch = None
    for run in runs:
        reports = glob.glob(os.path.join(run, 'classification_report_test_epoch*.json'))
        for rep in reports:
            with open(rep) as f:
                report = json.load(f)
                f1 = report.get('macro avg', {}).get('f1-score', -1)
                if f1 > best_f1:
                    best_f1 = f1
                    # Extract epoch number
                    epoch = int(rep.split('epoch')[-1].split('.')[0])
                    model_path = os.path.join(run, f'model_epoch_{epoch}.pt')
                    # Confirm model exists
                    if os.path.exists(model_path):
                        best_model_path = run
                        best_epoch = epoch
    if best_model_path is None:
        raise FileNotFoundError(f"No valid model checkpoint found for dataset {dataset_name}")
    print(f"[INFO] Selected checkpoint: {best_model_path} (epoch {best_epoch}, f1={best_f1:.4f})")
    return best_model_path, best_epoch



def load_data(num_samples):
    dataset = load_dataset("stanfordnlp/sst2", split="validation")
    texts = [item['sentence'] for item in dataset.select(range(num_samples))]
    labels = [item['label'] for item in dataset.select(range(num_samples))]
    return texts, labels


def predict_proba(texts, model, tokenizer, device, eval_counter=None):
    # Tokenize and run through model, returning softmax probabilities
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1).cpu().numpy()
    if eval_counter is not None:
        eval_counter.increment(1)
    return probs


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    # Determine model path
    if args.model_path is not None:
        model_dir = args.model_path
        print(f"[INFO] Using user-specified model path: {model_dir}")
    else:
        model_dir, best_epoch = find_best_checkpoint(args.dataset)
        print(f"[INFO] Using best checkpoint for dataset '{args.dataset}' at: {model_dir}, epoch {best_epoch}")

    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    print(f"Loading {args.num_samples} samples from SST-2 validation set...")
    texts, labels = load_data(args.num_samples)

    print("Preparing SHAP explainer...")
    # Use a sample of background data for masker
    background_texts = texts[:10]
    eval_counter = ForwardPassCounter()
    explainer = shap.Explainer(
        lambda x: predict_proba(x, model, tokenizer, device, eval_counter),
        masker=shap.maskers.Text(tokenizer)
    )

    print("Computing SHAP values & metrics...")
    explanation_times = []
    model_evals = []
    input_lengths = []
    peak_memories = []
    hardware_stats = []
    flops_per_instance = []
    shap_values = []
    for i, text in enumerate(texts):
        eval_counter.reset()
        input_lengths.append(len(tokenizer.tokenize(text)))
        with measure_time() as t_fn, measure_memory() as m_fn:
            sv = explainer([text])[0]
        explanation_times.append(t_fn())
        model_evals.append(eval_counter.get())
        peak_memories.append(m_fn())
        hardware_stats.append(profile_hardware())
        # Estimate FLOPs for this explanation
        flops = estimate_flops(model, tokenizer, text, device)
        flops_per_instance.append(flops)
        shap_values.append(sv)

    print(f"Saving SHAP visualizations to {args.output_dir}...")
    for i, (sv, text, label) in enumerate(zip(shap_values, texts, labels)):
        plt.figure(figsize=(10, 2))
        shap.plots.text(sv, display=False)
        plt.title(f"Sample {i} | True label: {label}")
        plt.savefig(os.path.join(args.output_dir, f"shap_text_{i}.png"), bbox_inches='tight')
        plt.close()

    # Save metrics
    metrics = {
        'explanation_time_per_instance': explanation_times,
        'model_evaluations_per_explanation': model_evals,
        'input_lengths': input_lengths,
        'peak_memory_bytes': peak_memories,
        'hardware_utilization': hardware_stats,
        'flops_per_instance': flops_per_instance,
        'mean_explanation_time': float(sum(explanation_times)/len(explanation_times)),
        'mean_model_evals': float(sum(model_evals)/len(model_evals)),
        'mean_peak_memory_bytes': float(sum(peak_memories)/len(peak_memories)),
        'mean_flops': float(sum([f for f in flops_per_instance if f is not None])/max(1, len([f for f in flops_per_instance if f is not None])))
    }
    with open(os.path.join(args.output_dir, 'metrics_shap.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    print("Done.")

if __name__ == "__main__":
    main()
