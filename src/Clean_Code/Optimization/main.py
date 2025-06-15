import torch
from Clean_Code.Optimization.custom_subgraphx import CustomSubgraphX
import numpy as np
from tqdm import tqdm
from Clean_Code.GNN_Training.gnn_models import GNN_Classifier
from Clean_Code.LazyTrainer.datasets import load_graph_data
from torch_geometric.loader import DataLoader
import json
import os

# --- Patch DIG Shapley to fix MarginalSubgraphDataset instantiation ---
from Clean_Code.Optimization.custom_shapley_patch import patch_marginal_contribution, value_func_wrapper
patch_marginal_contribution()

# Path to best model and args
BEST_MODEL_DIR = "/app/output_lazy/stanfordnlp_sst2_GCNConv_20250612_091841"
BEST_MODEL_PATH = os.path.join(BEST_MODEL_DIR, "best_model.pt")
ARGS_PATH = os.path.join(BEST_MODEL_DIR, "args.json")
VAL_DATA_PATH = "/app/src/Clean_Code/output/pyg_graphs/stanfordnlp/sst2/validation"

with open(ARGS_PATH, "r") as f:
    args = json.load(f)

# Model instantiation: use args from training
model = GNN_Classifier(
    input_dim=args.get("input_dim", 768),
    hidden_dim=args["hidden_dim"],
    output_dim=args.get("num_classes", 2),
    num_layers=args["num_layers"],
    dropout=args["dropout"],
    module=args["module"],
    layer_norm=args.get("layer_norm", False),
    residual=args.get("residual", False),
    pooling=args.get("pooling", "max")
)

# Validation data loading using LazyTrainer helper
_, val_loader = load_graph_data(
    data_dir=VAL_DATA_PATH,
    batch_size=1,
    shuffle=False,
    num_workers=4
)
model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location="cpu"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()


import pickle as pkl
from datetime import datetime
from autogoal.kb import AlgorithmBase, SemanticType
from autogoal.grammar import (
    DiscreteValue,
    ContinuousValue,
)
from autogoal.utils import nice_repr

"""HIPERPARÁMETROS A OPTIMIZAR"""

# num_hops = 1  # int
# rollout = 50  # int
# min_atoms = 2  # int
# c_puct = 10  # int
# expand_atoms = 2  # int
# local_radius = 1  # int
# sample_num = 5  # int
# max_nodes = 15  # int


def explain_with_subgraphx(model, loader, device, num_hops=1, rollout=50, min_atoms=2, c_puct=10, expand_atoms=2, local_radius=1, sample_num=5, max_nodes=15):
    explainer = CustomSubgraphX(
        model=model,
        num_classes=args.get("num_classes", 2),
        device=device,
        num_hops=num_hops,
        rollout=rollout,
        min_atoms=min_atoms,
        c_puct=c_puct,
        expand_atoms=expand_atoms,
        local_radius=local_radius,
        sample_num=sample_num,
        save_dir=os.path.join(BEST_MODEL_DIR, "subgraphx_explanations"),
        value_func=value_func_wrapper(model)
    )
    masked_list = []
    maskout_list = []
    sparsity_list = []
    
    for batch in tqdm(loader, desc="Explaining validation graphs"):
        batch = batch.to(device)
        x = batch.x
        edge_index = batch.edge_index
        label = int(batch.y.item()) if hasattr(batch, 'y') else 0
        batch_idx = batch.batch if hasattr(batch, 'batch') else torch.zeros(x.size(0), dtype=torch.long)

        _, related_pred = explainer.explain(
            x=x,
            edge_index=edge_index,
            label=label,
            max_nodes=max_nodes
        )
        masked_list.append(related_pred["masked"])
        maskout_list.append(related_pred["maskout"])
        sparsity_list.append(related_pred["sparsity"])

# --- AutoGOAL-compatible explain function ---
def explain(model, loader, device, num_hops=1, rollout=50, min_atoms=2, c_puct=10, expand_atoms=2, local_radius=1, sample_num=5, max_nodes=15, batch=None):
    """
    AutoGOAL-compatible explain function. The 'batch' argument is ignored (required for signature compatibility).
    Calls explain_with_subgraphx with the provided arguments.
    """
    return explain_with_subgraphx(
        model,
        loader,
        device,
        num_hops=num_hops,
        rollout=rollout,
        min_atoms=min_atoms,
        c_puct=c_puct,
        expand_atoms=expand_atoms,
        local_radius=local_radius,
        sample_num=sample_num,
        max_nodes=max_nodes
    )
    return masked_list, maskout_list, sparsity_list


def load_dataset():
    # Try to get the test split path, fallback to evaluation split if not present
    test_path = args.get("root_test_data_path")
    eval_path = args.get("root_eval_data_path")
    # Default hardcoded fallback for SST2, adjust as needed
    default_eval = "/app/src/Clean_Code/output/pyg_graphs/stanfordnlp/sst2/validation"
    if test_path is not None and os.path.exists(test_path):
        data_dir = test_path
    elif eval_path is not None and os.path.exists(eval_path):
        print("[WARN] Test set not found, using evaluation split.")
        data_dir = eval_path
    else:
        print("[WARN] No test/eval path in args or file not found, using default evaluation split.")
        data_dir = default_eval
    dataset, _ = load_graph_data(
        data_dir=data_dir,
        batch_size=1,  # batch_size is not used for sampling, set to 1
        shuffle=False,
        num_workers=4
    )
    return dataset


class XGraph(SemanticType):
    pass


class GraphSeqExplanation(SemanticType):
    pass

@nice_repr
class SubgraphXAutogoal(AlgorithmBase):
    def __init__(
        self,
        num_hops: DiscreteValue(1, 5),  # type: ignore
        rollout: DiscreteValue(50, 300),  # type: ignore
        min_atoms: DiscreteValue(1, 10),  # type: ignore
        c_puct: ContinuousValue(1, 30),  # type: ignore
        expand_atoms: DiscreteValue(1, 5),  # type: ignore
        local_radius: DiscreteValue(1, 5),  # type: ignore
        sample_num: DiscreteValue(1, 5),  # type: ignore
        max_nodes: DiscreteValue(2, 40),  # type: ignore
    ):
        self.model = GNN_Classifier(
            input_dim=args.get("input_dim", 768),
            hidden_dim=args["hidden_dim"],
            output_dim=args.get("num_classes", 2),
            num_layers=args["num_layers"],
            dropout=args["dropout"],
            module=args["module"],
            layer_norm=args.get("layer_norm", False),
            residual=args.get("residual", False),
            pooling=args.get("pooling", "max")
        )
        model_path = args.get('model_dir', BEST_MODEL_PATH)
        if 'model_dir' not in args:
            print(f"[WARN] 'model_dir' not found in args, using BEST_MODEL_PATH: {BEST_MODEL_PATH}")
        self.state_dict = torch.load(model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_hops = num_hops
        self.rollout = rollout
        self.min_atoms = min_atoms
        self.c_puct = c_puct
        self.expand_atoms = expand_atoms
        self.local_radius = local_radius
        self.sample_num = sample_num
        self.max_nodes = max_nodes

    def load_model(self):
        self.model.load_state_dict(self.state_dict)
        self.model.to(self.device)
        self.model.eval()

    def save_results(self, masked_score, maskout_score, sparsity_score):
        results = {
            "masked_score": masked_score,
            "maskout_score": maskout_score,
            "sparsity_score": sparsity_score,
        }
        current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        with open(
            f"/usrvol/explainability_results/results{current_datetime}.pkl", "wb"
        ) as f:
            pkl.dump(results, f)

    def run(self, dataset: XGraph) -> GraphSeqExplanation:
        self.load_model()
        # Call explain_with_subgraphx instead of undefined explain
        masked_list, maskout_list, sparsity_list = explain_with_subgraphx(
            self.model,
            dataset,  # dataset is expected to be a loader
            self.device,
            self.num_hops,
            self.rollout,
            self.min_atoms,
            self.c_puct,
            self.expand_atoms,
            self.local_radius,
            self.sample_num,
            self.max_nodes
        )
        # Optionally, you may want to wrap or return these in a GraphSeqExplanation or similar structure
        return GraphSeqExplanation(masked_list, maskout_list, sparsity_list)


def my_metric(X, result):
    masked_score = np.mean(result[0])
    maskout_score = np.mean(result[1])
    sparsity_score = np.mean(result[2])
    return np.mean([masked_score, maskout_score, sparsity_score])


def run_autogoal():
    # AutoGOAL Example: basic usage of the AutoML class
    from autogoal.kb import MatrixContinuousSparse, VectorDiscrete
    from autogoal.ml import AutoML, calinski_harabasz_score
    from autogoal.utils import Min, Gb, Hour, Sec
    from autogoal.search import PESearch, JsonLogger, ConsoleLogger
    from autogoal.utils._process import initialize_cuda_multiprocessing

    initialize_cuda_multiprocessing()

    # Load dataset
    initial_dataset = load_dataset()
    np.random.seed(54)
    # Use a default for dataset_length if not present
    dataset_length = args.get('dataset_length', len(initial_dataset))
    num_samples = args.get('num_samples', 100)
    if 'num_samples' not in args:
        print(f"[WARN] 'num_samples' not found in args, defaulting to {num_samples}.")
    sample = np.random.choice(dataset_length, num_samples, replace=False)
    training_sample = []
    # Sample graphs directly from the dataset
    for i in sample:
        try:
            training_sample.append(initial_dataset[i])
        except Exception as e:
            print(f"[ERROR] Could not access graph index {i}: {e}")
    X_train = training_sample
    # Optionally, print dataset stats for debugging
    print(f"[INFO] Training sample size: {len(X_train)} / {len(initial_dataset)} total graphs")

    automl = AutoML(
        # Declare the input and output types
        input=XGraph,
        output=GraphSeqExplanation,
        objectives=(my_metric,),
        registry=[
            SubgraphXAutogoal],
        # Search space configuration
        search_timeout=300 * Hour,
        evaluation_timeout=24 * Hour,
        memory_limit=8 * Gb,
    )

    # Run the pipeline search process
    automl.fit(X_train, logger=[
        ConsoleLogger(),
        JsonLogger("experiment.json")
    ])

    # Report the best pipelines
    print(automl.best_pipelines_)
    print(automl.best_scores_)

# optimizer = subgraphX_autogoal_no_analyzer()
# masked_score, maskout_score, sparsity_score = optimizer.main()

# print(f"Masked score: {masked_score}")
# print(f"Maskout score: {maskout_score}")
# print(f"Sparsity score: {sparsity_score}")
if __name__=="__main__":
    run_autogoal()