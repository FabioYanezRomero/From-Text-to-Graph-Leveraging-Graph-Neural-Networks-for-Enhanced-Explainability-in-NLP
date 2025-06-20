# LLM Explainability Pipeline (SHAP + BERT)

This pipeline explains predictions of fine-tuned BERT models on the SST-2 validation set using SHAP.

## Usage

1. Install requirements (in a clean environment):
   ```bash
   pip install -r requirements_explainability.txt
   ```
2. Run the pipeline:
   ```bash
   python explainability_pipeline.py --model_path /path/to/your/model --output_dir ./shap_outputs --num_samples 50 --device cpu
   ```

## Arguments
- `--model_path`: Path to the fine-tuned BERT model directory.
- `--output_dir`: Directory to save SHAP visualizations.
- `--num_samples`: Number of validation samples to explain.
- `--device`: Device to use (`cpu` or `cuda`).

## Notes
- This script uses the SST-2 validation set from the GLUE benchmark (via `datasets`).
- Only minimal dependencies are installed to avoid conflicts in deep learning environments.
