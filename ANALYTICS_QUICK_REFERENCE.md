# Analytics Quick Reference Card

## 🚀 Run Everything (One Command)

```bash
./run_all_analytics.sh
```

**That's it!** Wait ~10-15 minutes, then check results.

---

## 📊 What It Does

Analyzes **all combinations** of explainers:

### LLM vs GNN Agreement
- Token SHAP ↔️ Syntactic graphs
- Token SHAP ↔️ Constituency graphs
- Token SHAP ↔️ Skipgrams
- Token SHAP ↔️ Window graphs

### GNN vs GNN Agreement
- Constituency ↔️ Syntactic
- Skipgrams ↔️ Window
- All other graph type pairs

### Metrics Computed
- RBO, Spearman, Kendall correlations
- Feature overlap ratios
- Fidelity+/-, Faithfulness
- Insertion/Deletion AUC
- Contrastivity, Compactness
- Statistical tests

---

## ⚡ Quick Commands

```bash
# Full analysis with plots (10-15 min)
./run_all_analytics.sh

# Fast analysis without plots (3-5 min)
./run_all_analytics.sh --no-plots

# Custom directories
./run_all_analytics.sh --insights-dir /path --output-dir /path

# Help
./run_all_analytics.sh --help
```

---

## 📁 Key Output Files

```bash
# Executive summary (read first!)
cat outputs/analytics/complete/stanfordnlp_sst2/executive_summary.md

# Best methods
cat outputs/analytics/comparative/llm_vs_gnn_overall/method_ranking.csv

# Agreement analysis (all combinations!)
cat outputs/analytics/ranking_agreement/overall/agreement_metrics_table.csv

# Summary report
cat outputs/analytics/summary_*/ANALYTICS_SUMMARY.md
```

---

## 🔍 Check Results

```bash
# List all outputs
ls outputs/analytics/

# View plots
ls outputs/analytics/*/plots/*.png

# View agreement plots specifically
ls outputs/analytics/ranking_agreement/overall/plots/

# View comparison plots
ls outputs/analytics/comparative/*/plots/
```

---

## 📈 Output Structure

```
outputs/analytics/
├── complete/                    # Full pipeline results
├── stratified/                  # By class + correctness
├── comparative/                 # LLM vs GNN
│   └── llm_vs_gnn_overall/
│       └── method_ranking.csv   ← Best methods here
├── ranking_agreement/           # Agreement analysis
│   └── overall/
│       ├── agreement_metrics_table.csv  ← All agreements
│       └── plots/               ← Agreement visualizations
├── contrastivity_compactness/   # Quality metrics
├── enhanced_auc/                # AUC curves
├── comprehensive_faithfulness/  # Faithfulness
└── summary_<timestamp>/         # Consolidated summary
```

---

## 🎯 Interpreting Agreement Metrics

| Metric | Range | Good Value | Meaning |
|--------|-------|------------|---------|
| RBO | 0-1 | >0.7 | Top features agree |
| Spearman | -1 to 1 | >0.5 | Consistent ranking |
| Kendall | -1 to 1 | >0.5 | Rank correlation |
| Overlap | 0-1 | >0.6 | Shared features |
| Jaccard | 0-1 | >0.5 | Set similarity |

---

## ⏱️ Execution Time

| Configuration | Time | Outputs |
|--------------|------|---------|
| Full (with plots) | ~10-15 min | 500+ files |
| No plots | ~3-5 min | 300+ files |
| Skip modules | ~5-8 min | Selected only |

---

## 🛠️ Options

```bash
--no-plots              # Skip plots (faster)
--insights-dir DIR      # Custom insights path
--output-dir DIR        # Custom output path
--skip-complete         # Skip complete pipeline
--skip-stratified       # Skip stratified analysis
--skip-comparative      # Skip comparative analysis
--help                  # Show help
```

---

## 💡 Common Use Cases

### 1. Quick Overview
```bash
./run_all_analytics.sh --no-plots
cat outputs/analytics/comparative/llm_vs_gnn_overall/method_ranking.csv
```

### 2. Full Analysis
```bash
./run_all_analytics.sh
cat outputs/analytics/complete/*/executive_summary.md
```

### 3. Check Agreement Only
```bash
cat outputs/analytics/ranking_agreement/overall/agreement_metrics_table.csv
cat outputs/analytics/ranking_agreement/overall/ranking_agreement_summary.json
```

---

## 🔧 Troubleshooting

**Out of memory?**
```bash
./run_all_analytics.sh --no-plots
```

**No agreement data?**
- Check `*_agreement.json` files exist in insights
- Run Insights module first

**Partial execution?**
- Script continues on errors
- Check `complete_pipeline_summary.json` for details

---

## 📚 Documentation

- **Full guide**: `/app/ANALYTICS_SCRIPT_GUIDE.md`
- **Usage examples**: `/app/ANALYTICS_SCRIPT_COMPLETE.md`
- **Analytics docs**: `/app/src/Analytics/NEW_ANALYTICS_README.md`
- **Quick start**: `/app/QUICK_START_ANALYTICS.md`

---

## ✅ Ready to Run

```bash
cd /app
./run_all_analytics.sh
```

Wait ~15 minutes, then:

```bash
cat outputs/analytics/summary_*/ANALYTICS_SUMMARY.md
```

**That's all you need!** 🎉

---

## 📞 Quick Help

```bash
./run_all_analytics.sh --help
```

---

**Created:** October 2025  
**Version:** 1.0  
**Status:** ✅ Production Ready

