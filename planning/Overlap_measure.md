<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# Measuring Overlap Between TokenSHAP and GraphSVX Explanations: Complete Implementation Guide

Based on your scripts and the need for rigorous explainability comparison, I've created a comprehensive solution for measuring overlap between TokenSHAP (LLM) and GraphSVX (GNN) explanations. Here's what the implementation provides:

## Key Components

### **1. Multi-Dimensional Overlap Assessment**

The analysis evaluates agreement across multiple complementary dimensions :[^1][^2][^3][^4]

- **Rank-based agreement**: Kendall's τ and Spearman's ρ to capture monotonic relationships
- **Value-based similarity**: Pearson correlation and cosine similarity for directional/magnitude alignment
- **Sign agreement**: Matthews correlation coefficient for polarity consistency
- **Top-k overlap**: Jaccard similarity at different sparsity levels (k=1,3,5,10)
- **Distributional divergence**: Jensen-Shannon divergence and Earth Mover's distance
- **Faithfulness metrics**: Attribution mass overlap to ensure explanations reflect model behavior


### **2. Methodological Rigor**

Following best practices from explainability evaluation research :[^5][^3][^6][^4]

- **Statistical significance testing** with p-values for correlation metrics
- **Robustness to ties and outliers** using rank-based methods
- **Multiple baseline comparisons** to assess significance beyond chance
- **Cross-model faithfulness validation** through perturbation analysis


### **3. Data Structure Compatibility**

The implementation works directly with your existing data structures:

**TokenSHAP Results**: Uses the DataFrame from `explainer.analyze()` containing token combinations and predictions
**GraphSVX Results**: Leverages the `node_importance` array storing direct SHAP values per node

### **4. Special Token Handling**

Properly aligns both explanation methods by:

- Excluding [CLS] and [SEP] tokens consistently
- Using content-only tokens (indices 1 to num_nodes-2 for GraphSVX)
- Matching TokenSHAP's `HFWordpieceSplitter(include_special=False)` approach


## Critical Implementation Note

**The `extract_token_attributions_from_tokenSHAP` function requires customization** based on how TokenSHAP stores final attribution scores in its result DataFrame. You'll need to adapt this function to extract the actual Shapley values computed by TokenSHAP from the coalition data it generates.

## Usage Instructions

1. **Install dependencies**: `scipy`, `pandas`, `numpy`, `scikit-learn`
2. **Customize TokenSHAP extraction**: Modify the `extract_token_attributions_from_tokenSHAP` function based on your TokenSHAP output format
3. **Run analysis**:
```python
python overlap-analysis-implementation.py
```

4. **Review results**: The script generates:
    - `overlap_analysis_results.csv`: Sentence-by-sentence metrics
    - `overlap_analysis_summary.json`: Corpus-level statistics
    - `overlap_analysis_report.txt`: Human-readable detailed report

## Expected Insights

This rigorous analysis will reveal whether your LLM and GNN teacher-student setup produces:

- **High overlap** (τ > 0.5, Jaccard@5 > 0.4): Both models identify similar important tokens with consistent polarity, suggesting successful knowledge transfer
- **Partial overlap** (moderate correlations with some disagreement): Agreement on core features but different focus areas, indicating complementary insights
- **Low overlap** (weak correlations): Different explanation strategies, potentially revealing distinct model behaviors

The multi-metric approach ensures robust conclusions about explanation similarity while identifying specific dimensions where your TokenSHAP and GraphSVX explanations agree or diverge.[^3][^7][^1][^5]

<div style="text-align: center">⁂</div>

[^1]: https://ieeexplore.ieee.org/document/8900669/

[^2]: https://ieeexplore.ieee.org/document/10678530/

[^3]: https://arxiv.org/abs/2408.08137

[^4]: https://link.springer.com/10.1007/s11263-024-02282-6

[^5]: https://www.semanticscholar.org/paper/7ca43649f19856d50a563cacfab833a07d009623

[^6]: https://ieeexplore.ieee.org/document/10208308/

[^7]: https://neurips.cc/virtual/2023/poster/73690

[^8]: https://ojs.bonviewpress.com/index.php/JCCE/article/view/2694

[^9]: https://dl.acm.org/doi/10.1145/3524610.3527909

[^10]: https://www.semanticscholar.org/paper/852b9429cd3093af281415822c0854b2e97bfc8f

[^11]: https://arxiv.org/abs/2503.05335

[^12]: https://www.semanticscholar.org/paper/97cb64e02208567fc817b241434ad93e12984fdb

[^13]: https://dl.acm.org/doi/10.1145/3487553.3524211

[^14]: https://arxiv.org/abs/2505.11111

[^15]: https://www.mdpi.com/2392-7674/12/1/29

[^16]: https://arxiv.org/pdf/2209.13429.pdf

[^17]: https://arxiv.org/html/2502.12525v1

[^18]: https://arxiv.org/pdf/2102.05799.pdf

[^19]: https://arxiv.org/pdf/2110.09167v2.pdf

[^20]: https://arxiv.org/pdf/2106.12543.pdf

[^21]: http://arxiv.org/pdf/2403.13106.pdf

[^22]: https://arxiv.org/abs/2503.14469

[^23]: https://arxiv.org/pdf/2405.01848.pdf

[^24]: https://arxiv.org/pdf/1902.05622.pdf

[^25]: http://arxiv.org/pdf/2304.01811.pdf

[^26]: https://jmlr.org/papers/volume23/21-0439/21-0439.pdf

[^27]: https://proceedings.neurips.cc/paper_files/paper/2023/file/ee208bfc04b1bf6125a6a34baa1c28d3-Paper-Conference.pdf

[^28]: https://proceedings.mlr.press/v162/liu22i/liu22i.pdf

[^29]: https://arxiv.org/html/2406.04606v1

[^30]: https://arxiv.org/html/2302.05666v5

[^31]: https://aclanthology.org/2022.acl-long.345.pdf

[^32]: https://pmc.ncbi.nlm.nih.gov/articles/PMC8189022/

[^33]: https://en.wikipedia.org/wiki/Jaccard_index

[^34]: https://pmc.ncbi.nlm.nih.gov/articles/PMC10975804/

[^35]: https://thesis.eur.nl/pub/66661/Thesis-Julia.pdf

[^36]: https://aclanthology.org/2025.semeval-1.257.pdf

[^37]: https://arxiv.org/html/2504.06800v1

[^38]: https://www.ijcai.org/proceedings/2022/0778.pdf

[^39]: https://orbi.uliege.be/bitstream/2268/322654/1/ExploRE_Pirenne.pdf

[^40]: https://www.sciencedirect.com/science/article/pii/S0925231224010531

[^41]: https://christophm.github.io/interpretable-ml-book/shap.html

[^42]: https://arxiv.org/pdf/2010.06775.pdf

[^43]: https://www.nature.com/articles/s41598-025-09538-2

[^44]: https://eprints.whiterose.ac.uk/id/eprint/211725/1/2024 ShapROC postscript.pdf

[^45]: https://kth.diva-portal.org/smash/get/diva2:1704879/fulltext01.pdf

[^46]: https://dl.acm.org/doi/10.1145/3511808.3557418

[^47]: https://arxiv.org/abs/2402.08845

[^48]: https://dl.acm.org/doi/10.1145/3617380

[^49]: https://www.ijcai.org/proceedings/2024/59

[^50]: https://ieeexplore.ieee.org/document/10543008/

[^51]: https://arxiv.org/abs/2502.18848

[^52]: http://arxiv.org/pdf/2408.08137.pdf

[^53]: https://arxiv.org/html/2404.03426v3

[^54]: https://arxiv.org/pdf/2502.17022.pdf

[^55]: https://arxiv.org/pdf/2209.01782.pdf

[^56]: http://arxiv.org/pdf/2402.08845.pdf

[^57]: https://arxiv.org/html/2411.14946

[^58]: https://arxiv.org/pdf/2310.04178.pdf

[^59]: https://zenodo.org/record/8375413/files/Arias-Duart_A_Confusion_Matrix_for_Evaluating_Feature_Attribution_Methods_CVPRW_2023_paper.pdf

[^60]: https://arxiv.org/html/2303.01538

[^61]: https://arxiv.org/html/2310.06514v2

[^62]: https://julian-urbano.info/files/students/msc2018radja.pdf

[^63]: https://pmc.ncbi.nlm.nih.gov/articles/PMC10544769/

[^64]: https://iml.dfki.de/comprehensive-evaluation-of-feature-attribution-methods-in-explainable-ai-via-input-perturbation/

[^65]: https://www.statisticssolutions.com/free-resources/directory-of-statistical-analyses/kendalls-tau-and-spearmans-rank-correlation-coefficient/

[^66]: https://aclanthology.org/2021.naacl-main.75.pdf

[^67]: https://psychology.town/statistics/rank-order-correlations-spearman-kendall-explained/

[^68]: https://arxiv.org/abs/2308.03161

[^69]: https://openreview.net/forum?id=6zcfrSz98y\&noteId=dhrMalPaBR

[^70]: https://aclanthology.org/J06-4002.pdf

[^71]: https://www.sciencedirect.com/science/article/pii/S0167923624000812

[^72]: https://aclanthology.org/2025.acl-long.86.pdf

[^73]: https://www.atlantis-press.com/article/2270.pdf

[^74]: https://open-research-europe.ec.europa.eu/articles/5-191

[^75]: https://proceedings.neurips.cc/paper_files/paper/2023/file/05957c194f4c77ac9d91e1374d2def6b-Paper-Datasets_and_Benchmarks.pdf

[^76]: https://taylorandfrancis.com/knowledge/Medicine_and_healthcare/Medical_statistics_\&_computing/Rank_correlation/

[^77]: https://proceedings.neurips.cc/paper_files/paper/2024/file/b17799e0bbbf65687f4e2df1f98aa225-Paper-Datasets_and_Benchmarks_Track.pdf

[^78]: https://www.sciencedirect.com/science/article/pii/S1877050921019748/pdf?md5=d0f3ad9d8d592a9a901836c7428ba9d3\&pid=1-s2.0-S1877050921019748-main.pdf

[^79]: https://ppl-ai-code-interpreter-files.s3.amazonaws.com/web/direct-files/27d8643e4b58183ba877e9268392ae2c/bb568320-cdb4-4354-9248-6b5a9694e957/72da2d3d.md

[^80]: https://ppl-ai-code-interpreter-files.s3.amazonaws.com/web/direct-files/27d8643e4b58183ba877e9268392ae2c/c7bd0aab-11e2-4739-bc7b-d5bc8f907c2e/6abd7576.py

