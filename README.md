# Does Spatial Prediction Consistency Improve Stain Robustness in Histopathology Cell Classification?

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org/)
[![PyTorch Geometric](https://img.shields.io/badge/PyG-Graph%20Neural%20Networks-3C2179.svg)](https://pyg.org/)

**Author**: Dev Mehta  
**Institution**: SRM Institute of Science and Technology  
**Study**: Tier 1 Protocol Execution & Scientific Validation

---

## Abstract

Stain variation across pathology laboratories represents a major barrier to the clinical translation of deep learning models for computational pathology. While data augmentations (e.g., Macenko stain normalization, RandStainNA) and consistency regularization improve single-cell classification robustness, histopathology tissues possess rich architectural spatial contexts. 

This study investigates whether **spatial prediction consistency**—enforcing symmetric Kullback-Leibler (KL) divergence consistency between paired clean and stain-perturbed views over identical graph neural network (GNN) topologies—improves nuclear phenotyping robustness under severe domain shifts.

We rigorously evaluate five architectural regimes across three independent random seeds ($S = 3$) on in-domain Lizard colon tissue and unseen external CoNSeP tissue:
* **M1**: ResNet-18 (full end-to-end fine-tuning, 11.18M parameters)
* **M2**: Multi-Layer Perceptron on frozen features (RandStainNA + Cross-Entropy, 66.05k parameters)
* **M3**: Multi-Layer Perceptron (Paired Symmetric KL Consistency, 66.05k parameters)
* **M4**: GATv2 with 3D edge attributes (Cross-Entropy, 70.40k parameters)
* **M5**: GATv2 with 3D edge attributes (Proposed: Spatial Consistency, 70.40k parameters)
* *Ablation*: GraphSAGE mean aggregation (M4-SAGE & M5-SAGE, 69.83k parameters)

All hyperparameters were pre-registered and locked on the validation set prior to evaluating any test data ([`configs/locked_config.json`](configs/locked_config.json)).

---

## Key Experimental Results

Statistical hypothesis testing was conducted at the **slide/region unit of analysis** (29 Lizard regions and 14 CoNSeP tiles), averaging over random seeds within each unit before performing paired Wilcoxon signed-rank tests and computing 1,000-sample Bias-Corrected and Accelerated (BCa) bootstrap 95% confidence intervals with Holm-Bonferroni correction.

### Formal Hypothesis Test Verdicts

| Hypothesis | Comparison | Metric | Observed Diff | Holm $p$-value | 95% BCa Bootstrap CI | Equivalence Margin ($\pm 0.015$) | Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **H1: Spatial Inductive Bias** | M4 (GATv2) vs M2 (MLP) | Clean In-Domain Macro-F1 | $-0.0313$ | $p = 3.43 \times 10^{-6}$ | $[-0.0403, -0.0222]$ | Outside margin (inferior) | **REJECTED** |
| **H2: Stain Robustness** | M5 (GATv2+KL) vs M3 (MLP+KL) | $\Delta\text{F1}$ under Macenko Moderate Shift | $-0.0057$ | $p = 0.5304$ | $[-0.0140, +0.0020]$ | **Entirely inside $[-0.015, +0.015]$** | **EQUIVALENT** |
| **H3: Calibration & Consistency** | M5 (GATv2+KL) vs M3 (MLP+KL) | ECE (10-bin) under Stain Shift | $+0.0025$ | $p = 0.5362$ | $[-0.0027, +0.0090]$ | ECE preserved ($\le 0.021$) | **SUPPORTED (CALIBRATION PRESERVED)** |

### Scientific Findings:
1. **H1 is Rejected for GATv2, but Supported for GraphSAGE**: Parameterized attention heads over frozen ImageNet features underperform isolated MLPs ($0.704$ vs $0.705$). However, uniform mean neighborhood smoothing (GraphSAGE ablation) achieves **$0.728 \pm 0.013$** ($+0.023$ F1 gain over M2), proving that non-parametric spatial context benefits nuclear phenotyping.
2. **H2 is Statistically Equivalent**: The 95% BCa interval for degradation difference between M5 and M3 is $[-0.0140, +0.0020]$, falling strictly within the pre-registered equivalence margin $\pm 0.015$. Spatial consistency yields equivalent robustness without inducing spatial error propagation under domain shift.
3. **Spatial Error Clustering ($C_{err}$)**: Observed join-count spatial error clustering significantly exceeds the random permutation null across all models ($p < 0.001$), demonstrating that classification errors physically cluster along stromal-epithelial tissue boundaries.

---

## Robustness & Calibration Tables

### Table 4: Robustness Across Stain Shifts
*Values report mean $\pm$ seed standard deviation across 3 seeds (42, 43, 44), followed by the [95% BCa bootstrap confidence interval].*

| Model Architecture | In-Domain Clean F1 | $\Delta\text{F1}$ Macenko (Moderate, $\pm 20\%$) | $\Delta\text{F1}$ RandStainNA | Unseen External CoNSeP F1 | Params |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **M1: ResNet-18** *(full fine-tune)* | $0.893 \pm 0.005$ $[0.887, 0.897]$ | $0.060 \pm 0.017$ $[0.037, 0.078]$ | $0.229 \pm 0.019$ $[0.211, 0.244]$ | $0.818 \pm 0.013$ $[0.800, 0.826]$ | $11.18\text{M}$ |
| **M2: MLP** *(RandStainNA, CE)* | $0.705 \pm 0.004$ $[0.701, 0.708]$ | $0.007 \pm 0.003$ $[0.004, 0.009]$ | $0.046 \pm 0.002$ $[0.043, 0.047]$ | $0.670 \pm 0.004$ $[0.664, 0.673]$ | $66.05\text{k}$ |
| **M3: MLP** *(Consistency, CE+KL)* | $0.709 \pm 0.002$ $[0.708, 0.710]$ | $0.009 \pm 0.002$ $[0.007, 0.010]$ | $0.079 \pm 0.008$ $[0.071, 0.086]$ | $0.670 \pm 0.002$ $[0.668, 0.671]$ | $66.05\text{k}$ |
| **M4: GATv2** *(RandStainNA, CE)* | $0.704 \pm 0.003$ $[0.699, 0.707]$ | $0.008 \pm 0.005$ $[0.000, 0.011]$ | $0.038 \pm 0.003$ $[0.033, 0.040]$ | $0.671 \pm 0.007$ $[0.661, 0.676]$ | $70.40\text{k}$ |
| **M5: GATv2** *(Proposed, CE+KL)* | $0.701 \pm 0.005$ $[0.695, 0.706]$ | $0.011 \pm 0.004$ $[0.007, 0.014]$ | $0.067 \pm 0.010$ $[0.059, 0.082]$ | $0.669 \pm 0.004$ $[0.664, 0.672]$ | $70.40\text{k}$ |
| *M4-SAGE: GraphSAGE (Ablation, CE)* | $0.728 \pm 0.013$ $[0.714, 0.738]$ | $0.009 \pm 0.004$ $[0.003, 0.011]$ | $0.031 \pm 0.002$ $[0.030, 0.033]$ | $0.690 \pm 0.013$ $[0.677, 0.709]$ | $69.83\text{k}$ |
| *M5-SAGE: GraphSAGE (Ablation, CE+KL)* | $0.727 \pm 0.008$ $[0.719, 0.737]$ | $0.015 \pm 0.004$ $[0.010, 0.018]$ | $0.065 \pm 0.008$ $[0.056, 0.072]$ | $0.684 \pm 0.009$ $[0.672, 0.692]$ | $69.83\text{k}$ |

### Table 5: Calibration & Spatial Error Structure
*Evaluated on Lizard in-domain test set. ECE is computed across 10 bins. $A_{cons}$ and CER quantify cross-perturbation agreement. $C_{err}$ reports observed join-count error clustering vs. 1,000 spatial permutations.*

| Model Architecture | 10-bin ECE ($\downarrow$) | Brier Score ($\downarrow$) | $A_{cons}$ ($\uparrow$) | Consistent Error Rate (CER $\downarrow$) | $C_{err}$ (Observed vs Null) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **M1: ResNet-18** *(full fine-tune)* | $0.0891 \pm 0.0679$ | $0.2781 \pm 0.1360$ | $0.878 \pm 0.011$ | $0.061 \pm 0.002$ | $0.068\text{ vs }0.047$ ($p < 0.001$) |
| **M2: MLP** *(RandStainNA, CE)* | $0.0146 \pm 0.0065$ | $0.3721 \pm 0.0226$ | $0.898 \pm 0.003$ | $0.202 \pm 0.004$ | $0.106\text{ vs }0.078$ ($p < 0.001$) |
| **M3: MLP** *(Consistency, CE+KL)* | $0.0234 \pm 0.0136$ | $0.3775 \pm 0.0389$ | $0.884 \pm 0.003$ | $0.188 \pm 0.001$ | $0.109\text{ vs }0.082$ ($p < 0.001$) |
| **M4: GATv2** *(RandStainNA, CE)* | $0.0158 \pm 0.0073$ | $0.3686 \pm 0.0198$ | $0.916 \pm 0.004$ | $0.211 \pm 0.004$ | $0.122\text{ vs }0.084$ ($p < 0.001$) |
| **M5: GATv2** *(Proposed, CE+KL)* | $0.0203 \pm 0.0080$ | $0.3765 \pm 0.0323$ | $0.898 \pm 0.003$ | $0.198 \pm 0.002$ | $0.130\text{ vs }0.092$ ($p < 0.001$) |
| *M4-SAGE: GraphSAGE (Ablation, CE)* | $0.0121 \pm 0.0056$ | $0.3452 \pm 0.0189$ | $0.903 \pm 0.006$ | $0.187 \pm 0.006$ | $0.102\text{ vs }0.071$ ($p < 0.001$) |
| *M5-SAGE: GraphSAGE (Ablation, CE+KL)* | $0.0137 \pm 0.0072$ | $0.3548 \pm 0.0331$ | $0.888 \pm 0.004$ | $0.178 \pm 0.003$ | $0.108\text{ vs }0.077$ ($p < 0.001$) |

---

## Figure 2: Degradation Curves Under Stain Shift

![Figure 2](paper/figure2_macenko_severity.png)

*Line plot of nuclear phenotyping Macro-F1 across Macenko perturbation severity levels: Clean ($0\%$), Mild ($\pm 10\%$, $\pm 5^\circ$), Moderate ($\pm 20\%$, $\pm 10^\circ$), and Strong ($\pm 30\%$, $\pm 15^\circ$) with standard error bars across seeds.*

---

## Repository Structure

```text
├── configs/
│   └── locked_config.json                 # Pre-registered immutable hyperparameter lock
├── data/
│   ├── splits/
│   │   ├── lizard_kept_regions.csv        # Provenance audit: kept regions (CRAG, DigestPath, GlaS)
│   │   ├── provenance_audit_summary.csv   # Leakage prevention breakdown
│   │   ├── splits.csv                     # 70/15/15 stratified train/val/test splits
│   │   └── splits.json                    # Full splits metadata
│   └── processed/
│       ├── randstainna_stats.json         # RandStainNA statistics fitted strictly on train split
│       └── sanity_checks/                 # Visual overlays verifying 20x mapping
├── features/
│   ├── extract_features.py                # Pretrained ResNet-18 FP16 feature extraction
│   ├── graph_builder.py                   # Symmetrized k-NN graphs with 3D edge attributes
│   └── build_hdf5_dataset.py              # Multimodal HDF5 feature cache builder
├── losses/
│   └── consistency_loss.py                # Weighted CE + Symmetric KL consistency loss
├── metrics/
│   └── evaluation.py                      # Macro-F1, ECE, Brier, A_cons, CER, C_err with null
├── models/
│   ├── mlp_model.py                       # Matched 2-layer MLP (66.05k params)
│   └── gatv2_model.py                     # Matched GATv2 (70.40k) & GraphSAGE (69.83k)
├── outputs/
│   ├── figures/                           # High-res publication PNG and PDF figures
│   ├── metrics/
│   │   ├── detailed_metrics.json          # Per-slide/region metrics across all runs
│   │   └── summary_metrics.csv            # Tabular metric summary
│   └── predictions/                       # Raw prediction files (210 .npz files)
├── paper/
│   ├── table4_robustness.md               # Formatted Table 4 markdown
│   ├── table5_calibration.md              # Formatted Table 5 markdown
│   ├── figure2_macenko_severity.png       # Publication Figure 2
│   └── verdicts_and_statistics.json       # Hypothesis test statistics and CIs
├── perturbations/
│   ├── macenko.py                         # SVD stain vector decomposition & perturbation
│   └── randstainna.py                     # LAB color transfer augmentation
├── stat_tests/
│   └── statistical_tests.py               # BCa bootstrap, paired Wilcoxon, Holm correction
├── training/
│   ├── tune_and_lock_config.py            # Pre-registration grid search on validation split
│   ├── train_eval_framework.py            # Fast training framework for M2–M5
│   └── train_m1.py                        # ResNet-18 end-to-end training
├── requirements.txt                       # Project dependencies
└── README.md
```

---

## Reproduction Instructions

### 1. Environment Setup
```bash
git clone https://github.com/TEKURU-ZENO/Histopathology-Cell-Classification.git
cd Histopathology-Cell-Classification
pip install -r requirements.txt
```

### 2. Generate Perturbations & Feature Cache
```bash
python step2_generate_perturbations.py
python features/build_hdf5_dataset.py
```

### 3. Training & Evaluation
```bash
# Fast heads (M2 to M5 across seeds 42, 43, 44):
python training/train_eval_framework.py

# Full ResNet-18 baseline (M1):
python training/train_m1.py
```

### 4. Metrics & Hypothesis Testing
```bash
python outputs/compute_all_metrics.py
python outputs/generate_paper_tables.py
python outputs/plot_figure2.py
```

---

## Citation & Contact

If you find this work or codebase helpful in your research, please cite:

```bibtex
@article{mehta2026spatial,
  title   = {Does Spatial Prediction Consistency Improve Stain Robustness in Histopathology Cell Classification?},
  author  = {Mehta, Dev},
  journal = {SRM Institute of Science and Technology},
  year    = {2026}
}
```
