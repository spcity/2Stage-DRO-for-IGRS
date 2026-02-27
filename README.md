# 2Stage-DRO-for-IGRS
# Two-Stage Distributionally Robust Risk Quantification for Brain Tumor Segmentation

Official implementation of the two-stage DRO framework for conservative voxel-wise risk quantification in brain tumor segmentation.

## Overview

This framework addresses a critical question in clinical brain tumor segmentation: **"How much should we trust a segmentation result at each voxel?"**

We formulate risk quantification as a two-stage distributionally robust optimization (DRO) problem:

- **Stage 1 (Physical Prior Robustification):** Starting from a Fuzzy C-Means (FCM) soft segmentation of multi-modal MRI, we solve a per-voxel linear program (LP) over a Wasserstein ambiguity set to obtain a worst-case robust physical prior $P^*_{pve}(v)$.

- **Stage 2 (Consensus Risk Quantification):** Given $K$ deep learning segmentation models, we build an algorithmic consensus $\hat{P}_{seg}(v)$ and solve a second LP with dual Wasserstein constraints anchored to both the physical prior and the algorithmic consensus, yielding the final worst-case risk $R_{final}(v)$.

The key guarantee: $R_{final}(v) \geq C \cdot p(v)$ for all distributions $p$ within the Wasserstein ambiguity set, providing a **formal conservative upper bound** on tissue misclassification risk.

## Project Structure

```
.
├── common/                     # Core algorithmic modules
│   ├── lp_solvers.py           # Stage 1 & Stage 2 LP solvers (scipy HiGHS)
│   ├── wasserstein.py          # Wasserstein distance computation
│   ├── fcm.py                  # Fuzzy C-Means soft segmentation
│   ├── risk_params.py          # Risk cost vectors C and distance matrices D
│   └── test_solvers.py         # Unit tests for LP solvers
│
├── brainweb/                   # BrainWeb synthetic data experiments
│   ├── stage1/
│   │   ├── preprocess.py       # BrainWeb data preprocessing & PVE extraction
│   │   └── run_stage1.py       # Stage 1 pipeline for BrainWeb
│   ├── stage2/
│   │   ├── run_stage2.py       # Stage 2 pipeline for BrainWeb
│   │   └── segmentation_algorithms.py  # Simulated segmentation algorithms
│   └── analysis/
│       ├── stat_tables.py      # Statistical tables generation
│       ├── sensitivity_analysis.py     # Parameter sensitivity analysis
│       ├── cross_subject_analysis.py   # Cross-subject analysis
│       ├── visualize_pipeline.py       # Pipeline visualization
│       └── plot_utils.py       # Plotting utilities
│
├── brats2019/                  # BraTS2019 clinical data experiments
│   ├── stage1/
│   │   ├── build_pve_gt.py     # PVE ground truth construction from labels
│   │   └── run_stage1.py       # Stage 1 pipeline for BraTS2019
│   ├── stage2/
│   │   ├── run_inference.py    # Multi-model inference (6 architectures)
│   │   ├── build_consensus.py  # Consensus building from K models
│   │   └── run_stage2.py       # Stage 2 LP pipeline
│   ├── analysis/
│   │   ├── stat_tables.py      # Statistical tables and LaTeX generation
│   │   ├── benchmark_lp.py     # LP solver performance benchmarking
│   │   ├── run_reviewer_experiments.py  # Baseline comparisons & sensitivity
│   │   └── visualize_pipeline.py       # Visualization tools
│   ├── run_brats_pipeline.py   # End-to-end BraTS pipeline
│   ├── run_stage2_standalone.py # Standalone Stage 2 execution
│   └── run_ablation.py         # Ablation studies
│
├── .gitignore
└── README.md
```

## Method

### Stage 1: Physical Prior Robustification

Given an observed tissue distribution $q(v)$ from FCM and a PVE robustness radius $\varepsilon_1(v) = \alpha \cdot |\nabla I(v)| / \max|\nabla I|$:

$$R_1(v) = \max_{p: W(p, q) \leq \varepsilon_1} C^\top p$$

This LP has $N^2$ variables and $N+1$ constraints, solvable in ~0.8ms per voxel.

### Stage 2: Dual-Constrained Consensus Risk

Given algorithmic consensus $\hat{u}(v)$, physical prior $P^*(v)$, epistemic radius $\varepsilon_2$, and anchoring radius $\lambda$:

$$R_{final}(v) = \max_{p} C^\top p \quad \text{s.t.} \quad W(p, \hat{u}) \leq \varepsilon_2, \; W(p, P^*) \leq \lambda$$

This LP has $2N^2$ variables and $3N+2$ constraints, solvable in ~1.2ms per voxel.

### Key Parameters

| Parameter | Role | Typical Value |
|-----------|------|---------------|
| $\alpha$ | PVE robustness scaling | 0.05–0.50 |
| $\beta$ | Epistemic uncertainty weight | 0.50–2.00 |
| $C$ | Risk cost vector | [0, 0.5, 0.3, 1.0] (BraTS) |
| $D$ | Tissue distance matrix | Derived from clinical knowledge |

## Requirements

```
numpy>=1.20
scipy>=1.7       # HiGHS LP solver
joblib>=1.0      # Parallel batch LP solving
nibabel>=3.0     # NIfTI I/O (for BraTS experiments)
scikit-fuzzy     # FCM (optional, custom implementation included)
```

## Quick Start

### Run BrainWeb Experiment (Stage 1)

```bash
python -m brainweb.stage1.run_stage1 --subject 05 --alpha 0.20
```

### Run BraTS2019 Full Pipeline

```bash
# Stage 1: FCM + LP robustification
python -m brats2019.stage1.run_stage1 --alpha 0.20 --n-jobs 4

# Stage 2: Consensus building + LP risk quantification
python brats2019/run_stage2_standalone.py --alpha 0.20 --beta 1.00
```

### Run Unit Tests

```bash
python -m common.test_solvers
```

## Experimental Results

### Baseline Comparison (BraTS2019, N=17 patients, K=6 models)

| Method | Coverage | Conservatism |
|--------|----------|-------------|
| Soft Average (P̂\_seg) | 66.2±19.1% | 1.10× |
| Evidential (Dirichlet μ+2σ) | 68.4±18.6% | 1.14× |
| STAPLE (Majority Voting) | 78.7±11.3% | 1.11× |
| **Two-Stage DRO (Ours)** | **88.8±8.8%** | **1.63×** |

### Parameter Sensitivity

- **Cost vector C**: Scale-invariant (2× and 0.5× scaling: <0.2% coverage change); ordering perturbation: <1% change.
- **Distance matrix D**: ±30% scaling: <2% coverage change; far less influential than β (±10%).
- **Calibration**: β=0.50–0.75 achieves 90% target coverage across all α values.

## Computational Performance

| Metric | Stage 1 LP | Stage 2 LP |
|--------|-----------|-----------|
| Variables | N² = 16 | 2N² = 32 |
| Constraints | N+1 = 5 | 3N+2 = 14 |
| Time/voxel | ~0.8 ms | ~1.2 ms |
| Volume (128³, tumor) | ~80 s | ~120 s |
| Peak memory | ~200 MB | ~350 MB |

## Citation

*Paper under review. Citation information will be added upon acceptance.*

## License

This code is released for academic research purposes only. Please contact the authors for commercial use.
