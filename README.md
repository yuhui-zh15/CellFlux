# CellFlux: Simulating Cellular Morphology Changes via Flow Matching

<!-- ⚠️⚠️⚠️**Repo Under Construction**⚠️⚠️⚠️ -->

[![MIT license](https://img.shields.io/badge/License-MIT-blue.svg)](https://lbesson.mit-license.org/)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-311/)
[![Pytorch](https://img.shields.io/badge/Pytorch-2.5-red.svg)](https://pytorch.org/get-started/previous-versions/#v25)
[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)

This repo provides the PyTorch source code of our paper: [CellFlux: Simulating Cellular Morphology Changes via Flow Matching](https://arxiv.org/pdf/2502.09775) (ICML 2025). Check out project page [here](https://yuhui-zh15.github.io/CellFlux/)!

## 🔮 Abstract

Building a virtual cell capable of accurately simulating cellular behaviors in silico has long been a dream in computational biology. We introduce CellFlux, an image-generative model that simulates cellular morphology changes induced by chemical and genetic perturbations using flow matching. Unlike prior methods, CellFlux models distribution-wise transformations from unperturbed to perturbed cell states, effectively distinguishing actual perturbation effects from experimental artifacts such as batch effects—a major challenge in biological data. Evaluated on chemical (BBBC021), genetic (RxRx1), and combined perturbation (JUMP) datasets, CellFlux generates biologically meaningful cell images that faithfully capture perturbation-specific morphological changes, achieving a 35% improvement in FID scores and a 12% increase in mode-of-action prediction accuracy over existing methods. Additionally, CellFlux enables continuous interpolation between cellular states, providing a potential tool for studying perturbation dynamics. These capabilities mark a significant step toward realizing virtual cell modeling for biomedical research.

<img src="data/teaser.png"></img>
**Overview of CellFlux.**
(a) Objective. CellFlux aims to predict changes in cell morphology induced by chemical or gene perturbations in silico. In this example, the perturbation effect reduces the nuclear size. 
(b) Data. The dataset includes images from high-content screening experiments, where chemical or genetic perturbations are applied to target wells, alongside control wells without perturbations. Control wells provide prior information to contrast with target images, enabling the identification of true perturbation effects (e.g., reduced nucleus size) while calibrating non-perturbation artifacts such as batch effects—systematic biases unrelated to the perturbation (e.g., variations in color intensity). 
(c) Problem formulation. We formulate the task as a distribution-to-distribution problem (many-to-many mapping), where the source distribution consists of control images, and the target distribution contains perturbed images within the same batch. 
(d) Flow matching. CellFlux employs flow matching, a state-of-the-art generative approach for distribution-to-distribution problems. It learns a neural network to approximate a velocity field, continuously transforming the source distribution into the target by solving an ordinary differential equation (ODE). 
(e) Results. CellFlux significantly outperforms baselines in image generation quality, achieving lower Fr´echet Inception Distance (FID) and higher classification accuracy for mode-of-action (MoA) predictions.


## 🛠️ CellFlux Methods

Use ```bash example.sh``` to try our CellFlux methods!

## 💎 Capabilities

CellFlux enables accurate prediction of perturbation response, achieving tate-of-the-art performance on various datasets.
<div align="center">
    <img src="data/main_comparison.png" width="80%">
</div>

CellFlux unlocks new capa-bilities such as handling batch effects or visualizing cellular state transitions, significantly advancing the field towards a virtual cell for drug discovery and personalized therapy.
<div align="center">
    <img src="data/interpolation.png" width="80%">
</div>

## 🚀 Usage

### Environment

Create and activate the conda environment using the provided environment file:

```bash
conda env create -f environment.yml
conda activate cellflux
```

### Data

Three datasets (BBBC021, RxRx1, and JUMP/CPG0000) used in this project are same as [IMPA](https://github.com/theislab/IMPA). Pre-processed data are made available [here](https://zenodo.org/record/8307629).

Additionally, for RxRx1 and CPG0000, evaluation is performed on 100 random selected perturbations. The data index CSV files for evaluation can be downloaded from [Huggingface](https://huggingface.co/suyc21/CellFlux). 

In our implementation, we combined all the perturbation embeddings in CPG0000. The `combined_embeddings.csv` also needs to be downloaded [here](https://huggingface.co/suyc21/CellFlux). 


### Running CellFlux

#### 1. Configuration Setup

Before running CellFlux, update the configuration files in the `configs/` directory with your local paths:

For each dataset configuration file (`bbbc021_all.yaml`, `rxrx1.yaml`, `cpg0000.yaml`), update the following paths:

```yaml
# DIRECTORIES FOR DATA
image_path: /path/to/your/datasets/[dataset_name]
data_index_path: /path/to/your/datasets/[dataset_name]/metadata/[metadata_file].csv
embedding_path: /path/to/your/datasets/embeddings/[embedding_file].csv
```

#### 2. Training

**Quick Start with Example Script:**
```bash
bash scripts/example.sh
```

**Using Slurm for Distributed Training:**
```bash
# For BBBC021 dataset
bash scripts/slurm_bbbc021.sh

# For RxRx1 dataset  
bash scripts/slurm_rxrx1.sh

# For CPG0000 dataset
bash scripts/slurm_cpg0000.sh
```


#### 3. Evaluation with Pre-trained Checkpoints

Pretrained model checkpoints are provided at [Huggingface](https://huggingface.co/suyc21/CellFlux).

To quickly evaluate with specific checkpoints and generate images:

```bash
# Evaluate BBBC021
bash scripts/slurm_eval_bbbc021.sh

# Evaluate RxRx1
bash scripts/slurm_eval_rxrx1.sh

# Evaluate CPG0000
bash scripts/slurm_eval_cpg0000.sh
```

These scripts will:
- Load the specified checkpoint
- Generate sample images
- Calculate overall FID scores
- Save results to the given directory

#### 4. Detailed FID and KID Evaluation
For comprehensive evaluation metrics including detailed FID and KID results:

```bash
bash scripts/eval_fid.sh
```

This script will compute:
- Fréchet Inception Distance (FID)
- Kernel Inception Distance (KID)
- Detailed per-class metrics


## 🎯 Citation

If you use this repo in your research, please cite it as follows:
```
@inproceedings{CellFlux,
  title={CellFlux: Simulating Cellular Morphology Changes via Flow Matching},
  author={Zhang, Yuhui and Su, Yuchang and Wang, Chenyu and Li, Tianhong and Wefers, Zoe and Nirschl, Jeffrey and Burgess, James and Ding, Daisy and Lozano, Alejandro and Lundberg, Emma and Yeung-Levy, Serena},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2025}
}
```

## Acknowledgements

This repository is built upon the [Flow Matching](https://github.com/facebookresearch/flow_matching) framework. We gratefully acknowledge their foundational work that made this project possible.
