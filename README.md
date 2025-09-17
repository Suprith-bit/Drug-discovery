<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch"/>
  <img src="https://img.shields.io/badge/RDKit-Cheminformatics-00A859" alt="RDKit"/>
  <img src="https://img.shields.io/badge/License-MIT-43A047" alt="MIT"/>
</p>

<h1 align="center">🧬 Drug Discovery GAN</h1>
<p align="center">
  <b>Generate novel, valid, and diverse drug-like molecules using GANs + Reinforcement Learning</b>
</p>

<p align="center">
  <sub>Project by <a href="https://github.com/Suprith-bit">Suprith-bit</a></sub>
</p>

---

## 🚀 Overview

Drug Discovery GAN is a research-grade, end-to-end pipeline for de novo molecule generation. It combines:
- A sequence-based Generator (SMILES) built with LSTMs,
- A Discriminator with CNN feature extraction and a multi-head design (real/fake + property prediction),
- A training loop that supports supervised pretraining and GAN training enhanced with reinforcement signals,
- Robust evaluation (validity, uniqueness, novelty) and rich visualizations.

Core stack: PyTorch, RDKit, NumPy, Pandas, Matplotlib.

---

## ✨ Highlights

- 🎯 End-to-end workflow: data → training → generation → evaluation → visualization
- 🧠 Sequence Generator: `SMILESGenerator` (LSTM-based)
- 🧪 Discriminator with property head: `MolecularDiscriminator` predicts molecular properties alongside real/fake
- 🤝 Integrated model wrapper: `DrugGAN` orchestrates generator/discriminator and sampling
- 📊 Metrics: validity, uniqueness, novelty (`utils.molecular_metrics`)
- 🖼️ Visuals: Molecule grids, property distribution plots, training curves (`utils.visualization.MoleculeVisualizer`)
- ⚙️ Fully configurable via JSON (`configs/model_config.json`, `configs/training_config.json`)
- 🧰 Clean CLI in `src/main.py` to pretrain, train, generate, and evaluate

---

## 🧭 Architecture at a Glance

```mermaid
flowchart LR
    A[SMILES Dataset] --> B[Preprocess & Tokenize<br/>(SMILESPreprocessor)]
    B --> C[Generator (LSTM)<br/>SMILESGenerator]
    C --> D[Sample SMILES]
    D --> E[RDKit Validation<br/>Canonicalization]
    E --> F[Discriminator (CNN + Heads)<br/>MolecularDiscriminator]
    F -->|Rewards, Properties| C
    F --> G[Metrics & Plots<br/>(Validity, Uniqueness, Novelty)]
```

---

## 🖼️ Visual Gallery

Below images are saved by the pipeline inside `data/results/<experiment_name>/plots/`.
You can keep a stable name like `example_experiment` to have persistent links:

<p align="center">
  <img src="data/results/example_experiment/plots/sample_molecules.png" alt="Sample Generated Molecules" width="48%"/>
  <img src="data/results/example_experiment/plots/qed_distribution.png" alt="QED Distribution" width="48%"/>
</p>

<p align="center">
  <img src="data/results/example_experiment/plots/molwt_distribution.png" alt="Molecular Weight Distribution" width="48%"/>
  <img src="data/results/example_experiment/plots/logp_distribution.png" alt="LogP Distribution" width="48%"/>
</p>

<p align="center">
  <img src="data/results/example_experiment/plots/chemical_space_pca.png" alt="Chemical Space (PCA)" width="60%"/>
</p>

Training curves:

<p align="center">
  <img src="data/results/example_experiment/plots/generator_pretraining_loss.png" alt="Generator Pretraining Loss" width="48%"/>
  <img src="data/results/example_experiment/plots/discriminator_pretraining_loss.png" alt="Discriminator Pretraining Loss" width="48%"/>
</p>

<p align="center">
  <img src="data/results/example_experiment/plots/gan_training_history.png" alt="GAN Training History" width="60%"/>
</p>

---

## 🗂️ Project Structure

```
src/
  ├── data/
  │   ├── data_loader.py         # Load ZINC/CSV, split, property calc
  │   └── __init__.py            # SMILESDataset, Preprocessor, augmentation
  ├── models/
  │   ├── generator.py           # LSTM SMILESGenerator
  │   ├── discriminator.py       # CNN feature extractor + property head
  │   └── gan.py                 # DrugGAN wrapper (save/load, sampling)
  ├── training/
  │   ├── supervised.py          # Pretrain generator/discriminator
  │   └── reinforcement.py       # GAN + RL training loop
  ├── utils/
  │   ├── molecular_metrics.py   # validity, uniqueness, novelty
  │   └── visualization.py       # Molecule grids & plots
  └── main.py                    # CLI entrypoint
configs/
data/
  ├── raw/                       # Input SMILES (e.g., ZINC)
  ├── processed/
  └── results/                   # Experiments, models, plots
```

---

## ⚡ Quickstart

- Install dependencies:
```bash
pip install -r requirements.txt
```

- Train with GAN + RL:
```bash
python src/main.py --train_gan
```

- Generate molecules with a trained model:
```bash
python src/main.py --generate_only --load_gan data/results/example_experiment/models/trained_gan.pt
```

- Useful flags (see `src/main.py`):
  - `--pretrain_generator`, `--pretrain_discriminator`, `--train_gan`
  - `--num_molecules`, `--temperature`, `--device`
  - `--config`, `--training_config`, `--experiment_name`

---

## 🔍 What’s Inside the Models

- Generator: `SMILESGenerator` (LSTM)
  - Embedding → LSTM (n_layers) → Linear → Token distribution
  - Temperature-controlled multinomial sampling for diversity

- Discriminator: `MolecularDiscriminator`
  - `MolecularCNN` feature extractor over token sequences
  - Heads:
    - Real/Fake classifier (GAN signal)
    - Property predictor (`PropertyPredictor`) for learning chemically meaningful features

- Integrated: `DrugGAN`
  - Orchestrates training, sampling, save/load
  - Returns valid canonical SMILES using RDKit

---

## 📊 Evaluation

Key metrics from `utils.molecular_metrics`:
- Validity: fraction of syntactically/chemically valid SMILES
- Uniqueness: fraction of unique valid molecules
- Novelty: fraction of valid unique molecules not present in training data

Outputs are written as:
- `data/results/<experiment_name>/molecules/generated_molecules.csv`
- `data/results/<experiment_name>/molecules/evaluation_results.json`

---

## 🧑‍💻 What I Did (Resume-Ready)

- Designed and implemented a full deep learning pipeline to generate de novo drug-like molecules using GANs with reinforcement objectives, in PyTorch and RDKit.
- Built an LSTM-based SMILES generator and a CNN-based discriminator with dual heads for adversarial training and molecular property prediction.
- Engineered data ingestion, validation, and preprocessing for large SMILES datasets, including canonicalization and augmentation.
- Implemented rigorous evaluation (validity, uniqueness, novelty) and integrated RDKit property computations for scientific relevance.
- Automated visualization of results including molecule grids, property distributions (QED, MolWt, LogP), training curves, and chemical space projections.
- Developed a clean CLI with configurable JSONs enabling reproducible experiments, pretrained model loading, and controlled sampling.

---

## 🧰 Tech Stack

- Languages: Python
- ML/DL: PyTorch
- Cheminformatics: RDKit
- Data/Plots: NumPy, Pandas, Matplotlib

---

## 📄 License

MIT License. See [LICENSE](LICENSE) for details.

---
