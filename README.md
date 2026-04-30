# 🧪 Olfactory Project

A Graph Neural Network (GNN) for predicting odorant scent descriptors from molecular structure. Given a molecule's SMILES representation, the model learns to classify it across **138 scent categories** (e.g., floral, fruity, woody, smoky) using graph-based deep learning.

## Overview

Molecules are naturally represented as graphs — atoms are nodes, bonds are edges. This project leverages that structure by:

1. **Converting SMILES strings** into molecular graphs using DeepChem's `MolGraphConvFeaturizer`
2. **Learning atom-level embeddings** via a two-layer Graph Convolutional Network (GCN)
3. **Pooling** node embeddings into a single graph-level representation per molecule
4. **Classifying** each molecule into one or more scent descriptors (multi-label)

## Architecture

```
SMILES String
    │
    ▼
MolGraphConvFeaturizer (DeepChem)
    │
    ▼
┌─────────────────────────┐
│  GraphNeuralNetwork      │
│  ├─ GCNConv (in → hidden)│
│  ├─ ReLU                 │
│  └─ GCNConv (hidden → out)│
└─────────────────────────┘
    │
    ▼
Global Mean Pool (graph-level embedding)
    │
    ▼
Linear Classifier → 20 output classes
```

## Dataset

The project uses `odorant_dataset.csv`, containing **~4,983 molecules** with:

| Column | Description |
|---|---|
| `nonStereoSMILES` | Canonical SMILES string for the molecule |
| `descriptors` | Semicolon-separated human-readable scent labels |
| 138 binary columns | One-hot encoded scent categories (e.g., `floral`, `woody`, `sweet`) |

## Project Structure

```
olfactory-project/
├── main.py                 # Entry point — loads data, trains, and evaluates the model
├── gnn.py                  # GNN and Classifier model definitions, train/eval loops
├── molecule_processor.py   # Molecule featurization and data loading utilities
├── data_analysis.ipynb     # Exploratory data analysis notebook
├── odorant_dataset.csv     # Odorant molecule dataset (~5k molecules)
├── pyproject.toml          # Project metadata (uv/pip)
└── .python-version         # Python 3.14
```

### Module Breakdown

#### `molecule_processor.py`

| Function | Purpose |
|---|---|
| `load_deepchem(csv_path)` | Loads the CSV, featurizes SMILES into molecular graphs via `MolGraphConvFeaturizer`, and returns a DeepChem dataset |
| `convert_graph_data(dataset)` | Converts a DeepChem dataset into PyTorch Geometric `Data` objects |
| `coulomb_matrix(smiles)` | Computes a Coulomb matrix representation for a given SMILES string |
| `vectorize_descriptors(df)` | Extracts binary descriptor columns into a single vector per molecule |

#### `gnn.py`

| Class / Function | Purpose |
|---|---|
| `GraphNeuralNetwork` | Two-layer GCN that produces node-level embeddings |
| `Classifier` | Wraps the GNN with global mean pooling and a linear classification head |
| `train_model(...)` | Runs one epoch of training over a DataLoader |
| `evaluate_model(...)` | Evaluates model loss on a validation DataLoader |

#### `main.py`

Orchestrates the full pipeline:
1. Loads and featurizes the dataset
2. Splits into 80/20 train/validation
3. Creates PyTorch Geometric DataLoaders
4. Initializes the `Classifier` with Adam optimizer and CrossEntropyLoss
5. Trains the model

## Setup

### Prerequisites

- **Python 3.14+**
- [**uv**](https://docs.astral.sh/uv/) (recommended) or pip

### Installation

```bash
# Clone the repository
git clone https://github.com/darimaria/olfactory-project.git
cd olfactory-project

# Create a virtual environment and install dependencies with uv
uv venv
source .venv/bin/activate

# Install required packages
uv pip install torch torch-geometric deepchem rdkit pandas
```

Or with pip:

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch torch-geometric deepchem rdkit pandas
```

## Usage

### Train the model

```bash
python main.py
```

### Explore the data

Open `data_analysis.ipynb` in Jupyter:

```bash
uv pip install jupyter
jupyter notebook data_analysis.ipynb
```

## Dependencies

| Package | Purpose |
|---|---|
| [PyTorch](https://pytorch.org/) | Deep learning framework |
| [PyTorch Geometric](https://pyg.org/) | Graph neural network layers and data utilities |
| [DeepChem](https://deepchem.io/) | Molecular featurization and dataset loading |
| [RDKit](https://www.rdkit.org/) | Cheminformatics (SMILES parsing, molecular objects) |
| [pandas](https://pandas.pydata.org/) | Data manipulation and CSV handling |

## License

This project is for research and educational purposes.
