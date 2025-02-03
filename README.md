# GNN-Protein-Ligand-Prediction-Drug-Discovery

This project implements a **Graph Neural Network (GNN)** to predict protein-ligand binding interactions. It utilizes molecular SMILES strings and protein sequences to generate graph-based representations and trains a model using **PyTorch Geometric**.

## Features
- **Graph Representation**: Converts molecules (SMILES) and proteins into graph structures.
- **Deep Learning Model**: Utilizes a **GNN-based architecture** for classification.
- **K-Fold Cross-Validation**: Implements **5-fold cross-validation** for robust evaluation.
- **Early Stopping**: Prevents overfitting by monitoring validation loss.
- **AWS Deployment**: (Optional) Model can be deployed on AWS SageMaker for inference.

## Dataset
The dataset consists of:
- **Molecule SMILES**: Representing chemical compounds.
- **Protein Sequences**: Encoded using feature extraction techniques.
- **Binding Labels**: Binary labels (binds: 1, does not bind: 0).

## Installation
To set up the project, install dependencies:

```bash
pip install torch torchvision torchaudio torch-geometric torch-scatter torch-sparse torch-cluster torch-spline-conv rdkit numpy pandas scikit-learn
```
## Train the Model
Run the training script to train your GNN model:

```bash
python train.py
```
This script:

- Loads the dataset
- Trains the GNN model using k-fold cross-validation
- Saves the best model
