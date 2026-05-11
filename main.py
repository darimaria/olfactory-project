import warnings
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")
import deepchem as dc
from rdkit import Chem
import pandas as pd
from molecule_processor import load_deepchem, convert_graph_data
from gnn import GraphNeuralNetwork, Classifier, train_model, evaluate_model
import torch
from torch_geometric.data import DataLoader

def main():
    print("Loading dataset...")
    dataset = load_deepchem("odorant_dataset.csv")
    dataset.complete_shuffle()
    print(f"Dataset loaded: {len(dataset)} molecules")
    print("Converting to graph data...")
    dataset = convert_graph_data(dataset)
    print("Building dataloaders...")
    train_dataset = dataset[:int(len(dataset)*0.8)]
    val_dataset = dataset[int(len(dataset)*0.8):]
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)
    gnn = Classifier(input_dim=30, hidden_dim=20, output_dim=20)
    train_model(gnn, train_loader, torch.optim.Adam(gnn.parameters(), lr=0.001), torch.nn.CrossEntropyLoss())
if __name__ == "__main__":
    main()
