import deepchem as dc
from rdkit import Chem
import pandas as pd
from molecule_processor import load_deepchem, convert_graph_data
from gnn import GraphNeuralNetwork, Classifier, train_model, evaluate_model
import torch
from torch_geometric.data import DataLoader

def main():
    dataset = load_deepchem("odorant_dataset.csv")
    dataset.shuffle()
    train_dataset = dataset[:int(len(dataset)*0.8)]
    val_dataset = dataset[int(len(dataset)*0.8):]
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)
    gnn = Classifier(input_dim=10, hidden_dim=20, output_dim=20)
    train_model(gnn, train_loader, torch.optim.Adam(gnn.parameters(), lr=0.001), torch.nn.CrossEntropyLoss())
    
if __name__ == "__main__":
    main()
