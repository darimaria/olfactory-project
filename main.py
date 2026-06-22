import warnings
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")
print("Loading libraries...", flush=True)
import pandas as pd
from molecule_processor import smiles_to_graph, load_dataset
from gnn import GraphNeuralNetwork, Classifier, train_model, evaluate_model
import torch
from torch_geometric.data import DataLoader

def main():
    print("Loading dataset...", flush=True)
    dataset, scent_labels = load_dataset("odorant_dataset.csv")
    print(f"Dataset loaded: {len(dataset)} molecules", flush=True)
    print("Building dataloaders...", flush=True)
    split = int(len(dataset) * 0.8)
    train_loader = DataLoader(dataset[:split], batch_size=32, shuffle=True)
    val_loader = DataLoader(dataset[split:], batch_size=32, shuffle=False)
    num_classes = len(scent_labels)
    gnn = Classifier(input_dim=30, hidden_dim=20, output_dim=20, num_classes=num_classes)
    optimizer = torch.optim.Adam(gnn.parameters(), lr=0.001)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    epochs = 20
    print(f"Training for {epochs} epochs...", flush=True)
    for epoch in range(1, epochs + 1):
        train_loss = train_model(gnn, train_loader, optimizer, loss_fn)
        val_loss = evaluate_model(gnn, val_loader, loss_fn)
        print(f"Epoch {epoch}/{epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}", flush=True)

    torch.save(gnn.state_dict(), "model.pth")
    print("Model saved to model.pth", flush=True)
if __name__ == "__main__":
    main()
