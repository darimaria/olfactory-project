import torch
from torch_geometric.nn import GCNConv, global_mean_pool

class GraphNeuralNetwork(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GraphNeuralNetwork, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x

class Classifier(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Classifier, self).__init__()
        self.gnn = GraphNeuralNetwork(input_dim, hidden_dim, output_dim)
        self.classifier = torch.nn.Linear(output_dim, output_dim)

    def forward(self, data):
        x = self.gnn(data)
        # Pool node embeddings into a single graph-level embedding per molecule
        x = global_mean_pool(x, data.batch)
        x = self.classifier(x)
        return x

def train_model(model, train_loader, optimizer, loss_fn):
    model.train()
    total_loss = 0
    for data in train_loader:
        optimizer.zero_grad()
        out = model(data)
        loss = loss_fn(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def evaluate_model(model, val_loader, loss_fn):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data in val_loader:
            out = model(data)
            loss = loss_fn(out, data.y)
            total_loss += loss.item()
    return total_loss / len(val_loader)
