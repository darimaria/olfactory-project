import deepchem as dc
import pandas as pd
import torch
from torch_geometric.data import Data
from tqdm import tqdm

_FEATURIZER = dc.feat.MolGraphConvFeaturizer(use_edges=True)

def smiles_to_graph(smiles, labels=None):
    graph = _FEATURIZER.featurize([smiles])[0]
    data = Data(
        x=torch.tensor(graph.node_features, dtype=torch.float),
        edge_index=torch.tensor(graph.edge_index, dtype=torch.long),
        edge_attr=torch.tensor(graph.edge_features, dtype=torch.float),
    )
    if labels is not None:
        data.y = torch.tensor(labels, dtype=torch.float).unsqueeze(0)
    data.batch = torch.zeros(data.x.shape[0], dtype=torch.long)
    return data

def load_dataset(csv_path):
    df = pd.read_csv(csv_path)
    scent_labels = df.columns[2:].tolist()
    dataset = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Featurizing molecules", unit="mol"):
        try:
            data = smiles_to_graph(row["nonStereoSMILES"], row[scent_labels].values)
            dataset.append(data)
        except Exception:
            pass
    return dataset, scent_labels
