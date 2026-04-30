import deepchem as dc
import pandas as pd
from torch_geometric.data import Data

def coulomb_matrix(smiles_molecule):
    molecule = Chem.MolFromSmiles(smiles_molecule)
    featurizer = dc.feat.CoulombMatrix(max_atoms=25)
    coulomb_matrix = featurizer.featurize([molecule])
    return coulomb_matrix

def vectorize_descriptors(df):
    # first get the descriptor names
    descriptor_names = df.columns[2:-1]
    rows, cols = df.shape
    new_df = pd.DataFrame([], columns=['molecule', 'coulomb_matrix', 'description_vector'])
    for i in range(rows):
        molecule = df.iloc[i]['nonStereoSMILES']
        cm = df.iloc[i]['coulomb_matrix']
        description_vector = [int(df.iloc[i][descriptor]) for descriptor in descriptor_names]
        new_df.loc[i] = {'molecule': molecule, 'coulomb_matrix': cm, 'description_vector': description_vector}

    return new_df, descriptor_names

def load_deepchem(csv_path):
    # put all of the labels of the scent columns into a list
    df = pd.read_csv(csv_path)
    scent_labels = df.columns[2:].tolist()
    
    # initialize a featurizer that will convert the smiles strings into graph data
    featurizer = dc.feat.MolGraphConvFeaturizer(use_edges=True)

    # load from csv
    loader = dc.data.CSVLoader(tasks=scent_labels, feature_field="nonStereoSMILES", featurizer=featurizer)

    # create a deepchem dataset
    dataset = loader.create_dataset(csv_path)

    return dataset

def convert_graph_data(dataset):
    graph_data = []
    # create geometric data object to give pyg gnn
    for sample in dataset:
        data = Data(x=sample.X, edge_index=sample.edge_index, edge_attr=sample.edge_features)
        graph_data.append(data)
    return graph_data

