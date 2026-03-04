import deepchem as dc
from rdkit import Chem
import pandas as pd

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

        
def main():
    df = pd.read_csv('odorant_molecules_cm.csv')
    # new_df, descriptor_names = vectorize_descriptors(df)
    # new_df.to_csv('odorant_molecules_vectorized.csv', index=False)
    df2, descriptor_names = vectorize_descriptors(df)
    # create a dictionary that maps each descriptor string to its index
    print(len(df2.iloc[0]['description_vector']))
    print(df.columns[2:-1])
    
if __name__ == "__main__":
    main()
