import deepchem as dc
from rdkit import Chem
import pandas as pd
from molecule_processor import load_deepchem
        
def main():
    load_deepchem("odorant_dataset.csv")
    
if __name__ == "__main__":
    main()
