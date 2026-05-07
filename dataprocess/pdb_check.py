from Bio.PDB import PDBParser
from Bio.PDB.PDBExceptions import PDBConstructionWarning
import warnings
import os

def check_pdb_integrity(pdb_file):
    
    parser = PDBParser()
    
    warnings.simplefilter('ignore', PDBConstructionWarning)
    
    for filename in os.listdir(pdb_file):
        pdb_input = ''.join([pdb_file,filename])
    
        try:
            # ??PDB??
            structure = parser.get_structure('test', pdb_input)
        
            # ??????
        except Exception as e:
            print(f'{filename} loss')

# ????
pdb_list = check_pdb_integrity("/home/wuj/data/genome/species_pdb/AFDB_Mouse/")
