import os
import sys
from Bio.PDB import *

directory = sys.argv[1]
out_path = sys.argv[2]
chain_id = sys.argv[3]

class ChainSelector(Select):
    def accept_chain(self, chain):
        return chain.id == chain_id

files = os.listdir(directory)
parser = PDBParser()

for index, file_ in enumerate(files):
    if str(file_).find(str('pdb')) != -1:

        structure = parser.get_structure("input", directory+'/'+file_)
        out_name = file_.replace("_af2pred", "")
        print(out_name)

        out_file = str(out_path)+'/'+out_name
        io = PDBIO()
        io.set_structure(structure)
        io.save(out_file, ChainSelector())
