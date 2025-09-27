from Bio import PDB
import sys
from pathlib import Path

""" python pdb_extract.py dl_binder_design/proteinmpnn/P15646/ dl_binder_design/proteinmpnn/P15646.fa P15646  """

parser = PDB.PDBParser()
io = PDB.PDBIO()

nameOfFile_paths = sys.argv[1]
fileName_save = sys.argv[2]
key_word = sys.argv[3]

'''下面将氨基酸全称改为简写'''
longToShort = {'GLY': 'G', 'ALA': 'A', 'VAL': 'V', 'LEU': 'L', 'ILE': 'I', 'PHE': 'F', 'TRP': 'W', 'TYR': 'Y','ASP': 'D' ,'HIS': 'H', 'ASN': 'N', 'GLU': 'E', 'LYS': 'K', 'GLN': 'Q', 'MET': 'M', 'ARG': 'R', 'SER': 'S', 'THR': 'T', 'CYS': 'C', 'PRO': 'P', 'SEC': 'U', 'PYL': 'O'}

paths = list(Path(nameOfFile_paths).iterdir())
seq_name = {}
for path in paths:
    if str(path).find(str('.pdb')) != -1:
        name=(str(path).split("/")[-1]).replace(".pdb", "")
        struct = parser.get_structure('ID', str(path))
        aa = []
        for model in struct: #遍历所有model
            for chain in model: #遍历model中的所有chain
                if chain.id == "A":
                    for residue in chain:
                        resname = residue.get_resname()
                        aa.append(longToShort[resname])
        seq_name[key_word+'_'+name] = ''.join(aa)


with open(fileName_save, 'a+') as f_w:
    for ID in seq_name.keys():
        f_w.write('>'+ID+'\n')
        f_w.write(seq_name[ID]+'\n')
f_w.close()
