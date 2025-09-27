from Bio.PDB import PDBParser
from Bio.PDB.DSSP import DSSP
from Bio.PDB.DSSP import dssp_dict_from_pdb_file
import sys

data_path = sys.argv[1]
file_out = sys.argv[2]
key_word = sys.argv[3]

dssp_tuple = dssp_dict_from_pdb_file(data_path)
dssp_dict = dssp_tuple[0]
dict_valuesList = [dssp_dict[dict_key][1] for dict_key in dssp_dict]

dict_valuesList = ['C' if c_ == '-' else c_ for c_ in dict_valuesList]

f=open(file_out,"w")
f.write('>'+key_word+'\n')

for line in dict_valuesList:
    f.write(line)
f.write('\n')
f.close()
