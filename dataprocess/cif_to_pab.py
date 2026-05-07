from Bio.PDB import *
import sys

input_cif = sys.argv[1]
output_pdb = sys.argv[2]

parser = MMCIFParser()
structure = parser.get_structure("input", input_cif)
io = PDBIO()
io.set_structure(structure)
io.save(output_pdb)

