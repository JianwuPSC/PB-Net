import Bio
from Bio.PDB import *
import sys
import importlib
import os

sys.path.append("/home/wuj/data/tools/masif/source")

from input_output.protonate import protonate
from default_config.masif_opts import masif_opts

#conda activate masif
#pdb_filename = '/home/wuj/data/genome/species_pdb/AFDB_Sac_baker_yeast/AF-P00560-F1-model_v4.pdb'
#protonated_file = '/home/wuj/data/tools/masif/sample/P00560_prot.pdb'
#protonated_file_chain_pdb = '/home/wuj/data/tools/masif/sample/P00560_prot_chain.pdb'
#protonated_file_name = '/home/wuj/data/tools/masif/sample/P00560_prot_chain'

pdb_filename = sys.argv[1]
protonated_file = sys.argv[2] 
protonated_file_chain_pdb = sys.argv[3]
protonated_file_name = sys.argv[4]

protonate(pdb_filename, protonated_file)

###########################

# Extract chains of interest.
from input_output.extractPDB import extractPDB
extractPDB(protonated_file, protonated_file_chain_pdb, 'A')

# Compute MSMS of surface w/hydrogens, 
from triangulation.computeMSMS import computeMSMS

vertices1, faces1, normals1, names1, areas1 = computeMSMS(protonated_file_chain_pdb, protonate=True)
    
# Compute "charged" vertices
from triangulation.computeCharges import computeCharges
vertex_hbond = computeCharges(protonated_file_name+'_chain', vertices1, names1)

# For each surface residue, assign the hydrophobicity of its amino acid.
from triangulation.computeHydrophobicity import computeHydrophobicity
vertex_hphobicity = computeHydrophobicity(names1)

# If protonate = false, recompute MSMS of surface, but without hydrogens (set radius of hydrogens to 0).
vertices2 = vertices1
faces2 = faces1

# # Fix the mesh.
from triangulation.fixmesh import fix_mesh
import pymesh

mesh = pymesh.form_mesh(vertices2, faces2)
regular_mesh = fix_mesh(mesh, masif_opts['mesh_res'])

# Compute the normals
from triangulation.compute_normal import compute_normal
vertex_normal = compute_normal(regular_mesh.vertices, regular_mesh.faces)
# Assign charges on new vertices based on charges of old vertices (nearest
# neighbor)

from triangulation.computeCharges import assignChargesToNewMesh

if masif_opts['use_hbond']:
    vertex_hbond = assignChargesToNewMesh(regular_mesh.vertices, vertices1, vertex_hbond, masif_opts)

if masif_opts['use_hphob']:
    vertex_hphobicity = assignChargesToNewMesh(regular_mesh.vertices, vertices1, vertex_hphobicity, masif_opts)

from triangulation.computeAPBS import computeAPBS
if masif_opts['use_apbs']:
    vertex_charges = computeAPBS(regular_mesh.vertices, protonated_file_chain_pdb, protonated_file_name)

import numpy as np
from input_output.read_ply import read_ply
from input_output.save_ply import save_ply
from sklearn.neighbors import KDTree

iface = np.zeros(len(regular_mesh.vertices))

if 'compute_iface' in masif_opts and masif_opts['compute_iface']:
    # Compute the surface of the entire complex and from that compute the interface.
    v3, f3, _, _, _ = computeMSMS(protonated_file, protonate=True)
    # Regularize the mesh
    mesh = pymesh.form_mesh(v3, f3)
    # I believe It is not necessary to regularize the full mesh. This can speed up things by a lot.
    full_regular_mesh = mesh
    # Find the vertices that are in the iface.
    v3 = full_regular_mesh.vertices
    # Find the distance between every vertex in regular_mesh.vertices and those in the full complex.
    kdt = KDTree(v3)
    d, r = kdt.query(regular_mesh.vertices)
    d = np.square(d) # Square d, because this is how it was in the pyflann version.
    assert(len(d) == len(regular_mesh.vertices))
    iface_v = np.where(d >= 2.0)[0]
    iface[iface_v] = 1.0
    # Convert to ply and save.
    save_ply(protonated_file_name+".ply", regular_mesh.vertices,\
                        regular_mesh.faces, normals=vertex_normal, charges=vertex_charges,\
                        normalize_charges=True, hbond=vertex_hbond, hphob=vertex_hphobicity,\
                        iface=iface)
