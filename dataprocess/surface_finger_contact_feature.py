
"""
python /home/wuj/data/project/ppi_predict/dataprocess/surface_finger_contact_feature.py /home/wuj/data/project/ppi_predict/Sac_baker/feature/af2_pdb/ /home/wuj/data/genome/species_pdb/AFDB_Sac_baker_yeast/AF-P00546-F1-model_v4.pdb /home/wuj/data/project/ppi_predict/Sac_baker/feature/APBS/ /home/wuj/data/project/ppi_predict/Sac_baker//feature/af2_pdb/
### output P00546_8-295_9_dldesign_9_af2pred_surface.txt
"""
import numpy as np
from pathlib import Path
import pandas as pd
import random
import sys
import re
from Bio.PDB import PDBParser

###### source pdb

af2_pdb_path = sys.argv[1]
AFDB_pdb = sys.argv[2]
sly_path = sys.argv[3]
file_out = sys.argv[4]



def is_number(s):
    try:  # 如果能运行float(s)语句，返回True（字符串s是浮点数）
        float(s)
        return True
    except ValueError:  # ValueError为Python的一种标准异常，表示"传入无效的参数"
        pass  # 如果引发了ValueError这种异常，不做任何事情（pass：不做任何事情，一般用做占位语句）
    try:
        import unicodedata  # 处理ASCii码的包
        unicodedata.numeric(s)  # 把一个表示数字的字符串转换为浮点数返回的函数
        return True
    except (TypeError, ValueError):
        pass
    return False


def Contact_map(paths, pdb_path):
    
    dis_dict={}
    dis_cluster_dict={}
    parser = PDBParser()

    C_chain =[]
    structure = parser.get_structure('P1', str(pdb_path))
    for model in structure:
        for chain in model:
            #if chain.id == "A":
            for residue in chain:
                for atom in residue:
                    if atom.name == 'CA':
                        C_chain.append(atom.coord)
                            
    paths = list(Path(paths).iterdir())
    for path in paths:
        if str(path).find(str('.pdb')) != -1:
            name = (str(path).split("/")[-1]).replace(".pdb", "")
            int_cut = int((str(name).split("_")[1]).split('-')[0])
            
            A_chain = []
            with open(path, 'r') as file:
                lines = file.readlines()
                for line in lines:
                    if 'ATOM' in line.strip():
                        b = line.split()
                        if b[0] == "ATOM" and b[2] == "CA" and b[4] == "A" and is_number(b[6]) and is_number(b[7]) and is_number(b[8]):
                            A_chain.append((float(b[6]),float(b[7]),float(b[8])))
            B_chain = []
            with open(path, 'r') as file:
                lines = file.readlines()
                for line in lines:
                    if 'ATOM' in line.strip():
                        b = line.split()
                        if b[0] == "ATOM" and b[2] == "CA" and b[4] == "B" and is_number(b[6]) and is_number(b[7]) and is_number(b[8]):
                            B_chain.append((float(b[6]),float(b[7]),float(b[8])))            
            dis1 = []
            for b in range(len(A_chain)):
                dis2 = []
                for c in range(len(B_chain)):
                    dis = ((A_chain[b][0] - B_chain[c][0]) ** 2 + (A_chain[b][1] - B_chain[c][1]) ** 2 + (A_chain[b][2] - B_chain[c][2]) ** 2) ** 0.5
                    dis2.append(dis)
                dis1.append(dis2)
                        
            dis_arry = np.array(dis1)
            near = int(np.where(dis_arry==np.min(dis_arry))[1]) + int_cut - 1
            disarray_min = round(float(np.min(dis_arry)),3)
            min_coor = C_chain[near]
            dis_dict[name] = [disarray_min, min_coor]
            
    for keys_name in list(dis_dict.keys()):
        name=(str(keys_name).split("_")[0])
        if name not in dis_cluster_dict.keys():
            keys_name_list = []
        keys_name_list.append(keys_name)
        dis_cluster_dict[name] =  keys_name_list
    
    return  dis_cluster_dict, dis_dict

####################################################

dis_cluster_dict, dis_dict = Contact_map(af2_pdb_path,AFDB_pdb)

sys.path.append("/home/wuj/data/tools/masif/source")
import pymesh
from geometry.compute_polar_coordinates import compute_polar_coordinates
from default_config.masif_opts import masif_opts
from masif_modules.read_data_from_surface import normalize_electrostatics, mean_normal_center_patch, compute_ddc

for exam in list(dis_cluster_dict.keys()):
    
    protonated_file_name = sly_path+'/'+exam
    mesh = pymesh.load_mesh(protonated_file_name+".ply")
    vertex = mesh.vertices
    n1 = mesh.get_attribute("vertex_nx")
    n2 = mesh.get_attribute("vertex_ny")
    n3 = mesh.get_attribute("vertex_nz")
    normals = np.stack([n1,n2,n3], axis=1)
    mesh.add_attribute("vertex_mean_curvature")
    H = mesh.get_attribute("vertex_mean_curvature")
    mesh.add_attribute("vertex_gaussian_curvature")
    K = mesh.get_attribute("vertex_gaussian_curvature")
    elem = np.square(H) - K
    # In some cases this equation is less than zero, likely due to the method that computes the mean and gaussian curvature.
    # set to an epsilon.
    elem[elem<0] = 1e-8
    k1 = H + np.sqrt(elem)
    k2 = H - np.sqrt(elem)
    # Compute the shape index 
    si = (k1+k2)/(k1-k2)
    si = np.arctan(si)*(2/np.pi)
    # Normalize the charge.
    charge = mesh.get_attribute("vertex_charge")
    charge = normalize_electrostatics(charge)
    # Hbond features
    hbond = mesh.get_attribute("vertex_hbond")
    # Hydropathy features
    # Normalize hydropathy by dividing by 4.5
    hphob = mesh.get_attribute("vertex_hphob")/4.5

    rho, theta, neigh_indices, mask = compute_polar_coordinates(mesh, radius=masif_opts['ppi_search']['max_distance'], max_vertices=masif_opts['ppi_search']['max_shape_size'])
    
    for exam_list_one in list(dis_cluster_dict[exam]):
        point = dis_dict[exam_list_one][1]
        point_dis = []
        for c in range(len(vertex)):
            dis = ((vertex[c][0] - point[0]) ** 2 + (vertex[c][1] - point[1]) ** 2 + (vertex[c][2] - point[2]) ** 2) ** 0.5
            point_dis.append(round(float(dis),3))
            
        vix = np.argmin(point_dis)
        
        input_feat = np.zeros((masif_opts['ppi_search']['max_shape_size'], 5))

        neigh_vix = np.array(neigh_indices[vix])
        # Compute the distance-dependent curvature for all neighbors of the patch. 
        patch_v = mesh.vertices[neigh_vix]
        patch_n = normals[neigh_vix]
        patch_cp = np.where(neigh_vix == vix)[0][0] # central point
        mask_pos = np.where(mask[vix] == 1.0)[0] # nonzero elements
        patch_rho = rho[vix][mask_pos] # nonzero elements of rho
        ddc = compute_ddc(patch_v, patch_n, patch_cp, patch_rho)

        input_feat[:len(neigh_vix), 0] = si[neigh_vix]
        input_feat[:len(neigh_vix), 1] = ddc
        input_feat[:len(neigh_vix), 2] = hbond[neigh_vix]
        input_feat[:len(neigh_vix), 3] = charge[neigh_vix]
        input_feat[:len(neigh_vix), 4] = hphob[neigh_vix]
        
        np.savetxt(file_out+'/'+exam_list_one+'_surface.txt', input_feat, fmt='%.3f',delimiter='\t')

