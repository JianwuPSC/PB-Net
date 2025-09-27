# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 10:31:15 2024

@author: jianw
"""

import numpy as np
from pathlib import Path
import pandas as pd
import random
import sys
import re

###### source pdb

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
          
###############################################################
### source single

############# target pdb

from Bio.PDB import PDBParser

def Acc_map(paths,pdb_paths):
    
    acc_dict={}
    pdb_dict={}
    parser = PDBParser()
    paths = list(Path(paths).iterdir())
    for path in paths:
        if str(path).find(str('.acc')) != -1:
            name=(str(path).split("/")[-1]).replace(".acc", "")
            acc_list = []
            with open(path, 'r') as file:
                lines = file.readlines()
                for line in lines:
                    line_sig = line.split()
                    acc_list.append(int(line_sig[3]))
            acc_dict[name] = acc_list
    
    pdb_paths = list(Path(pdb_paths).iterdir())
    for path in pdb_paths:
        if str(path).find(str('.pdb')) != -1:
            name = (str(path).split("/")[-1]).replace(".pdb", "")
            name1 = (str(name).split("-")[1])
            print(name)
            
            A_chain =[]
            structure = parser.get_structure('P1', str(path))
            for model in structure:
                for chain in model:
                    #if chain.id == "A":
                    for residue in chain:
                        for atom in residue:
                            if atom.name == 'CA':
                                A_chain.append(atom.coord)  
            pdb_dict[name1] = A_chain      
    
    return  acc_dict, pdb_dict


##################################################################


def Target_region(data_path, acc_dict, pdb_dict):
    
    target_dict={}
    target_cluster_dict={}
    
    with open(data_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            line = line.split()
            
            line_name = line[0]+'_'+line[1]+'_'+line[2]
            name = line[0]
            start = int(line[1])
            end = int(line[2])
            lst = acc_dict[name]
            max_value = max(acc_dict[name][start-1:end-1])
            max_values_index = [(x, index) for index, x in enumerate(lst) if x == max_value and index>=start-1 and index<end-1]
            max_value = max_values_index[0][0]
            max_indx = max_values_index[0][1]
            target_coor = pdb_dict[name][max_indx]
            
            target_dict[line_name] = [max_value, target_coor]
    
    for sig in list(target_dict.keys()):
        data_list = (str(sig).split("_"))
        name = data_list[0]
        if name not in target_cluster_dict.keys():
            taget_list = []
        taget_list.append(str(data_list[0])+'_'+str(data_list[1])+'_'+str(data_list[2]))
        target_cluster_dict[name] = taget_list
    
    return target_cluster_dict, target_dict
    

############################

dis_cluster_dict, dis_dict = Contact_map('/home/wuj/data/project/ppi_predict/Sac_baker/feature/af2_pdb','/home/wuj/data/genome/species_pdb/AFDB_Sac_baker_yeast/AF-P00546-F1-model_v4.pdb')



acc_dict, pdb_dict = Acc_map('/home/wuj/data/genome/species_pdb/AFDB_Sac_baker_yeast_mkdssp','/home/wuj/data/genome/species_pdb/AFDB_Sac_baker_yeast')

target_cluster_dict, target_dict = Target_region('/home/wuj/data/project/ppi_predict/Sac_baker/DSSP_align/zz.list',acc_dict,pdb_dict)

############################

sys.path.append("/home/wuj/data/tools/masif/source")

import pymesh
from geometry.compute_polar_coordinates import compute_polar_coordinates
from default_config.masif_opts import masif_opts
from masif_modules.read_data_from_surface import normalize_electrostatics, mean_normal_center_patch, compute_ddc

file_out = '/home/wuj/data/project/ppi_predict/Sac_baker/feature/af2_pdb/'

for exam in list(dis_cluster_dict.keys()):
    
    protonated_file_name = '/home/wuj/data/project/ppi_predict/Sac_baker/feature/APBS/'+exam
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

#######################################################



for exam in list(target_cluster_dict.keys()):
    
    protonated_file_name = '/home/wuj/data/project/ppi_predict/Sac_baker/feature/APBS/'+exam
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
    
    for exam_list_one in list(target_cluster_dict[exam]):
        point = target_dict[exam_list_one][1]
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
