# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 16:26:15 2024

python ~/data/project/ppi_predict/dataprocess/uniproKB_domain_arrange.py  C:/Users/jianw/Desktop/Arath_domain.csv json_out file_count_out
"""

import pandas as pd
import numpy as np

import sys

domain_csv = sys.argv[1]
json_out = sys.argv[2]
file_out = sys.argv[3]


def delet_index(range_list,name):
    index_total=[]
    for index, value in enumerate(range_list):
        if value == name:
            index_total.append(index)
    return index_total


def include_index(range_list,name1,name2,name3):
    index_total=[]
    for index, value in enumerate(range_list):
        if (value.find(name1) == -1 and value.find(name2) == -1 and value.find(name3) == -1) :
            index_total.append(index)
    return index_total


def key_index(range_list,name1):
    index_total=[]
    for index, value in enumerate(range_list):
        if (value.find(name1) != -1) :
            index_total.append(index)
    return index_total

def find_mutil_fields(str,fields):
    positions = []
    for field in fields:
        position = str.find(field)
        if position != -1:
            positions.append(position)
    return positions


Coiled_coil={}
Compositional_bias={}
Region={}
Repeat={}
Domain={}

df_example = pd.read_csv(domain_csv)


for index,row in df_example.iterrows():
    name = row['Entry']
    
    if row['Coiled_coil'] is not np.nan:
        Coiled_coil_split_total  = (row['Coiled_coil'].split(';'))
        split_number = key_index(Coiled_coil_split_total,'COILED')
        Coiled_coil_split = [Coiled_coil_split_total[i].strip() for i in list(split_number)]
    else:
        Coiled_coil_split = ['COILED 0..50']
    
    if row['Compositional_bias'] is not np.nan:
        Compositional_bias_split_total  = (row['Compositional_bias'].split(';'))
        split_number = key_index(Compositional_bias_split_total,'COMPBIAS')
        Compositional_bias_split = [Compositional_bias_split_total[i].strip() for i in list(split_number)]
    else:
        Compositional_bias_split = ['COMPBIAS 0..50']
            
    if row['Domain_FT'] is not np.nan:
        Domain_FT_split_total  = (row['Domain_FT'].split(';'))
        split_number = key_index(Domain_FT_split_total,'DOMAIN')
        Domain_FT_split = [Domain_FT_split_total[i].strip() for i in list(split_number)]
    else:
        Domain_FT_split = ['DOMAIN 0..50']

    if row['Region'] is not np.nan:
        Region_split_total  = (row['Region'].split(';'))
        need_delte = include_index(Region_split_total,'REGION','note','=')
      
        for i in reversed(need_delte):
            del Region_split_total[i]
          
        #range_delet1 = [i-1 for i in delet_index(Region_split_total,' /note="Interaction with target DNA"')]
        range_delet2 = [i-1 for i in delet_index(Region_split_total,' /note="Disordered"')]
        split_number = key_index(Region_split_total,'REGION')
        split_number = list(set(list(split_number))-set(range_delet2))
        #split_number = list(set(list(split_number))-set(range_delet1)-set(range_delet2))
        Region_split = [Region_split_total[i].strip() for i in list(split_number)]
        
        if len(Region_split) == 0:
            Region_split = ['REGION 0..50']
        
    else:
        Region_split = ['REGION 0..50']


    if row['Repeat'] is not np.nan:
        Repeat_split_total  = (str(row['Repeat']).split(';'))      
        split_number = key_index(Repeat_split_total,'REPEAT')
        Repeat_split = [Repeat_split_total[i].strip() for i in list(split_number)]
    else:
        Repeat_split = ['REPEAT 0..50']
      
    ########################################################
    
    Coiled_coil[name] = Coiled_coil_split
    Compositional_bias[name] = Compositional_bias_split
    Region[name] = Region_split
    Repeat[name] = Repeat_split
    Domain[name] = Domain_FT_split
      


domain = {}

domain['Coiled_coil_doamin'] = Coiled_coil
domain['Compositional_bias_doamin'] = Compositional_bias
domain['Domain_doamin'] = Domain
domain['Region_doamin'] = Region
domain['Repeat_domain'] = Repeat


import json
import re

file_ = json.dumps(domain)
f2 = open(json_out, 'w')
f2.write(file_)
f2.close()


total_txt=[]

for name1 in list(Coiled_coil.keys()):
    
    Coiled_coil_total_count = 0
    Coiled_coil_count = 0
    
    Compositional_bias_total_count = 0
    Compositional_bias_count = 0
    
    Domain_total_count = 0
    Domain_count = 0
    
    Region_total_count = 0
    Region_count = 0
    
    Repeat_total_count = 0
    Repeat_count = 0
    
    for single in Coiled_coil[name1]:
        if single != 'COILED 0..50':
            Coiled_coil_total_count=Coiled_coil_total_count + 1

        need_position = len(find_mutil_fields(single,['..']))
        del_position = len(find_mutil_fields(single,['?','>','<']))

        if single != 'COILED 0..50' and need_position >0 and (del_position ==0):
            pro_start_end = (re.split(' |:', single)[-1]).split('..')
            if (int(pro_start_end[1]) - int(pro_start_end[0])) > 20:
                Coiled_coil_count = Coiled_coil_count+1
        
    
    for single in Compositional_bias[name1]:
        if single != 'COMPBIAS 0..50':
            Compositional_bias_total_count=Compositional_bias_total_count + 1

        need_position = len(find_mutil_fields(single,['..']))
        del_position = len(find_mutil_fields(single,['?','>','<']))

        if single != 'COMPBIAS 0..50' and need_position >0 and (del_position ==0):
            pro_start_end = (re.split(' |:', single)[-1]).split('..')
            if (int(pro_start_end[1]) - int(pro_start_end[0])) > 20:
                Compositional_bias_count = Compositional_bias_count+1            
    
    for single in Domain[name1]:
        if single != 'DOMAIN 0..50':
            Domain_total_count=Domain_total_count + 1

        need_position = len(find_mutil_fields(single,['..']))
        del_position = len(find_mutil_fields(single,['?','>','<']))

        if single != 'DOMAIN 0..50' and need_position >0 and (del_position ==0):
            pro_start_end = (re.split(' |:', single)[-1]).split('..')
            if (int(pro_start_end[1]) - int(pro_start_end[0])) > 20:
                Domain_count = Domain_count+1    

    for single in Region[name1]:
        if single != 'REGION 0..50':
            Region_total_count=Region_total_count + 1

        need_position = len(find_mutil_fields(single,['..']))
        del_position = len(find_mutil_fields(single,['?','>','<']))

        if single != 'REGION 0..50' and need_position >0 and (del_position ==0):
            pro_start_end = (re.split(' |:', single)[-1]).split('..')
            if (int(pro_start_end[1]) - int(pro_start_end[0])) > 20:
                Region_count = Region_count+1
    
    
    for single in Repeat[name1]:
        if single != 'REPEAT 0..50':
            Repeat_total_count = Repeat_total_count + 1

        need_position = len(find_mutil_fields(single,['..']))
        del_position = len(find_mutil_fields(single,['?','>','<']))

        if single != 'REPEAT 0..50' and need_position >0 and (del_position ==0):
            pro_start_end = (re.split(' |:', single)[-1]).split('..')
            if (int(pro_start_end[1]) - int(pro_start_end[0])) > 20:
                Repeat_count = Repeat_count+1    

    total_txt.append(str(name1)+','+str(Coiled_coil_total_count)+','+str(Coiled_coil_count)+','+
                     str(Compositional_bias_total_count)+','+str(Compositional_bias_count)+','+
                     str(Domain_total_count)+','+str(Domain_count)+','+
                     str(Region_total_count)+','+str(Region_count)+','+
                     str(Repeat_total_count)+','+str(Repeat_count))
    


f=open(file_out,"w")
f.write('#Entry'+','+'Coiled_coil_total_count'+','+'Coiled_coil_count'+
        ','+'Compositional_bias_total_count'+','+'Compositional_bias_count'+
        ','+'Domain_total_count'+','+'Domain_count'+
        ','+'Region_total_count'+','+'Region_count'+
        ','+'Repeat_total_count'+','+'Repeat_count'+'\n')

for line in total_txt:
    f.write(line+'\n')
f.close()
