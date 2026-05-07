"""
python ~/data/project/ppi_predict/dataprocess/rfdiffusion_domain_input.py fasta json out_list
"""
import sys

fasta = sys.argv[1]
json_out = sys.argv[2]
file_out = sys.argv[3]


def find_mutil_fields(str,fields):
    positions = []
    for field in fields:
        position = str.find(field)
        if position != -1:
            positions.append(position)
    return positions

with open(fasta) as fa:
    fa_dict = {}
    for line in fa:
        # 去除末尾换行符
        line = line.replace('\n','')
        if line.startswith('>'):
            # 去除 > 号
            seq_name = line[1:]
            fa_dict[seq_name] = ''
        else:
            # 去除末尾换行符并连接多行序列
            fa_dict[seq_name] += line.replace('\n','')

#############################

import json

jsonData = json_out

with open(jsonData) as user_file:
  domain = json.load(user_file)


Coiled_coil = domain['Coiled_coil_doamin']
Compositional_bias = domain['Compositional_bias_doamin']
Domain = domain['Domain_doamin']
Region = domain['Region_doamin']
Repeat = domain['Repeat_domain']

import re
import numpy as np

total_txt=[]

for name1 in list(Coiled_coil.keys()):
    
    Coiled_coil_count = []
    Compositional_bias_count =[]
    Domain_count = []
    Region_count = []
    Repeat_count = []
    
    for single in Coiled_coil[name1]:
        
        need_position = len(find_mutil_fields(single,['..']))
        del_position = len(find_mutil_fields(single,['?','>','<']))
        
        if single != 'COILED 0..50' and need_position >0  and (del_position ==0):
            pro_start_end = (re.split(' |:', single)[-1]).split('..')
            if (int(pro_start_end[1]) - int(pro_start_end[0])) > 20:
                Coiled_coil_count.append(str(int(pro_start_end[0]))+'-'+str(int(pro_start_end[1])))
                
    if len(Coiled_coil_count)==0:
        Coiled_coil_count='None' 
    else:
        Coiled_coil_count=" ".join(Coiled_coil_count)
    
    ########################################
    
    for single in Compositional_bias[name1]:
        
        need_position = len(find_mutil_fields(single,['..']))
        del_position = len(find_mutil_fields(single,['?','>','<']))
        
        if single != 'COMPBIAS 0..50' and need_position >0 and (del_position ==0):
            pro_start_end = (re.split(' |:', single)[-1]).split('..')
            if (int(pro_start_end[1]) - int(pro_start_end[0])) > 20:
                Compositional_bias_count.append(str(int(pro_start_end[0]))+'-'+str(int(pro_start_end[1])))         
                
    if len(Compositional_bias_count)==0:
       Compositional_bias_count='None'     
    else:
       Compositional_bias_count=" ".join(Compositional_bias_count)
    ########################################
    
    for single in Domain[name1]:
        
        need_position = len(find_mutil_fields(single,['..']))
        del_position = len(find_mutil_fields(single,['?','>','<']))
        
        if single != 'DOMAIN 0..50' and need_position >0 and (del_position ==0):
            pro_start_end = (re.split(' |:', single)[-1]).split('..')
            if (int(pro_start_end[1]) - int(pro_start_end[0])) > 20:
                Domain_count.append(str(int(pro_start_end[0]))+'-'+str(int(pro_start_end[1])))
    
    if len(Domain_count)==0:
       Domain_count='None'     
    else:
       Domain_count=" ".join(Domain_count)
    ########################################
    
    for single in Region[name1]:
        
        need_position = len(find_mutil_fields(single,['..']))
        del_position = len(find_mutil_fields(single,['?','>','<']))
        
        if single != 'REGION 0..50' and need_position > 0 and (del_position ==0):
            pro_start_end = (re.split(' |:', single)[-1]).split('..')
            if (int(pro_start_end[1]) - int(pro_start_end[0])) > 20:
                Region_count.append(str(int(pro_start_end[0]))+'-'+str(int(pro_start_end[1])))
                
    if len(Region_count)==0:
       Region_count='None'    
    else:
       Region_count=" ".join(Region_count)
    ##########################################    
    
    for single in Repeat[name1]:
            
        need_position = len(find_mutil_fields(single,['..']))
        del_position = len(find_mutil_fields(single,['?','>','<']))
        
        if single != 'REPEAT 0..50' and need_position >0 and (del_position ==0):
            pro_start_end = (re.split(' |:', single)[-1]).split('..')
            if (int(pro_start_end[1]) - int(pro_start_end[0])) > 20:
                Repeat_count.append(str(int(pro_start_end[0]))+'-'+str(int(pro_start_end[1])))

    if len(Repeat_count)==0:
       Repeat_count='None' 
    else:
       Repeat_count=" ".join(Repeat_count)       
    ###########################################
        
    
    other = 'None'
    if Coiled_coil_count == 'None' and Compositional_bias_count == 'None' and Domain_count == 'None' and Region_count == 'None' and Repeat_count == 'None':
        other = '1-50'

    total_txt.append(str(name1)+','+str(len(fa_dict[name1]))+','+(Coiled_coil_count)+','+
                     (Compositional_bias_count)+','+
                     (Domain_count)+','+
                     (Region_count)+','+
                     (Repeat_count)+','+other)


############################################################################

f=open(file_out,"w")
f.write('#Entry'+','+'length'+','+'Coiled_coil_count'+','+'Compositional_bias_count'+','+
        'Domain_count'+','+'Region_count'+','+'Repeat'+','+'other'+'\n')
for line in  total_txt:
    f.write(line+'\n')
f.close()

