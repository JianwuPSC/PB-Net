import numpy as np
from pathlib import Path
import pandas as pd
import random
import json
import os

path='/home/wuj/data/project/ppi_predict/RIN/af2complex/RIN_target_indentity.txt'
evalue_filter_dict = {}
evalue_dict = {}
traget_length=397

with open(path) as blast_out:
    #fa_dict = {}
    for line in blast_out:
    # 去除末尾换行符
    # line = line.replace('\n','')
        if not line.startswith('#'):
            # 去除 > 号            
            name1=str(line).split(" ")[0]
            name2=str(line).split(" ")[1]
            # 去除末尾换行符并连接多行序列
            evalue_filter_dict[name2] = ""
            evalue_dict[name2] = name1+'='+str((line).split(" ")[3])


with open(path) as blast_out:
    #fa_dict = {}
    for line in blast_out:
    # 去除末尾换行符
    # line = line.replace('\n','')
        if not line.startswith('#'):
            # 去除 > 号            
            name1=str(line).split(" ")[0]
            name2=str(line).split(" ")[1]
            
            evalue_filter_dict[name2] += name1+','

            if float(evalue_dict[name2].split('=')[1]) > float(line.split(" ")[3]):
                name1=str(line).split(" ")[0]
                evalue_dict[name2] = name1+'='+str((line).split(" ")[3])

###############################################################################
s = 'json'
dataset_path='/home/wuj/data/tools/af2complex/example/af2c_mod_examples'
paths = list(Path(dataset_path).iterdir())
protein_mutil=[]
protein_high_mutil=[]
protein_low_mutil=[]
  
for path in paths:
    for filewalks in os.walk(path):
        for files in filewalks[2]:
            #print('true files',files)
            if s in files:
               json_file = os.path.join(filewalks[0],files)
               
    protein_name = str(path).split('/')[-1]
    with open(json_file, 'r') as fcc_file:
        fcc_data = json.load(fcc_file)
        
    model =  list(fcc_data['interfacial residue number'].keys())[0]    
    interface_score = fcc_data['interface score'][model]
    interfacial_residue = fcc_data['interfacial residue number'][model]
    iptm_ptm = fcc_data['iptm+ptm'][model] # model confidence = 0.8 · ipTM + 0.2 · pTM
    pitm = fcc_data['pitms'][model] # Inerface pTM > 0.6
    plddt = fcc_data['plddts'][model]
    ptm = fcc_data['ptms'][model] # predicted TM-score 
    mpnn_name = (evalue_dict[protein_name]).split('=')[0]
    filter_name = (evalue_filter_dict[protein_name])

    ### output
    output=' '.join([protein_name,mpnn_name,model,interface_score,interfacial_residue,iptm_ptm,pitm,ptm,plddt])
    protein_mutil.append(output)
    
    ### filter high score
    if pitm > 0.7 and plddt > 0.8:
        high_output=' '.join([protein_name,mpnn_name,model,interface_score,interfacial_residue,iptm_ptm,pitm,ptm,plddt])
        protein_high_mutil.append(high_output)
        
    ### filter low score
    if pitm < 0.7 and plddt > 0.8:
        low_output=' '.join([protein_name,filter_name,model,interface_score,interfacial_residue,iptm_ptm,pitm,ptm,plddt])
        protein_low_mutil.append(low_output)    
    
    
###########################################################
#### output 
f=open("/home/wuj/data/project/ppi_predict/RIN/af2complex/protein_mutil_pdb_info.txt","w")
f.write('#proname mpnn_name model interface_score interfacial_residue iptm_ptm pitm ptm plddt\n')

for line in protein_mutil:
    f.write(line+'\n')
f.close()


f=open("/home/wuj/data/project/ppi_predict/RIN/af2complex/protein_high_info.txt","w")
f.write('#proname mpnn_name model interface_score interfacial_residue iptm_ptm pitm ptm plddt\n')

for line in protein_high_mutil:
    f.write(line+'\n')
f.close()

f=open("/home/wuj/data/project/ppi_predict/RIN/af2complex/protein_low_info.txt","w")
f.write('#proname mpnn_name model interface_score interfacial_residue iptm_ptm pitm ptm plddt\n')

for line in protein_low_mutil:
    f.write(line+'\n')
f.close()
