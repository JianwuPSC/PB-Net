import numpy as np
from pathlib import Path
import pandas as pd
import random

with open('/home/wuj/data/genome/SL4.0_protein/ITAG4.0_proteins.fasta') as fa:
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


#####################################################################
path='/home/wuj/data/project/ppi_predict/RIN/blast/mpnnseq_blastout.txt'
indentity_dict={}
evalue_dict={}

traget_length=397

with open(path) as blast_out:
    #fa_dict = {}
    for line in blast_out:
    # 去除末尾换行符
    # line = line.replace('\n','')
        if not line.startswith('#'):
            # 去除 > 号            
            name1=str(line).split("\t")[0]
            name2=str(line).split("\t")[1]
            # 去除末尾换行符并连接多行序列
            #indentity_dict[name1+'='+name2] = str(line).split("\t")[2]
            #evalue_dict[name1+'='+name2] = str(line).split("\t")[10]
            if float(line.split("\t")[2]) >40.0 or  float(str(line).split("\t")[10])<25.0: #filter
                evalue_dict[name1+'='+name2] =  float((line).split("\t")[10])
                indentity_dict[name1+'='+name2] =  float((line).split("\t")[2])              
            
        # filter2
        #if  not line.startswith('#') and indentity_dict[name1+'_'+name2] < str(line).split("\t")[2]:
        #    indentity_dict[name1+'_'+name2] = str(line).split("\t")[2]
               
        #if  not line.startswith('#') and evalue_dict[name1+'_'+name2] > str(line).split("\t")[10]:
        #    evalue_dict.get[name1+'_'+name2] = str(line).split("\t")[10]
        
        
######################################################################################

target_list=[]
for name in list(evalue_dict.keys()):    
    gene = str(name).split('=')[1]    
    af2_list = "/".join(['Solyc05g012020.4.1',gene])
    af2_list_total = " ".join([af2_list,str(traget_length+len(fa_dict[gene])),gene])    
    target_list.append(af2_list_total)              


indentity_list=[]
for name in list(evalue_dict.keys()):    
    gene = str(name).split('=')[1]    
    ID = str(name).split('=')[0]
    af2_list_total = " ".join([ID,gene,str(indentity_dict[name]),str(evalue_dict[name])])    
    indentity_list.append(af2_list_total) 

####################################################################################

file_out = '/home/wuj/data/project/ppi_predict/RIN/af2complex/RIN_target.lst'                
f=open(file_out,"w")
f.write('#model_name model_preset MSA_mode\n')
f.write('*model_3_multimer_v3 multimer_np all\n')
f.write('##Target(components) Size(AAs) Name(for output)\n')
for line in list(set(target_list)):
    f.write(line+'\n')
f.close() 


file_out = '/home/wuj/data/project/ppi_predict/RIN/af2complex/RIN_target_indentity.txt'                
f=open(file_out,"w")
f.write('#ID protein identity evalue\n')
for line in list(set(indentity_list)):
    f.write(line+'\n')
f.close() 

