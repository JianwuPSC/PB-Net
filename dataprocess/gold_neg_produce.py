import pandas as pd
import numpy as np
import sys
import random

path_a = sys.argv[1]
data_path = sys.argv[2]
data_path2 = sys.argv[3]
file_out = sys.argv[4]
signal_noise = 10

####################
fa_dict = {}
with open(path_a) as fa:    
    for line in fa:
        # 去除末尾换行符
        line = line.replace('\n','')
        if not line.startswith('#'):
            if line.startswith('>'):
                # 去除 > 号
                seq_name = line[1:]
                fa_dict[seq_name] = ''
            else:
                # 去除末尾换行符并连接多行序列
                fa_dict[seq_name] += line.replace('\n','')

######################

inter_example = pd.read_table(data_path,names=['source',"target"])
uniq_dict={}

for index,row in inter_example.iterrows():
    name = row['source']
    if name not in uniq_dict.keys():
        uniq_dict[name] = []
    uniq_dict[name].append(row['target'])



inter_example = pd.read_table(data_path2,names=['source',"target"])
target_dict={}

for index,row in inter_example.iterrows():
    name = row['source']
    if name not in target_dict.keys():
        target_dict[name] = []
    target_dict[name].append(row['target'])
    

other_dict={}

for row in target_dict.keys():
    source = row
    neg_set = list(set(list(fa_dict.keys())) - set(list(uniq_dict[source])))

    target_total = random.sample(neg_set, int(len(target_dict[source])*signal_noise))
    for i_sample in target_total:
        target = i_sample
        name = source+'-'+target
        name_del = target+'-'+source
        if name not in other_dict.keys() and name_del not in other_dict.keys():
            other_dict[name]=target
        else :
            neg_set1 = list(set(neg_set) - set(target_total) - set(target) - set(source))
            target_del = random.sample(neg_set1, 1)
            other_dict[name] = target_del

        
txt=[]
for name1 in list(other_dict.keys()):
    total = str(other_dict[name1])
    txt.append(name1)

#file_out ='/home/wuj/data/genome/PPI/CCSB/3.txt'

f=open(file_out,"w")
for line in txt:
    f.write(line+'\n')
f.close()
