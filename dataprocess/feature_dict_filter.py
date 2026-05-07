import pandas as pd
import numpy as np
import sys
import random


data_path = sys.argv[1]
key_word = sys.argv[2]
file_out = sys.argv[3]

####################

inter_example = pd.read_table(data_path,names=['source',"target"])

uniq_dict={}

for index,row in inter_example.iterrows():
    name = row['source']
    name_del = row['target']

    if name not in uniq_dict.keys():
        uniq_dict[name]= name_del

    elif name in uniq_dict.keys():
        if str(key_word) == "low" and float(name_del) < float(uniq_dict[name]):
           uniq_dict[name]= name_del

        elif str(key_word) == "low" and float(name_del) >= float(uniq_dict[name]):
           uniq_dict[name]= uniq_dict[name]

        if str(key_word) == "high" and float(name_del) > float(uniq_dict[name]):
           uniq_dict[name]= name_del

        elif str(key_word) == "high" and float(name_del) <= uniq_dict[name]:
           uniq_dict[name]= uniq_dict[name]

#file_out ='/home/wuj/data/genome/PPI/CCSB/3.txt'

f=open(file_out,"w")
for line in list(uniq_dict.keys()):
    f.write(line+'\t'+str(uniq_dict[line])+'\n')
f.close()
           
