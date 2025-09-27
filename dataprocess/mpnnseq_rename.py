import numpy as np
from pathlib import Path
import pandas as pd
import random

import sys
import numpy as np

dataset_path = sys.argv[1]
file_output = sys.argv[2]
protein_name = sys.argv[3]

fa_dict={}

with open(dataset_path) as fa:
    #fa_dict = {}
    for line in fa:
    # 去除末尾换行符
        line = line.replace('\n','')
        if line.startswith('>'):
            # 去除 > 号
            name = line[1:]            
            seq_name = protein_name+name
            fa_dict[seq_name] = ''
        else:
            # 去除末尾换行符并连接多行序列
            fa_dict[seq_name] += line.replace('\n','')
txt=[]
for name in list(fa_dict.keys()):
    file_out = file_output
    txt.append('>'+name)
    txt.append(fa_dict[name])
    f=open(file_out,"w")
    for line in txt:
        f.write(line+'\n')
    f.close()
