import numpy as np
from pathlib import Path
import pandas as pd
import random
import sys
import re

dataset_path = sys.argv[1]
file_out = sys.argv[2]
key_word = sys.argv[3]

paths = list(Path(dataset_path).iterdir())


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

dis_dict={}
count_dict={}
near_dict={}

for path in paths:
    
    if str(path).find(str(key_word)) != -1:
        name=(str(path).split("/")[-1]).replace(".pdb", "")
        
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

        count=0
        dis1 = []
        dist_8 = []
        
        for b in range(len(A_chain)):
          dis2 = []
          for c in range(len(B_chain)):
            dis = ((A_chain[b][0] - B_chain[c][0]) ** 2 + (A_chain[b][1] - B_chain[c][1]) ** 2 + (A_chain[b][2] - B_chain[c][2]) ** 2) ** 0.5
            if dis == 0.:
              dis2.append(0)
            elif (dis <= 8.) & (dis != 0.):
              dis2.append(round(float(dis),3))
              count=count+1
              dist_8.append(str(b)+'-'+str(c))
              
            elif dis > 8.:
              dis2.append(round(float(dis),3))
          dis1.append(dis2)
          
          dis_dict[name] = dis1
          count_dict[name] = count
          near_dict[name] = dist_8

####################
f=open(file_out,"w")
for keys in count_dict.keys():
    f.write(keys+'\t'+str(count_dict[keys])+'\n')
f.close()
