"""
python ppi_predict/dataprocess/pdb_contact_map.py /home/wuj/data/project/ppi_predict/Sac_baker/rfdiffusion/sample/P35191/Domain/59-127_1.pdb /home/wuj/data/project/ppi_predict/Sac_baker/distance_map.txt /home/wuj/data/project/ppi_predict/Sac_baker/distance_8A.txt
"""
#data='/home/wuj/data/project/ppi_predict/Sac_baker/rfdiffusion/sample/P35191/Domain/59-127_1.pdb'
#file_out ='/home/wuj/data/project/ppi_predict/Sac_baker/distance_map.txt'
#file_out ='/home/wuj/data/project/ppi_predict/Sac_baker/distance_8A.txt'

import numpy as np
from pathlib import Path
import pandas as pd
import random
import sys
import re

dataset = sys.argv[1]
key_word = sys.argv[2]
file_out = sys.argv[3]
file_out1 = sys.argv[4]

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


A_chain = []
with open(dataset, 'r') as file:
    lines = file.readlines()
    for line in lines:
        if 'ATOM' in line.strip():
            b = line.split()
            if b[0] == "ATOM" and b[2] == "CA" and b[4] == "A" and is_number(b[6]) and is_number(b[7]) and is_number(b[8]):
                A_chain.append((float(b[6]),float(b[7]),float(b[8])))

B_chain = []
with open(dataset, 'r') as file:
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
      dist_8.append(str(b)+' '+str(c))
      
    elif dis > 8.:
      dis2.append(round(float(dis),3))
  dis1.append(dis2)

#file_out ='/home/wuj/data/project/ppi_predict/Sac_baker/distance_map.txt'
np.savetxt(file_out, np.c_[dis1],fmt='%.3f',delimiter='\t')

#dis_8A_out ='/home/wuj/data/project/ppi_predict/Sac_baker/distance_8A.txt'

f=open(file_out1,"w")
for i in range (len (dist_8)):
    f.write(key_word+'\t'+str(dist_8[i])+'\n')
f.close()
