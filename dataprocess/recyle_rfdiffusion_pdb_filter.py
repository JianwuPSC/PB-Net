from pathlib import Path
import pandas as pd
import random
from collections import Counter
import numpy as np
import math

path = '/home/wuj/data/project/ppi_predict/RIN/af2complex/protein_low_info.txt'

pdb_list_model = []
with open(path) as blast_out:
    for line in blast_out:
        if not line.startswith('#'):
            pdb_list = list(map(lambda x: '_'.join(x.split('_')[:-1]),
                                (line.split(' ')[1]).split(',')))[:-1]
            
            pdb_list_model.extend(pdb_list)
            
pdb_rf_count = Counter(pdb_list_model)

pdb_rf_count= {'design_ppi_0':100,'design_ppi_1':60,'design_ppi_2':10}
#### design_ppi_0:100
#### design_ppi_1:60


def nim_max_map(data,MIN,MAX):
    """
    归一化映射到任意区间
    :param data: 数据
    :param MIN: 目标数据最小值
    :param MAX: 目标数据最小值
    :return:
    """
    d_max = np.max(data)    # 当前数据最大值
    d_min = np.min(data)    # 当前数据最小值
    return MIN +(MAX-MIN)/(d_max-d_min) * (data - d_min)

rf_normal_count = list(map(int,nim_max_map(list(pdb_rf_count.values()),1,10)))
rf_normal_count_dict = {}

for idx,count in enumerate(pdb_rf_count.keys()):
    rf_normal_count_dict[count] = rf_normal_count[idx]
    # design_ppi_0:100
    # design_ppi_1:10
