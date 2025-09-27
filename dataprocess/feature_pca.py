import pandas as pd
import numpy as np
import sys
import random
import tqdm

#python feature_pca.py sample_73_pca.txt high 1 zzz.out

data_path = sys.argv[1]
key_word = sys.argv[2]
col_word = sys.argv[3]
file_out = sys.argv[4]

####################
col_word = int(col_word)

with open(data_path, "r", encoding="utf-8") as f:
    lines = [line[:-1].split("\t")
            for line in tqdm.tqdm(f, desc="Loading blast", total=None) if not line.startswith('#')]
    corpus_lines = len(lines)

uniq_dict={}
dis_dict={}
compare_dict={}


def merge_intervals(intervals):
    # 按照区间的起始点排序
    intervals.sort(key=lambda x: x[0])
    
    merged = []
    for interval in intervals:
        # 如果合并列表为空，或者当前区间的起始点大于合并列表中最后一个区间的终点
        if not merged or merged[-1][1] < interval[0]:
            merged.append(interval)
        else:
            # 否则，合并当前区间到合并列表中最后一个区间
            merged[-1][1] = max(merged[-1][1], interval[1])
    
    return merged


for item in range(corpus_lines):
    name_all = lines[item][0]
    source = name_all.split('_')[0]
    target = name_all.split('_')[5]
    start = name_all.split('_')[6]
    end = name_all.split('_')[7]
    name = source+'_'+target
    
    if name not in dis_dict.keys():
        dis_dict[name] = [[int(start),int(end)]]

    else:
        dis_dict[name].append([int(start),int(end)])



for dis_key in dis_dict.keys():
    compare_dict[dis_key] = merge_intervals(dis_dict[dis_key])


for item in range(corpus_lines):
    name_all = lines[item][0]
    source = name_all.split('_')[0]
    target = name_all.split('_')[5]

    start = name_all.split('_')[6]
    end = name_all.split('_')[7]

    name = source+'_'+target
    sort_col = lines[item][col_word]
    for range_dis in list(compare_dict[name]):
        names = name+'_'+str(range_dis[0])+'_'+str(range_dis[1])
        if  names not in uniq_dict.keys() and int(start) >= range_dis[0] and int(end) <= range_dis[1]:
            name_del=[]
            for col in range(len(lines[item])):
                name_del.append(lines[item][col])
            uniq_dict[names]= name_del
        elif names in uniq_dict.keys() and  int(start) >= range_dis[0] and int(end) <= range_dis[1]:
            
            if str(key_word) == "low" and float(sort_col) < float(uniq_dict[names][col_word]):
                name_del=[]
                for col in range(len(lines[item])):
                    name_del.append(lines[item][col])
                uniq_dict[names] = name_del

            elif str(key_word) == "low" and float(sort_col) >= float(uniq_dict[names][col_word]):
                uniq_dict[names]= uniq_dict[names]

            if str(key_word) == "high" and float(sort_col) > float(uniq_dict[names][col_word]):
                name_del=[]
                for col in range(len(lines[item])):
                    name_del.append(lines[item][col])
                uniq_dict[names]= name_del

            elif str(key_word) == "high" and float(sort_col) <= float(uniq_dict[names][col_word]):
                uniq_dict[names]= uniq_dict[names]

#file_out ='/home/wuj/data/genome/PPI/CCSB/3.txt'

f=open(file_out,"w")
for line in list(uniq_dict.keys()):
    str_name = "\t".join(uniq_dict[line])
    line1 = line.split('_')
    f.write(line1[0]+'_'+line1[1]+'\t'+line1[2]+'-'+line1[3]+'\t'+str_name+'\n')
f.close()
