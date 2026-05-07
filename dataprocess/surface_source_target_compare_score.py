import sys
import os
import scipy.stats as stats
import pandas as pd
import numpy as np
# 创建示例数据
np.random.seed(0)

def Bin_range(max_num,min_num,count):
    
    bins = []
    bins_name = []
    
    for i in range(count):
        size_bin = (max_num - min_num)/count
        bin_one = i*size_bin + min_num
        bins.append(round(bin_one,3))
        bins_name.append(str(round(bin_one,3))+'_'+str(round(size_bin,3)))
        
    return bins, bins_name

def shape_score_arrange(source,source_name,target,target_name, col=0, bin_count=30):
    
    #source_name = source.split("/")[-1].replace("_surface.txt", "")
    #target_name = target.split("/")[-1].replace("_surface.txt", "")
    name = source_name+'_'+target_name
    
    poem=np.loadtxt(source)
    poem1=np.loadtxt(target)
    data = {'source': poem[:,col],'target': poem1[:,col]}
    df = pd.DataFrame(data)
    min_num = np.nanmin([list(poem[:,col]),list(poem1[:,col])])
    max_num = np.nanmax([list(poem[:,col]),list(poem1[:,col])])
    
    bins, bins_name = Bin_range(max_num,min_num,bin_count)
    df['source_group'] = pd.cut(df['source'], bins=bins, labels=bins_name[:-1])
    df['target_group'] = pd.cut(df['target'], bins=bins, labels=bins_name[:-1])
    
    data = {'class': bins_name[:-1],
            'source': df.groupby(['source_group'])['source'].count(),
            'target': df.groupby(['target_group'])['source'].count()}
    
    df_data = pd.DataFrame(data)
    source = np.array(df_data['source']).reshape(bin_count-1,1)
    target = np.array(df_data['target']).reshape(bin_count-1,1)

    if np.sum(target) == 0:
        target = source
    if np.sum(source) == 0:
        source = target  
    
    ober = np.concatenate((source,target), axis=1)
    ober_right = ober[[not np.all(ober[i] == 0) for i in range(ober.shape[0])], :]

    chi2_value, p_value, dof, expected = stats.chi2_contingency(ober_right)

    source_arg = float((bins_name[:-1][np.argmax((ober[:,0] - ober[:,1]))]).split('_')[0])
    target_arg = float((bins_name[:-1][np.argmax((ober[:,1] - ober[:,0]))]).split('_')[0])
    
    source_max = float(np.nanmax((ober[:,0] - ober[:,1])))
    target_max = float(np.nanmax((ober[:,1] - ober[:,0])))
    
    score_single = -np.log(p_value) * source_arg * target_arg * -1.0 * source_max * target_max / 1000
    
    ###########################################################
    statistic, pvalue = stats.wilcoxon(poem[:,col], poem1[:,col])  # 配对样本的非参数检验
    rev_statistic, rev_pvalue = stats.wilcoxon(poem[:,col], poem1[:,col]*-1.0)
    p_ratio = -np.log(pvalue)/(-np.log(rev_pvalue)+0.001)
    
    return name, score_single, p_ratio #bins_name[:-1], ober[:,0], ober[:,1]


input_path = sys.argv[1]
output_path = sys.argv[2]
protA_path = sys.argv[3]
protB_path = sys.argv[4]

shape_dict = {}
ddc_dict = {}
free_dict = {}
ce_dict = {}
hyd_dict = {}

input_example = pd.read_table(input_path,names=['source',"target",'source_start','source_end','target_start','target_end'])

for index,row in input_example.iterrows():
    #target_name = (row['target'].split("/")[-1]).split("_")[0]

    target_name1 = row['target']
    target_surface= row['target']+'_'+str(row['target_start'])+'_'+str(row['target_end'])

    source_name = row['source']+'_'+str(row['source_start'])+'-'+str(row['source_end'])
    target_name = row['target']+'_'+str(row['target_start'])+'-'+str(row['target_end'])

    source = str(protA_path)+row['source']+'_af2pred_surface.txt'
    target = str(protB_path)+target_surface+'_surface.txt'
    #print(f'{source}  {target}  {source_name}  {target_name}')
    #print({source},{target})
    if os.path.exists(source) and os.path.exists(target):
        name, score_single, p_ratio  = shape_score_arrange(source,source_name,target,target_name, col=0, bin_count=20)
        shape_dict[name] = str(score_single)+','+str(p_ratio)
        name, score_single, p_ratio  = shape_score_arrange(source,source_name,target,target_name, col=1, bin_count=20)
        ddc_dict[name] = str(score_single)+','+str(p_ratio)
        name, score_single, p_ratio  = shape_score_arrange(source,source_name,target,target_name, col=2, bin_count=20)
        free_dict[name] = str(score_single)+','+str(p_ratio)
        name, score_single, p_ratio  = shape_score_arrange(source,source_name,target,target_name, col=3, bin_count=20)
        ce_dict[name] = str(score_single)+','+str(p_ratio)
        name, score_single, p_ratio  = shape_score_arrange(source,source_name,target,target_name, col=4, bin_count=20)
        hyd_dict[name] = str(score_single)+','+str(p_ratio)
    
f=open(output_path,"w")
for line in list(shape_dict.keys()):
    f.write(line+'\t'+shape_dict[line]+'\t'+ddc_dict[line]+'\t'+free_dict[line]+'\t'+ce_dict[line]+'\t'+hyd_dict[line]+'\n')
f.close()
