import sys
import re

path_a = sys.argv[1]
path_b = sys.argv[2]
output_a3m = sys.argv[3]

#path_a = '/home/wuj/data/project/ppi_predict/Sac_baker/rosettafold/Sac_eku_msa/mmseq_Q12248.a3m'
#path_b = '/home/wuj/data/project/ppi_predict/Sac_baker/rosettafold/Sac_eku_msa/mmseq_P53168.a3m'

def fasta_dict(path_a):
    faa_dict = {}
    with open(path_a) as fa:
        for line in fa:
            # 去除末尾换行符
            line = line.replace('\n','')
            if not line.startswith('#'):
                if line.startswith('>'):
                    # 去除 > 号
                    seq_name = str(line).split('\t')[0][1:]
                    faa_dict[seq_name] = ''
                else:
                    # 去除末尾换行符并连接多行序列
                    faa_dict[seq_name] += line.replace('\n','')
    return faa_dict


def a3m_merge(fa_dict, ba_dict):
    
    a3m_merge_list = []
    a3m_merge_list.append('>'+list(fa_dict.keys())[0]+'_'+list(ba_dict.keys())[0])
    a3m_merge_list.append(fa_dict[list(fa_dict.keys())[0]]+ba_dict[list(ba_dict.keys())[0]])

    for i in range(1,len(fa_dict.keys())):
        species = list(fa_dict.keys())[i].split('_')[0]
        for filename in list(ba_dict.keys()):
            if species in filename:
                a3m_merge_list.append('>'+list(fa_dict.keys())[i]+'_'+filename)
                a3m_merge_list.append(fa_dict[list(fa_dict.keys())[i]]+ba_dict[filename])
                
    return a3m_merge_list


def clean_a3m(lines):
    # 读取输入文件
    target_seq = None
    homolog_seqs = []
    for line in lines:
        if line.startswith(">"):
            homolog_seqs.append(line)  # 保存序列名称
        else:
            if target_seq is None:
                target_seq = line.strip()  # 第一行是目标序列
            else:
                homolog_seqs.append(line.strip())  # 保存同源序列
    # 找到目标序列中需要保留的列（非 '-' 的位置）
    keep_indices = [i for i, char in enumerate(target_seq) if char != '-']
    # 生成新的目标序列（去除 '-'）
    new_target_seq = "".join([target_seq[i] for i in keep_indices])
    # 生成新的同源序列（去除对应列的 '-'）
    new_homolog_seqs = []
    for line in homolog_seqs:
        if line.startswith(">"):
            new_homolog_seqs.append(line)  # 保留序列名称
        else:
            new_seq = "".join([line[i] for i in keep_indices])
            new_homolog_seqs.append(new_seq)
    
    return new_target_seq, new_homolog_seqs[1:]


fa_dict = fasta_dict(path_a)
ba_dict = fasta_dict(path_b)

a3m_merge_list = a3m_merge(fa_dict,ba_dict)

new_target_seq, new_homolog_seqs = clean_a3m(a3m_merge_list)


with open(output_a3m, "w") as f:
    f.write(f">query\n{new_target_seq}\n")  # 写入目标序列
    for seq in new_homolog_seqs:
        f.write(f"{seq}\n")  # 写入同源序列
