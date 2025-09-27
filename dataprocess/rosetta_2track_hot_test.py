import numpy as np
import sys
from scipy.stats import ttest_ind

file_path = sys.argv[1]
file_out = sys.argv[2]
key_word = sys.argv[3]

poem=np.load(file_path,allow_pickle=True)
poem.files

dist = poem['dist']
A, B = dist.shape

# 预先计算行和列的和
row_sums = np.sum(dist, axis=1)  # 每行的和
col_sums = np.sum(dist, axis=0)  # 每列的和
total_sum = np.sum(dist)         # 矩阵的总和
# 计算 score 矩阵
score = np.zeros_like(dist)  # 初始化 score 矩阵
# 使用向量化操作计算 score
mask = dist >= 0.9  # 条件 mask
score[mask] = 2 * dist[mask] - (row_sums[:, None] * col_sums[None, :] / total_sum)[mask]
score[~mask] = dist[~mask] - (row_sums[:, None] * col_sums[None, :] / total_sum)[~mask]

top_score = np.amax(score[0:A-10,10:B])


f=open(file_out,"w")
f.write(key_word+'\t'+str(top_score)+'\n')
f.close()

