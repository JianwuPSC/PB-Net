# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 23:35:42 2025

@author: Admin
"""
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn_extra.cluster import KMedoids
from sklearn import linear_model
from sklearn.neural_network import MLPRegressor,MLPClassifier
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
import xgboost
from xgboost import XGBRegressor,XGBClassifier
from sklearn.neighbors import KNeighborsRegressor,KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessRegressor,GaussianProcessClassifier
from sklearn.metrics import mean_squared_error, r2_score
from scipy.spatial.distance import cdist
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

source_esm_file = sys.argv[1]
binder_esm_file = sys.argv[2]
target_esm_file = sys.argv[3]
source_saprot_file = sys.argv[4]
binder_saprot_file = sys.argv[5]
target_saprot_file = sys.argv[6]
model_file = sys.argv[7]
out_put = sys.argv[8]
 
class SimpleMLP(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(SimpleMLP, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_size, hidden_size*4),
            nn.ReLU(),
            nn.Linear(hidden_size*4, hidden_size),
            nn.ReLU(),            
            nn.Linear(hidden_size, 2)
        )

    def forward(self, x):
        return self.classifier(x)

# 
input_size = 1280*4  #10
hidden_size = 128  #
batch_size = 12
learning_rate = 0.00001
num_epochs = 100


model = SimpleMLP(input_size,hidden_size)
model.load_state_dict(torch.load(model_file))

esm_represent_soucer_df1 = pd.read_csv(source_esm_file,header=None)
esm_represent_binder_df1 = pd.read_csv(binder_esm_file,header=None)
esm_represent_target_df1 = pd.read_csv(target_esm_file,header=None)

sapro_represent_soucer_df1 = pd.read_csv(source_saprot_file,header=None)
sapro_represent_binder_df1 = pd.read_csv(binder_saprot_file,header=None)
sapro_represent_target_df1 = pd.read_csv(target_saprot_file,header=None)

name_df = sapro_represent_soucer_df1.rename(columns={0: 'name'})
class_df = sapro_represent_soucer_df1.rename(columns={1281: 'class'})

name_df = name_df[['name']]
class_df = class_df[['class']]

esm_represent_soucer_df1 = esm_represent_soucer_df1.iloc[:,1:-1]
esm_represent_binder_df1 = esm_represent_binder_df1.iloc[:,1:-1]
esm_represent_target_df1 = esm_represent_target_df1.iloc[:,1:-1]

sapro_represent_soucer_df1 = sapro_represent_soucer_df1.iloc[:,1:-1]
sapro_represent_binder_df1 = sapro_represent_binder_df1.iloc[:,1:-1]
sapro_represent_target_df1 = sapro_represent_target_df1.iloc[:,1:-1]

esm_total = esm_represent_soucer_df1-esm_represent_binder_df1-esm_represent_target_df1
sapro_total = sapro_represent_soucer_df1-sapro_represent_binder_df1-sapro_represent_target_df1

result = pd.concat([name_df,sapro_represent_soucer_df1,sapro_represent_binder_df1,sapro_represent_target_df1,esm_total,class_df],axis=1)

x = result.drop(['class','name'], axis=1)
y = result['class']

X = torch.tensor(np.array(x),dtype=torch.float32)
y = torch.tensor(np.array(y)).long()

# TensorDataset and DataLoader
dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

predict_list=[]

model.eval()  # test
with torch.no_grad():  # test
    correct = 0
    total = 0
    f1_score_test = 0
    precision_test = 0
    recall_test = 0
    crcle = 0
    for inputs, targets in dataloader:
        outputs = model(inputs)
        #print(outputs.data)
        _, predicted = torch.max(outputs.data, 1)
        predict_list.append(predicted)
        #print(f'true:{targets}')
        #print(f'predict:{predicted}')
        crcle +=1
        total += targets.size(0)
        correct += accuracy_score(targets, predicted)
        f1_score_test += (f1_score(targets, predicted, average=None)[0])
        precision_test += (precision_score(targets, predicted, average=None)[0])
        recall_test += (recall_score(targets, predicted, average=None)[0])


#############################################################################

model.eval() ## predict
with torch.no_grad():
    outputs = model(X)
    output_softmax = torch.softmax(outputs,dim=1)[:,0]
    _, predicted = torch.max(outputs.data, 1)

correct = accuracy_score(y, predicted)
f1_score_test = (f1_score(y, predicted, average=None))
precision_test = (precision_score(y, predicted, average=None)) #
recall_test = (recall_score(y, predicted, average=None))  #

print(f'{precision_test}  {recall_test}  {correct}  {f1_score_test}')

data = {'name': result['name'],'predict': predicted.numpy(),"softmax":output_softmax.numpy()}
df = pd.DataFrame(data)
df.to_csv(out_put, index=False)
