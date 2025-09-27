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

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
# 
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

input_size = 1280*4
#hidden_size = 128
#batch_size = 12
#learning_rate = 0.00001
#num_epochs = 100

def main(args):

    esm_represent_soucer_df1 = pd.read_csv(args.protA_esm_train_path,header=None)
    esm_represent_binder_df1 = pd.read_csv(args.binder_esm_train_path,header=None)
    esm_represent_target_df1 = pd.read_csv(args.protB_esm_train_path,header=None)
    sapro_represent_soucer_df1 = pd.read_csv(args.protA_saprot_train_path,header=None)
    sapro_represent_binder_df1 = pd.read_csv(args.binder_saprot_train_path,header=None)
    sapro_represent_target_df1 = pd.read_csv(args.protB_saprot_train_path,header=None)

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
    result = pd.concat([name_df,sapro_represent_soucer_df1,sapro_represent_target_df1,class_df],axis=1).iloc[0:17913,]

    subset_columns = result.columns[1:]
    duplicated_rows = result[result.duplicated(subset=subset_columns, keep='first')]
    df_unique = result.drop_duplicates(subset=subset_columns, keep='first')
    deleted_indexes = duplicated_rows.index.tolist()
    result = df_unique

    x_train = result.drop(['class','name'], axis=1)  #
    y_train = result['class']  #

    print(len(result.iloc[:,0]))

    #########################################################################################

    esm_represent_soucer_df1 = pd.read_csv(args.protA_esm_valid_path,header=None)
    esm_represent_binder_df1 = pd.read_csv(args.binder_esm_valid_path,header=None)
    esm_represent_target_df1 = pd.read_csv(args.protB_esm_valid_path,header=None)
    sapro_represent_soucer_df1 = pd.read_csv(args.protA_saprot_valid_path,header=None)
    sapro_represent_binder_df1 = pd.read_csv(args.binder_saprot_valid_path,header=None)
    sapro_represent_target_df1 = pd.read_csv(args.protB_saprot_valid_path,header=None)

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
    result = pd.concat([name_df,sapro_represent_soucer_df1,sapro_represent_target_df1,class_df],axis=1).iloc[0:17913,]

    subset_columns = result.columns[1:]
    duplicated_rows = result[result.duplicated(subset=subset_columns, keep='first')]
    df_unique = result.drop_duplicates(subset=subset_columns, keep='first')
    deleted_indexes = duplicated_rows.index.tolist()
    result = df_unique

    x_valid = result.drop(['class','name'], axis=1)  #
    y_valid = result['class']  #

    print(len(result.iloc[:,0]))

    ###############################################################################
    x_train = torch.tensor(np.array(x_train),dtype=torch.float32)
    y_train = torch.tensor(np.array(y_train)).long()
    x_valid = torch.tensor(np.array(x_valid),dtype=torch.float32)
    y_valid = torch.tensor(np.array(y_valid)).long()

    train_dataset = TensorDataset(x_train, y_train)
    valid_dataset = TensorDataset(x_valid, y_valid)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    model = SimpleMLP(input_size, hidden_size)

    #weights = torch.tensor([0.15, 0.85], dtype=torch.float32)
    #criterion = nn.CrossEntropyLoss(weights)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        for inputs, targets in train_dataloader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * inputs.size(0)

        with torch.no_grad():
            for inputs, targets in valid_dataloader:
                outputs = model(inputs)
                valid_loss = criterion(outputs, target)        
                total_valid_loss += valid_loss.item() * inputs.size(0)

        print(f'Epoch [{epoch+1}/{num_epochs}], train_loss: {total_loss / len(train_dataloader.dataset):.8f}, valid_loss: {total_valid_loss / len(valid_dataloader.dataset):.8f}')

        model.eval()
        outputs = model(x_valid)
        _, predicted = torch.max(outputs.data, 1)

        correct = accuracy_score(y_valid, predicted)
        f1_score_test = (f1_score(y_valid, predicted, average=None))
        precision_test = (precision_score(y_valid, predicted, average=None))
        recall_test = (recall_score(y_valid, predicted, average=None))

        print(f'{precision_test}  {recall_test}  {correct}  {f1_score_test}')

    torch.save(model.state_dict(), args.model_out)


##########################################################

if __name__ == "__main__":

    os.environ["WANDB_MODE"] = "offline"

    parser = argparse.ArgumentParser()

    # General args
    parser.add_argument("--protA_esm_train_path", type=str, default="train_path")
    parser.add_argument("--protA_esm_valid_path", type=str, default="valid_path")
    parser.add_argument("--protA_esm_test_path", type=str, default="test_path")
    parser.add_argument("--protB_esm_train_path", type=str, default="train_path")
    parser.add_argument("--protB_esm_valid_path", type=str, default="valid_path")
    parser.add_argument("--protB_esm_test_path", type=str, default="test_path")
    parser.add_argument("--binder_esm_train_path", type=str, default="train_path")
    parser.add_argument("--binder_esm_valid_path", type=str, default="valid_path")
    parser.add_argument("--binder_esm_test_path", type=str, default="test_path")
    parser.add_argument("--protA_saprot_train_path", type=str, default="train_path")
    parser.add_argument("--protA_saprot_valid_path", type=str, default="valid_path")
    parser.add_argument("--protA_saprot_test_path", type=str, default="test_path")
    parser.add_argument("--protB_saprot_train_path", type=str, default="train_path")
    parser.add_argument("--protB_saprot_valid_path", type=str, default="valid_path")
    parser.add_argument("--protB_saprot_test_path", type=str, default="test_path")
    parser.add_argument("--binder_saprot_train_path", type=str, default="train_path")
    parser.add_argument("--binder_saprot_valid_path", type=str, default="valid_path")
    parser.add_argument("--binder_saprot_test_path", type=str, default="test_path")

    parser.add_argument("--model_out", type=str, default="sample.pt")
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--learning_rate", type=float, default=0.00001)
    parser.add_argument("--num_epochs", type=int, default=100)

    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--use_mlp_for_cv", action="store_true", default=False)
    parser.add_argument("--normalize", action="store_true", default=False)
    parser.add_argument("--sep", action="store_true", default=False)

    args = parser.parse_args()
    main(args)
