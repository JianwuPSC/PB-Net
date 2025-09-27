
####### conda activate dplm

import os
import shutil
import argparse
import pathlib
import pandas as pd
import torch
from esm import Alphabet, FastaBatchedDataset, ProteinBertModel, pretrained, MSATransformer
import sys

#target_fa='/home/wuj/data/genome/uniprotKB/Budding_yeast_unipro_pdb.fa'
#binder_fa='/home/wuj/data/project/ppi_predict/Sac_baker/feature/DSSP_align/sample_all/sample_all_cluster.fa'
#target_pdb='/home/wuj/data/genome/species_pdb/AFDB_Sac_baker_yeast/'
#binder_pdb='/home/wuj/data/project/ppi_predict/Sac_baker/feature/foldseek/sample_all_pdb_chainA/'
#feature_path='/home/wuj/data/project/ppi_predict/Sac_baker/filter_model/PLM_model/dataset/sample_all_PLM_quality_filter/sample_all_train_set_rd.txt'

#python /home/wuj/data/project/ppi_predict/Sac_baker/filter_model/PLM_model/PLM_embedding_to_csv.py /home/wuj/data/genome/uniprotKB/Budding_yeast_unipro_pdb.fa /home/wuj/data/project/ppi_predict/Sac_baker/feature/DSSP_align/sample_all/sample_all_cluster.fa /home/wuj/data/genome/species_pdb/AFDB_Sac_baker_yeast/ /home/wuj/data/project/ppi_predict/Sac_baker/feature/foldseek/sample_all_pdb_chainA/ /home/wuj/data/project/ppi_predict/Sac_baker/filter_model/PLM_model/dataset/sample_all_PLM_quality_filter/sample_all_train_set_rd.txt dataset/sample_all_PLM_quality_filter1 cuda:0

target_fa = sys.argv[1]
binder_fa = sys.argv[2]
target_pdb = sys.argv[3]
binder_pdb = sys.argv[4]
feature_path = sys.argv[5]
output_path = sys.argv[6]
device = sys.argv[7] # cuda:0


def MSA_representation(model_path,fasta_path,device):
    
    """
    model_path : /home/wuj/.cache/torch/hub/checkpoints/esm2_t33_650M_UR50D.pt
    fasta_path : [('ID','MDAFREA')] fasta.fa
    """
    
    model, alphabet = pretrained.load_model_and_alphabet(model_path)
    model.eval()
    
    if isinstance(model, MSATransformer):
        raise ValueError("This script currently does not handle models with MSA input (MSA Transformer).")
        
    if torch.cuda.is_available():
        model = model.cuda()

    dataset = FastaBatchedDataset.from_file(fasta_path)
    batches = dataset.get_batch_indices(512, extra_toks_per_seq=1)
    data_loader = torch.utils.data.DataLoader(dataset, collate_fn=alphabet.get_batch_converter(), batch_sampler=batches)
    print(f"Read {args.fasta_file} with {len(dataset)} sequences")

    assert all(-(model.num_layers + 1) <= i <= model.num_layers for i in [-1])
    repr_layers = [(i + model.num_layers + 1) % (model.num_layers + 1) for i in [-1]]
    
    label_list = []
    represent_list = []
    with torch.no_grad():
        for batch_idx, (labels, strs, toks) in enumerate(data_loader):

            if torch.cuda.is_available():
                toks = toks.to(device="cuda:0", non_blocking=True)
            
            print(f"Device: {toks.device}")

            out = model(toks, repr_layers=repr_layers, return_contacts=False)

            logits = out["logits"].to(device="cpu")
            representations = {layer: t.to(device="cpu") for layer, t in out["representations"].items()}
            
            for i, label in enumerate(labels):

                result = {"label": label}
                truncate_len = min(args.truncation_seq_length, len(strs[i]))
                represent_list.append((representations[repr_layers[0]])[i, 1 : truncate_len + 1].mean(0).clone())

        result = {'label' : label_list, 'representation' : represent_list}
         
    return result

##################################

def Sequen_representation(model,alphabet,fasta_seq,device):
    
    """
    model_path : /home/wuj/.cache/torch/hub/checkpoints/esm2_t33_650M_UR50D.pt
    fasta_seq : [('ID','MDAFREA')]
    """
    #model, alphabet = pretrained.load_model_and_alphabet(model_path)
    model.eval()
    
    if isinstance(model, MSATransformer):
        raise ValueError("This script currently does not handle models with MSA input (MSA Transformer).")
    if torch.cuda.is_available():
        model = model.to(device)

    batch_converter = alphabet.get_batch_converter()
    data_batch = batch_converter(fasta_seq)

    assert all(-(model.num_layers + 1) <= i <= model.num_layers for i in [-1])
    repr_layers = [(i + model.num_layers + 1) % (model.num_layers + 1) for i in [-1]]
    
    label_list = []
    represent_list = []
    
    with torch.no_grad():
        
        labels, strs, toks = data_batch
        
        if torch.cuda.is_available():
            toks = toks.to(device=device, non_blocking=True)
            
        print(f"Device: {toks.device}")

        out = model(toks, repr_layers=repr_layers, return_contacts=False)

        logits = out["logits"].to(device="cpu")
        representations = {layer: t.to(device="cpu") for layer, t in out["representations"].items()}
            
        for i, label in enumerate(labels):
                
            label_list.append(label)
            truncate_len = min(1280, len(strs[i]))
            represent_list.append((representations[repr_layers[0]])[i, 1 : truncate_len + 1].mean(0).clone())

        result = {'label':label_list, 'representation':represent_list[0].detach().numpy()}
         
    return result

#######################################

import sys
sys.path.append("/home/wuj/data/tools/SaProt")
from utils.foldseek_util import get_struc_seq
from model.saprot.base import SaprotBaseModel
from transformers import EsmTokenizer

def Saprot_representation(model,tokenizer,pdb_path,name,device):
    
    """
    foldseek_path : /home/wuj/data/tools/foldseek/bin/foldseek
    model_path : /home/wuj/data/tools/SaProt/weights/PLMs/SaProt_650M_AF2
    pdb_path : pdb pathway
    """
    
    parsed_seqs = get_struc_seq('/home/wuj/data/tools/foldseek/bin/foldseek', pdb_path, ["A"], plddt_mask=False)["A"]
    seq, foldseek_seq, combined_seq = parsed_seqs
    model.eval()
    
    #config = {
    #    "task": "base",
    #    "config_path": model_path, # Note this is the directory path of SaProt, not the ".pt" file
    #    "load_pretrained": True,}
    #model = SaprotBaseModel(**config)
    #tokenizer = EsmTokenizer.from_pretrained(config["config_path"])
    with torch.no_grad():
        device = device
        model.to(device)

        tokens = tokenizer.tokenize(combined_seq)
        inputs = tokenizer(combined_seq, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        embeddings = model.get_hidden_states(inputs, reduction="mean")
    
    result = {'label':name, 'representation':embeddings[0].to('cpu').detach().numpy()}
    
    return result

########################################

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


def Top_layer(result,vail_test=None,smote_scaler=None,regression_type='randomforest', top_n=None, final_round=10, experimental=False):
    
    x = result.drop(['class'], axis=1)  # 特征
    y = result['class']
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    x_train_drop = x_train.drop(['name'], axis=1)
    x_test_drop = x_test.drop(['name'], axis=1)
    
    X_train = x_train_drop
    X_test = x_test_drop
    
    x = result.drop(['class'], axis=1)  # 特征
    y = result['class']  # 标签
    
    if regression_type == 'linear':
        model = linear_model.LogisticRegression()
    elif regression_type == 'neuralnet':
        model = MLPClassifier(hidden_layer_sizes=(5), max_iter=1000, activation='relu', solver='adam', alpha=0.001,
                         batch_size='auto', learning_rate='constant', learning_rate_init=0.001, power_t=0.5,
                         momentum=0.9, nesterovs_momentum=True, shuffle=True, random_state=1, tol=0.0001,
                         verbose=False, warm_start=False, early_stopping=False, validation_fraction=0.1, beta_1=0.9,
                         beta_2=0.999, epsilon=1e-08)
    elif regression_type == 'randomforest':
        model = RandomForestClassifier(n_estimators=100, criterion='log_loss', max_depth=None, min_samples_split=2,
                                  min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=1.0,
                                  max_leaf_nodes=None, min_impurity_decrease=0.0, bootstrap=True, oob_score=False,
                                  n_jobs=None, random_state=1, verbose=0, warm_start=False, ccp_alpha=0.0,
                                  max_samples=None)
    elif regression_type == 'gradientboosting':
        model = xgboost.XGBClassifier(objective='reg:squarederror', colsample_bytree=0.3, learning_rate=0.1,
                                 max_depth=5, alpha=10, n_estimators=10)
    elif regression_type == 'knn':
        model = KNeighborsClassifier(n_neighbors=25, weights='uniform', algorithm='auto', leaf_size=30, p=2,
                                metric='minkowski', metric_params=None, n_jobs=None)
    elif regression_type == 'gp':
        model = GaussianProcessClassifier(kernel=None, alpha=1e-10, optimizer='fmin_l_bfgs_b', n_restarts_optimizer=0,
                                    normalize_y=False, copy_X_train=True, random_state=None)

    if smote_scaler=="smote":
        
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
    model.fit(X_train, y_train)

    # make predictions on train data
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    accuracy_train = accuracy_score(y_train, y_pred_train)
    precision_train = precision_score(y_train, y_pred_train, average=None)  #micro/macro/weighted
    recall_train = recall_score(y_train, y_pred_train, average=None)  #micro/macro/weighted
    f1_train = f1_score(y_train, y_pred_train, average=None)  #micro/macro/weighted
    
    accuracy_test = accuracy_score(y_test, y_pred_test)
    precision_test = precision_score(y_test, y_pred_test, average=None)  #micro/macro/weighted
    recall_test = recall_score(y_test, y_pred_test, average=None)  #micro/macro/weighted
    f1_test = f1_score(y_test, y_pred_test, average=None)  #micro/macro/weighted
    
    test_dataframe = {'name': x_test['name'], 'predict_class': y_pred_test,'true_class': y_test}
    test_dataframe = pd.DataFrame(test_dataframe)
    
    if vail_test is not None:
        vail = vail_test.drop(['name','class'], axis=1)  # 特征
        vail_name = vail_test['name']  # 标签
        vail_class = vail_test['class']
        vail_pred = model.predict(vail)
        
        vail_dataframe = {'name': vail_name, 'predict_class': vail_pred,'true_class':vail_class}
        vail_dataframe = pd.DataFrame(vail_dataframe)
        
        return vail_dataframe,recall_test,precision_test,accuracy_test,f1_test
    else:
        return test_dataframe,recall_test,precision_test,accuracy_test,f1_test

#############################

def fasta_dict(path):
    with open(path) as fa:
        fa_dict = {}
        for line in fa:
            # 去除末尾换行符
            line = line.replace('\n','')
            if line.startswith('>'):
                # 去除 > 号
                seq_name = line[1:]
                fa_dict[seq_name] = ''
            else:
                # 去除末尾换行符并连接多行序列
                fa_dict[seq_name] += line.replace('\n','')
    return fa_dict

#############################

#target_fa='/home/wuj/data/genome/uniprotKB/Budding_yeast_unipro_pdb.fa'
#binder_fa='/home/wuj/data/project/ppi_predict/Sac_baker/feature/DSSP_align/sample_all/sample_all_cluster.fa'
#target_pdb='/home/wuj/data/genome/species_pdb/AFDB_Sac_baker_yeast/'
#binder_pdb='/home/wuj/data/project/ppi_predict/Sac_baker/feature/foldseek/sample_all_pdb_chainA/' 
#feature_path='/home/wuj/data/project/ppi_predict/Sac_baker/filter_model/PLM_model/dataset/sample_all_PLM_quality_filter/sample_all_train_set_rd.txt'

import tqdm

target_fa_dict = fasta_dict(target_fa)
binder_fa_dict = fasta_dict(binder_fa)

encoding="utf-8"; corpus_lines=None; on_memory=True

with open(feature_path, "r", encoding=encoding) as f:
    lines = [line[:-1].split("\t")
        for line in tqdm.tqdm(f, desc="Loading blast", total=corpus_lines) if not line.startswith('#')]
    corpus_lines = len(lines)


##### sequence_embedding
from esm import Alphabet, FastaBatchedDataset, ProteinBertModel, pretrained, MSATransformer
seq_model, seq_alphabet = pretrained.load_model_and_alphabet('/home/wuj/.cache/torch/hub/checkpoints/esm2_t33_650M_UR50D.pt')
#seq_model.eval()

##### sapro_embedding
import sys
import torch
sys.path.append("/home/wuj/data/tools/SaProt")
from utils.foldseek_util import get_struc_seq
from model.saprot.base import SaprotBaseModel
from transformers import EsmTokenizer

config = {"task": "base","config_path":'/home/wuj/data/tools/SaProt/weights/PLMs/SaProt_650M_AF2', "load_pretrained": True,}
sapro_model = SaprotBaseModel(**config)
sapro_tokenizer = EsmTokenizer.from_pretrained(config["config_path"])
#sapro_model.eval()

name_list=[]
represent_list=[]
class_list=[]

esm_represent_soucer_list=[]
esm_represent_binder_list=[]
esm_represent_target_list=[]

sapro_represent_soucer_list=[]
sapro_represent_binder_list=[]
sapro_represent_target_list=[]

for item in range(len(lines)):
    
    name = lines[item][0].split('_') # P02829_1-50_14_dldesign_13_4-63_P53691_304-367
    binder = '_'.join(name[0:5]) # binder P02829_1-50_14_dldesign_13
    target = name[6] # traget P53691
    source = binder.split('_')[0] # source P02829
    
    result = 0 if lines[item][-1] == "posi" else 1
    ppi_class = int(result) # posi:0 neg:1
    
    source_sapro_input = ''.join([target_pdb,'AF-'+source+'-F1-model_v4.pdb'])
    target_sapro_input = ''.join([target_pdb,'AF-'+target+'-F1-model_v4.pdb'])
    binder_sapro_input = ''.join([binder_pdb,binder+'.pdb'])

    if os.path.exists(source_sapro_input) and os.path.exists(binder_sapro_input) and os.path.exists(target_sapro_input) and \
        (source in target_fa_dict.keys()) and (binder in binder_fa_dict.keys()) and (target in target_fa_dict.keys()):

        source_seq_input = [(source,target_fa_dict[source])]
        binder_seq_input = [(binder,binder_fa_dict[binder])]
        target_seq_input = [(target,target_fa_dict[target])]
    
        ########################## representation
    
        source_seq_repre = Sequen_representation(seq_model,seq_alphabet,source_seq_input,device)
        binder_seq_repre = Sequen_representation(seq_model,seq_alphabet,binder_seq_input,device)
        target_seq_repre = Sequen_representation(seq_model,seq_alphabet,target_seq_input,device)
    
        source_sapro_repre = Saprot_representation(sapro_model,sapro_tokenizer,source_sapro_input,source,device)
        binder_sapro_repre = Saprot_representation(sapro_model,sapro_tokenizer,binder_sapro_input,binder,device)
        target_sapro_repre = Saprot_representation(sapro_model,sapro_tokenizer,target_sapro_input,target,device)
    
        total_repre = source_seq_repre['representation']-binder_seq_repre['representation']-target_seq_repre['representation']+ \
            source_sapro_repre['representation']-binder_sapro_repre['representation']-target_sapro_repre['representation']
    
        name_list.append(lines[item][0])
        represent_list.append(total_repre)
        class_list.append(ppi_class)
    
        esm_represent_soucer_list.append(source_seq_repre['representation'])
        esm_represent_binder_list.append(binder_seq_repre['representation'])
        esm_represent_target_list.append(target_seq_repre['representation'])

        sapro_represent_soucer_list.append(source_sapro_repre['representation'])
        sapro_represent_binder_list.append(binder_sapro_repre['representation'])
        sapro_represent_target_list.append(target_sapro_repre['representation'])
    
        del source_seq_repre; del binder_seq_repre; del target_seq_repre
        del source_sapro_repre; del binder_sapro_repre; del target_sapro_repre
    
        torch.cuda.empty_cache()
        print(item)


#esm_represent_soucer_df = pd.DataFrame([i['representation'] for i in esm_represent_soucer_list])
#esm_represent_binder_df = pd.DataFrame([i['representation'] for i in esm_represent_binder_list])
#esm_represent_target_df = pd.DataFrame([i['representation'] for i in esm_represent_target_list])

#sapro_represent_soucer_df = pd.DataFrame([i['representation'] for i in sapro_represent_soucer_list])
#sapro_represent_binder_df = pd.DataFrame([i['representation'] for i in sapro_represent_binder_list])
#sapro_represent_target_df = pd.DataFrame([i['representation'] for i in sapro_represent_target_list])

################################################################


from sklearn.model_selection import train_test_split

name_df = pd.DataFrame(name_list)
name_df = name_df.rename(columns={0: 'name'})
represent_df = pd.DataFrame(represent_list)

esm_represent_soucer_df = pd.DataFrame(esm_represent_soucer_list)
esm_represent_binder_df = pd.DataFrame(esm_represent_binder_list)
esm_represent_target_df = pd.DataFrame(esm_represent_target_list)

sapro_represent_soucer_df = pd.DataFrame(sapro_represent_soucer_list)
sapro_represent_binder_df = pd.DataFrame(sapro_represent_binder_list)
sapro_represent_target_df = pd.DataFrame(sapro_represent_target_list)

class_df = pd.DataFrame(class_list)
class_df = class_df.rename(columns={0: 'class'})

result = pd.concat([name_df,represent_df,class_df],axis=1)
result.to_csv(output_path+'/'+'sample_all_train_set_embedding.csv', index=False,header=False)

esm_represent_soucer_df = pd.concat([name_df,esm_represent_soucer_df,class_df],axis=1)
esm_represent_soucer_df.to_csv(output_path+'/'+'esm_represent_soucer_df.csv', index=False,header=False)

esm_represent_binder_df = pd.concat([name_df,esm_represent_binder_df,class_df],axis=1)
esm_represent_binder_df.to_csv(output_path+'/'+'/esm_represent_binder_df.csv', index=False,header=False)

esm_represent_target_df = pd.concat([name_df,esm_represent_target_df,class_df],axis=1)
esm_represent_target_df.to_csv(output_path+'/'+'esm_represent_target_df.csv', index=False,header=False)

sapro_represent_soucer_df = pd.concat([name_df,sapro_represent_soucer_df,class_df],axis=1)
sapro_represent_soucer_df.to_csv(output_path+'/'+'sapro_represent_soucer_df.csv', index=False,header=False)

sapro_represent_binder_df = pd.concat([name_df,sapro_represent_binder_df,class_df],axis=1)
sapro_represent_binder_df.to_csv(output_path+'/'+'sapro_represent_binder_df.csv', index=False,header=False)

sapro_represent_target_df = pd.concat([name_df,sapro_represent_target_df,class_df],axis=1)
sapro_represent_target_df.to_csv(output_path+'/'+'sapro_represent_target_df.csv', index=False,header=False)

#############################################################################################################
