
####### conda activate dplm

import os
import shutil
import argparse
import pathlib
import pandas as pd
import torch
from esm import Alphabet, FastaBatchedDataset, ProteinBertModel, pretrained, MSATransformer
import sys
from esm import Alphabet, FastaBatchedDataset, ProteinBertModel, pretrained, MSATransformer
seq_model, seq_alphabet = pretrained.load_model_and_alphabet('/home/wuj/.cache/torch/hub/checkpoints/esm2_t33_650M_UR50D.pt')

import torch
sys.path.append("/home/wuj/data/tools/SaProt")
from utils.foldseek_util import get_struc_seq
from model.saprot.base import SaprotBaseModel
from transformers import EsmTokenizer

config = {"task": "base","config_path":'/home/wuj/data/tools/SaProt/weights/PLMs/SaProt_650M_AF2', "load_pretrained": True,}
sapro_model = SaprotBaseModel(**config)
sapro_tokenizer = EsmTokenizer.from_pretrained(config["config_path"])

#target_fa='Budding_yeast_unipro.fa'
#binder_fa='binder.fa'
#target_pdb='AFDB_Sac_baker_yeast/'
#binder_pdb='binder_pdb/'
#feature_path='sample_all_train_set_rd.txt'

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

#############################

def fasta_dict(path):
    with open(path) as fa:
        fa_dict = {}
        for line in fa:
            line = line.replace('\n','')
            if line.startswith('>'):
                seq_name = line[1:]
                fa_dict[seq_name] = ''
            else:
                fa_dict[seq_name] += line.replace('\n','')
    return fa_dict


import tqdm

target_fa_dict = fasta_dict(target_fa)
binder_fa_dict = fasta_dict(binder_fa)

encoding="utf-8"; corpus_lines=None; on_memory=True

with open(feature_path, "r", encoding=encoding) as f:
    lines = [line[:-1].split("\t")
        for line in tqdm.tqdm(f, desc="Loading blast", total=corpus_lines) if not line.startswith('#')]
    corpus_lines = len(lines)


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

