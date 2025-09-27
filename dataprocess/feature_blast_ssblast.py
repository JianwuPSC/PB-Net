from Bio import pairwise2
from Bio import Seq
from Bio import Align
from Bio.Align import substitution_matrices
import numpy as np
import sys
import re

target = sys.argv[1]
binder_fasta = sys.argv[2]
binder_ss = sys.argv[3]
target_fasta = sys.argv[4]
target_ss = sys.argv[4]

fasta_align_out = sys.argv[5]
ss_align_out = sys.argv[6]

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

##########################################################

def format_alignment(pseq1,pseq2):
    blosum62 = substitution_matrices.load("BLOSUM62")
    lPRT = pairwise2.align.localds(pseq1,pseq2,blosum62,-4,-1)
    if not lPRT :
        seqA=''
        seqB=''
        score=0
    else:
        seqA=lPRT[0][0]
        seqB=lPRT[0][1]
        score=lPRT[0][2]
    return score, seqA, seqB

###########################################################

lst_2=['HG','GH','HI','GI','IH','IG','EB','BE','CS','SC'] 
lst_5=['HT','HS','HC','GT','GS','GC','IT','IS','IC','BT','BS','BC',
       'ET','ES','EC','TH','TG','TI','TB','TE','TS','TC','SH','SG',
       'SI','SB','SE','ST','CH','CG','CI','CB','CE','CT']
  
def score(a,b):
    #scoring function 
    score=0
    if a==b: 
      score +=2 
    elif a+b in lst_2: 
      score += -2
    elif a+b in lst_5: 
      score += -5  
    else: 
      score += -7 
    return score 
 
def SSblast(seq1,seq2,GAP=-5,cove_count=0,non_C_cover=0):
  #Basic Local Alignment Search Tool 

  l1 = len(seq1)
  l2 = len(seq2)
  
  scores =[] 
  point =[]
   
  for j in range(l2+1): 
    if j == 0: 
      line1=[0] 
      line2=[0] 
      for i in range(1,l1+1): 
        line1.append(GAP*i)
        line2.append(2) 
    else: 
      line1=[] 
      line2=[] 
      line1.append(GAP*j) 
      line2.append(3) 
    scores.append(line1) 
    point.append(line2) 
   
  #fill the blank of scores and point 
  for j in range(1,l2+1): 
    letter2 = seq2[j-1] 
    for i in range(1,l1+1): 
      letter1 = seq1[i-1] 
      
      diagonal_score = score(letter1, letter2) + scores[j-1][i-1] 
      left_score = GAP + scores[j][i-1] 
      up_score = GAP + scores[j-1][i] 
      max_score = max(diagonal_score, left_score, up_score) 
      scores[j].append(max_score)      
       
      if scores[j][i] == diagonal_score: 
        point[j].append(1) 
      elif scores[j][i] == left_score: 
        point[j].append(2) 
      else: 
        point[j].append(3)
         
  #trace back 
  alignment1='' 
  alignment2='' 
  i = l2 
  j = l1 
  #print('scores =',scores[i][j])
  scores = scores[i][j]
  while True: 
    if point[i][j] == 0: 
      break 
    elif point[i][j] == 1: 
      alignment1 += seq1[j-1] 
      alignment2 += seq2[i-1] 
      i -= 1 
      j -= 1 
    elif point[i][j] == 2: 
      alignment1 += seq1[j-1] 
      alignment2 += '-' 
      j -= 1 
    elif point[i][j] == 3:  
      alignment1 += '-' 
      alignment2 += seq2[i-1] 
      i -= 1 
       
  #reverse alignment 
  alignment1 = alignment1[::-1] 
  alignment2 = alignment2[::-1]
  
  for cov1 in range(len(alignment1)):
     # if alignment1[cov1] == alignment2[cov1]:
     #     cove_count = cove_count+1
      if (alignment1[cov1] == alignment2[cov1] or (alignment1[cov1]+alignment2[cov1] in lst_2)):
          non_C_cover = non_C_cover+1
  #identity = cove_count/l1
  #identity = round(identity,3)
  non_C_identity = non_C_cover/l1
  non_C_identity = round(non_C_identity,3)
  
  return scores,alignment1,alignment2,non_C_identity


binder_fasta_dict = fasta_dict(binder_fasta)
binder_ss_dict = fasta_dict(binder_ss)

target_fasta_dict = fasta_dict(target_fasta)
target_ss_dict = fasta_dict(target_ss)


data = np.loadtxt(target, delimiter='\t', dtype=str)

f_fasta = open(fasta_align_out,"w")
f_ss = open(ss_align_out,"w")

for i in range(len(data)): 
    print(data[i,:])
    binder = data[i,0]
    target = data[i,1]
    binder_start = int(data[i,2])
    binder_end = int(data[i,3])
    target_start = int(data[i,4])
    target_end = int(data[i,5])
    
    binder_fasta_split = ''.join(list(binder_fasta_dict[binder])[binder_start-1:binder_end])
    target_fasta_split = ''.join(list(target_fasta_dict[target])[target_start-1:target_end])
    
    binder_ss_split = ''.join(list(binder_ss_dict[binder])[binder_start-1:binder_end])
    target_ss_split = ''.join(list(target_ss_dict[target])[target_start-1:target_end])
    
    fasta_score, SeqA, SeqB = format_alignment(binder_fasta_split,target_fasta_split)
    ss_score, SsA, SsB, C_id = SSblast(binder_ss_split,target_ss_split)
    
    f_fasta.write(binder+'\t'+target+'\t'+str(binder_start)+'\t'+str(binder_end)+'\t'+str(target_start)+'\t'+str(target_end)+'\t'+str(fasta_score)+'\t'+SeqA+'\t'+SeqB+'\n')
    f_ss.write(binder+'\t'+target+'\t'+str(binder_start)+'\t'+str(binder_end)+'\t'+str(target_start)+'\t'+str(target_end)+'\t'+str(ss_score)+'\t'+SsA+'\t'+SsB+'\t'+str(C_id)+'\n')
    
f_fasta.close()
f_ss.close()
