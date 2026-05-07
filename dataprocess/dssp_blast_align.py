
#>seq1
#HHHHHHHHHHHTCHHHHHHHHHHTTCCHHHHHHHHHHHHH
#>seq2
#HHHHHHHHHHHTCHHHHHHHHHHTTCCHHHHHHHHHHHHH
#HHHHHHHHHHHTCHHHHHHHHHHTTCCHHHHHHHHHHHHH
#HHHHHHHHHHHTCHHHHHHHHHHTTCCHHHHHHHHHHHHH
#HHHHHHHHHHHTCHHHHHHHHHHTTCCHHHHHHHHHHHHH
import sys

data_path = sys.argv[1]
target_path = sys.argv[2]
file_out = sys.argv[3]


#	H	G	I	B	E	T	S	C
#H	2	-2	-2	-7	-7	-5	-5	-5
#G	-2	2	-2	-7	-7	-5	-5	-5
#I	-2	-2	2	-7	-7	-5	-5	-5
#B	-7	-7	-7	2	-2	-5	-5	-5
#E	-7	-7	-7	-2	2	-5	-5	-5
#T	-5	-5	-5	-5	-5	2	-5	-5
#S	-5	-5	-5	-5	-5	-5	2	-2
#C	-5	-5	-5	-5	-5	-5	-2	2

''''' 
@author: wuj
''' 
lst_2=['HG','GH','HI','GI','IH','IG','EB','BE','CS','SC']

lst_5=['HT','HS','HC','GT','GS','GC','IT','IS','IC','BT','BS','BC',
       'ET','ES','EC','TH','TG','TI','TB','TE','TS','TC','SH','SG',
       'SI','SB','SE','ST','CH','CG','CI','CB','CE','CT']

#lst_2=['HG','GH','HI','GI','IH','IG','EB','BE','CS','SC','TS','TC','ST','SC','CT','CS']
#lst_5=[]
  
def score(a,b):#scoring function 
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
 
def BLAST(seq1,seq2,GAP=-5,cove_count=0,non_C_cover=0):#Basic Local Alignment Search Tool 

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

##########################################################  

data_path = sys.argv[1]
target_path = sys.argv[2]
file_out = sys.argv[3]

with open(data_path) as fa:
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


with open(target_path) as fa:
    target_dict = {}
    for line in fa:
        # 去除末尾换行符
        line = line.replace('\n','')
        if line.startswith('>'):
            # 去除 > 号
            seq_name = line[1:]
            target_dict[seq_name] = ''
        else:
            # 去除末尾换行符并连接多行序列
            target_dict[seq_name] += line.replace('\n','')
            

f=open(file_out,"w")

wins=70
step=5

scores_dict={}
alignment1_dict={}
alignment2_dict={}
non_C_identity_dict={}
start_end_dict={}

for query in list(fa_dict.keys()):
    seq1 = fa_dict[query]
    
    for target in list(target_dict.keys()):
        seq2 = target_dict[target]
        #seq2_split = [seq2[i:i+wins] if i<len(seq2)-wins+step else '' for i in range(0,len(seq2)-wins+step,step)]
        seq2_split=[]
        seq2_split.append(seq2)
    
        scores_dict[query+target] = -1e5
        alignment1_dict[query+target] = ''
        alignment2_dict[query+target] = ''
        non_C_identity_dict[query+target] = 0
        start_end_dict[query+target] = ''
    
        for seq_i in range(len(seq2_split)):        
            scores,alignment1,alignment2,non_C_identity = BLAST(seq1, seq2_split[seq_i])
            start_end = str(seq_i*step)+'-'+str(seq_i*step+wins)
            if  scores > scores_dict[query+target] and non_C_identity > non_C_identity_dict[query+target]:
                scores_dict[query+target] = scores
                alignment1_dict[query+target] = alignment1
                alignment2_dict[query+target] = alignment2
                non_C_identity_dict[query+target] = non_C_identity
                start_end_dict[query+target] = start_end

#    f.write(list(fa_dict.keys())[0]+'\t'+target+'\t'+str(scores_dict[target])+'\t'+alignment1_dict[target]+'\t'+alignment2_dict[target]+'\t'+str(non_C_identity_dict[target])+'\t'+start_end_dict[target]+'\n')
            f.write(query+'\t'+target+'\t'+str(scores)+'\t'+alignment1+'\t'+alignment2+'\t'+str(non_C_identity)+'\t'+start_end+'\n')
f.close()
