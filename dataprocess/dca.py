import sys
import numpy as np
import string
import tensorflow as tf
import tqdm

paired_list = sys.argv[1]  # M234323_TS3343  243
msa_path = sys.argv[2] # msa path
out_path = sys.argv[3] # out name

def parse_a3m(a3mlines):
     seqs = []
     labels = []
     for line in a3mlines:
         if '>' in line[0]:
             labels.append(line[0])
         else:
             seqs.append(line[0])
     alphabet = np.array(list("ARNDCQEGHILKMFPSTWYV-"), dtype='|S1').view(np.uint8)
     seq_num = np.array([list(s) for s in seqs], dtype='|S1').view(np.uint8)
     
     for i in range(alphabet.shape[0]):
         seq_num[seq_num == alphabet[i]] = i
     seq_num[seq_num > 20] = 20
     return {'seqs' : seq_num, 'labels' : labels }

def tf_cov(x,w=None):
    if w is None:
        num_points = tf.cast(tf.shape(x)[0],tf.float32)-1
        x_mean = tf.reduce_mean(x, axis=0, keep_dims=True)
        x = (x - x_mean)
    else:
        num_points = tf.reduce_sum(w) - tf.sqrt(tf.reduce_mean(w))
        x_mean = tf.reduce_sum(x * w[:,None], axis=0, keepdims=True) / num_points
        x = (x - x_mean) * tf.sqrt(w[:,None])
    return tf.matmul(tf.transpose(x),x)/num_points

###################################################################################
#fp = open(sys.argv[1] + ".alignments", "r")
#pair2a3m = {}
#pair = ""
#pairs = []
#a3m = []
#for line in fp:
#    if line[:2] == ">>":
#        if pair and a3m:
#            pair2a3m[pair] = a3m
#            pairs.append(pair)
#            pair = line[2:-1]
#            a3m = []
#        else:
#            a3m.append(line)
#fp.close()

#if pair and a3m:
#    pair2a3m[pair] = a3m
#    pairs.append(pair)

#################################################################################

def read_a3m(msa_path, name):
    with open(msa_path+'/'+name+'.a3m', "r", encoding="utf-8") as f:
         a3mlines = [line[:-1].split("\t")
             for line in tqdm.tqdm(f, desc="Loading blast", total=None) if not line.startswith('#')]
    return a3mlines

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

def compute_covariance(x_feat, x_weights):
    x_c = tf.matmul(tf.transpose(x_feat), x_feat) / tf.reduce_sum(x_weights)
    x_c += tf.eye(x_c.shape[0]) * 4.5 / tf.sqrt(tf.reduce_sum(x_weights))
    x_c_inv = tf.linalg.inv(x_c)
    return x_c_inv

def compute_weights(x_c_inv, x_nc, x_ns):
    x_w = tf.reshape(x_c_inv, (x_nc, x_ns, x_nc, x_ns))
    x_wi = tf.sqrt(tf.reduce_sum(tf.square(x_w[..., :-1, :-1]), axis=[1, 3])) * (1 - tf.eye(x_nc))
    return x_wi

class MSAFeatureModel(tf.keras.Model):
    def __init__(self, x_ns):
        super(MSAFeatureModel, self).__init__()
        self.x_ns = x_ns

    def call(self, x):
        x_shape = tf.shape(x)
        x_nr = x_shape[0]
        x_nc = x_shape[1]
        x_msa = tf.one_hot(x,self.x_ns)
        x_cutoff = tf.cast(x_nc,tf.float32) * 0.8
        x_pw = tf.tensordot(x_msa, x_msa, [[1,2], [1,2]])
        x_cut = x_pw > x_cutoff
        x_weights = 1.0/tf.reduce_sum(tf.cast(x_cut, dtype=tf.float32),-1)
        x_feat = tf.reshape(x_msa,(x_nr,x_nc*self.x_ns))
        x_c = tf_cov(x_feat,x_weights) + tf.eye(x_nc*self.x_ns) * 4.5/tf.sqrt(tf.reduce_sum(x_weights))
        x_c_inv = tf.linalg.inv(x_c)
        x_w = tf.reshape(x_c_inv,(x_nc,self.x_ns,x_nc,self.x_ns))
        x_wi = tf.sqrt(tf.reduce_sum(tf.square(x_w[:,:-1,:,:-1]),(1,3))) * (1-tf.eye(x_nc))
        return x_wi

#############################################################

fp = open(paired_list, "r")
pairs = []
pairs_length = {}
for line in fp:
     line = line.strip()
     name = line.split('\t')[0]
     pairs_length[name] = line.split('\t')[1]
     
fp.close()


results = []
get_pairs = []
score_dict = {}

model = MSAFeatureModel(x_ns=21)

for item in list(pairs_length.keys()):
    a3mlines = read_a3m(msa_path,item)
    msa = parse_a3m(a3mlines) 
    wip = model(msa['seqs']).numpy().astype(np.float16)
    print(int(pairs_length[item]))
    print(wip.shape)
    max_score = np.amax(wip[:int(pairs_length[item]),int(pairs_length[item]):])
    score_dict[item] = max_score
    get_pairs.append(item)
    results.append(wip)

np.savez_compressed(out_path + '_compress', *results, names=get_pairs)

with open(out_path + ".log", "w") as rp:
    for pair in get_pairs:
        rp.write(pair+','+str(pairs_length[pair])+','+str(score_dict[pair])+'\n')
