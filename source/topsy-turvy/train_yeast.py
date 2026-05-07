"""
Modified Topsy-Turvy training script with custom TSV input (train/valid/test).
Supports --use_glider for GLIDER-based supervision.
Input files: TSV with columns: prot1\tprot2\tlabel (no header)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

from sklearn.metrics import average_precision_score as average_precision

import sys
import argparse
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

import topsy_turvy.src.fasta as fa
from topsy_turvy.src.utils import PairedDataset, collate_paired_sequences
from topsy_turvy.src.models.embedding import FullyConnectedEmbed
from topsy_turvy.src.models.contact import ContactCNN
from topsy_turvy.src.models.interaction import ModelInteraction
from topsy_turvy.src.models.glider import glide_compute_map


def predict_interaction(model, n0, n1, tensors, use_cuda):
    b = len(n0)
    p_hat = []
    for i in range(b):
        z_a = tensors[n0[i]]
        z_b = tensors[n1[i]]
        if use_cuda:
            z_a = z_a.cuda()
            z_b = z_b.cuda()
        p_hat.append(model.predict(z_a, z_b))
    p_hat = torch.stack(p_hat, 0)
    return p_hat


def predict_cmap_interaction(model, n0, n1, tensors, use_cuda):
    b = len(n0)
    p_hat = []
    c_map_mag = []
    for i in range(b):
        z_a = tensors[n0[i]]
        z_b = tensors[n1[i]]
        if use_cuda:
            z_a = z_a.cuda()
            z_b = z_b.cuda()
        cm, ph = model.map_predict(z_a, z_b)
        p_hat.append(ph)
        c_map_mag.append(torch.mean(cm))
    p_hat = torch.stack(p_hat, 0)
    c_map_mag = torch.stack(c_map_mag, 0)
    return c_map_mag, p_hat


def interaction_grad(model, n0, n1, y, tensors, use_cuda, weight=0.35):
    c_map_mag, p_hat = predict_cmap_interaction(model, n0, n1, tensors, use_cuda)
    if use_cuda:
        y = y.cuda()
    y = Variable(y)

    if model.use_glider:
        g_score = []
        for i in range(len(n0)):
            g_score.append(torch.tensor(model.glider_score(n0[i], n1[i]), dtype=torch.float64))
        g_score = torch.stack(g_score, 0)
        if use_cuda:
            g_score = g_score.cuda()

    bce_loss = F.binary_cross_entropy(p_hat.float(), y.float())
    combined_loss = bce_loss

    if model.use_glider:
        glider_loss = F.binary_cross_entropy(p_hat.float(), g_score.float())
        combined_loss = model.glider_param * glider_loss + (1 - model.glider_param) * bce_loss

    cmap_loss = torch.mean(c_map_mag)
    loss = weight * combined_loss + (1 - weight) * cmap_loss
    b = len(p_hat)
    loss.backward()

    if use_cuda:
        y = y.cpu()
        p_hat = p_hat.cpu()

    with torch.no_grad():
        guess_cutoff = 0.5
        p_hat = p_hat.float()
        p_guess = (guess_cutoff * torch.ones(b) < p_hat).float()
        y = y.float()
        correct = torch.sum(p_guess == y).item()
        mse = torch.mean((y.float() - p_hat) ** 2).item()

    return loss, correct, mse, b


def interaction_eval(model, data_loader, tensors, use_cuda):
    p_hat = []
    true_y = []
    device = next(model.parameters()).device

    for n0, n1, y in data_loader:
        p_hat.append(predict_interaction(model, n0, n1, tensors, use_cuda))
        true_y.append(y)

    y = torch.cat(true_y, 0)
    p_hat = torch.cat(p_hat, 0)

    y = y.to(device)
    p_hat = p_hat.to(device)

    loss = F.binary_cross_entropy(p_hat.float(), y.float()).item()
    b = len(y)

    with torch.no_grad():
        guess_cutoff = torch.tensor(0.5, device=device).float()
        p_guess = (guess_cutoff > p_hat).float()
        correct = torch.sum(p_guess == y).item()
        mse = torch.mean((y.float() - p_hat) ** 2).item()

        tp = torch.sum(y * p_hat).item()
        pr = tp / torch.sum(p_hat).item() if torch.sum(p_hat).item() > 0 else 0
        re = tp / torch.sum(y).item() if torch.sum(y).item() > 0 else 0
        f1 = 2 * pr * re / (pr + re) if (pr + re) > 0 else 0

    y = y.cpu().numpy()
    p_hat = p_hat.data.cpu().numpy()
    aupr = average_precision(y, p_hat)

    return loss, correct, mse, pr, re, f1, aupr


def load_data_from_tsv(tsv_path, augment=False):
    """
    Load TSV file with columns: prot1\tprot2\tlabel (no header).
    Returns: (dataset, prot1_list, prot2_list)
    """
    df = pd.read_csv(tsv_path, sep='\t', header=None)
    df.columns = ['prot1', 'prot2', 'label']
    if augment:
        prot1 = pd.concat([df['prot1'], df['prot2']], axis=0).reset_index(drop=True)
        prot2 = pd.concat([df['prot2'], df['prot1']], axis=0).reset_index(drop=True)
        labels = torch.from_numpy(pd.concat([df['label'], df['label']]).values.copy())
    else:
        prot1 = df['prot1']
        prot2 = df['prot2']
        labels = torch.from_numpy(df['label'].values.copy())
    dataset = PairedDataset(prot1, prot2, labels)
    return dataset, prot1.tolist(), prot2.tolist()


def parse_args():
    parser = argparse.ArgumentParser('Training protein interaction model with TSV input')

    # Data
    parser.add_argument('--train-tsv', required=True, help='Training pairs TSV (prot1\tprot2\tlabel)')
    parser.add_argument('--valid-tsv', help='Validation pairs TSV (optional)')
    parser.add_argument('--test-tsv', required=True, help='Test pairs TSV (prot1\tprot2\tlabel)')
    parser.add_argument('--embedding', required=True, help='HDF5 embedding file')
    parser.add_argument('--augment', action='store_true', help='Augment data by swapping pairs')
    parser.add_argument('--max-seq-len', type=int, default=800, help='Truncate sequences longer than this')

    # Embedding model
    parser.add_argument('--dumb-embed-switch', action='store_true', help='Use last 100 dims of embedding')
    parser.add_argument('--projection-dim', type=int, default=100, help='Projection layer dimension')
    parser.add_argument('--dropout-p', type=float, default=0.2, help='Dropout probability')

    # Contact model
    parser.add_argument('--hidden-dim', type=int, default=50, help='Hidden units in contact CNN')
    parser.add_argument('--kernel-width', type=int, default=7, help='Convolution kernel width')

    # Interaction model
    parser.add_argument('--use-w', action='store_true', help='Use weight matrix in interaction')
    parser.add_argument('--do-pool', action='store_true', help='Use max pooling')
    parser.add_argument('--pool-width', type=int, default=9, help='Max-pool width')
    parser.add_argument('--sigmoid', action='store_true', help='Use sigmoid activation')

    # Training
    parser.add_argument('--num-epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=25)
    parser.add_argument('--weight-decay', type=float, default=0)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lambda', dest='lambda_', type=float, default=0.05, help='Weight on similarity objective')
    parser.add_argument('--epoch-scale', type=int, default=1, help='Report validation/test every N epochs')

    # GLIDER
    parser.add_argument('--use_glider', action='store_true', help='Use GLIDER supervision')
    parser.add_argument('--glider_param', type=float, default=0.2, help='GLIDER loss weight')
    parser.add_argument('--glider_thresh', type=float, default=92.5, help='GLIDER percentile threshold')

    # Output
    parser.add_argument('-o', '--output', help='Output log file')
    parser.add_argument('--save-prefix', required=True, help='Prefix for saved models')
    parser.add_argument('-d', '--device', type=int, default=-1, help='GPU device (-1 for CPU)')
    parser.add_argument('--checkpoint', help='Load model from checkpoint')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()
    return args


def load_embeddings_with_truncation(embedding_h5, all_proteins, max_seq_len, dumb_embed_switch):
    h5fi = h5py.File(embedding_h5, 'r')
    tensors = {}
    for prot_name in tqdm(all_proteins, desc='Loading embeddings'):
        embed = h5fi[prot_name][:, :]
        if embed.shape[0] > max_seq_len:
            embed = embed[:max_seq_len, :].copy()
        if dumb_embed_switch:
            embed = embed[:, -100:]
        tensors[prot_name] = torch.from_numpy(embed).float()
    h5fi.close()
    return tensors


def main():
    args = parse_args()

    # Set output and logging
    output = args.output
    if output is None:
        output = sys.stdout
    else:
        output = open(output, 'w')

    print(f'# Called as: {" ".join(sys.argv)}', file=output)
    if output is not sys.stdout:
        print(f'Called as: {" ".join(sys.argv)}')

    # Random seed
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    # Device
    use_cuda = (args.device >= 0) and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(args.device)
        print(f'# Using CUDA device {args.device} - {torch.cuda.get_device_name(args.device)}', file=output)
    else:
        print('# Using CPU', file=output)
        args.device = 'cpu'

    # Load datasets from TSV files
    print('# Loading data...', file=output)
    train_dataset, train_p1, train_p2 = load_data_from_tsv(args.train_tsv, augment=args.augment)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              collate_fn=collate_paired_sequences, shuffle=True)

    # Validation set (optional)
    valid_loader = None
    valid_p1, valid_p2 = [], []
    if args.valid_tsv:
        valid_dataset, valid_p1, valid_p2 = load_data_from_tsv(args.valid_tsv, augment=False)
        valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size,
                                  collate_fn=collate_paired_sequences, shuffle=False)
    else:
        valid_dataset = None

    test_dataset, test_p1, test_p2 = load_data_from_tsv(args.test_tsv, augment=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             collate_fn=collate_paired_sequences, shuffle=False)

    # Collect all proteins
    all_proteins = set(train_p1).union(train_p2).union(test_p1).union(test_p2)
    if valid_loader is not None:
        all_proteins.update(valid_p1)
        all_proteins.update(valid_p2)

    print(f'# Total unique proteins: {len(all_proteins)}', file=output)

    # Load embeddings with truncation
    print('# Loading embeddings...', file=output)
    tensors = load_embeddings_with_truncation(args.embedding, all_proteins,
                                              args.max_seq_len, args.dumb_embed_switch)

    # GLIDER preparation (only if training with GLIDER)
    if args.use_glider:
        # Build positive pairs for GLIDER matrix from training set
        train_df = pd.read_csv(args.train_tsv, sep='\t', header=None)
        train_df.columns = ['prot1', 'prot2', 'label']
        pos_pairs = train_df[train_df['label'] == 1][['prot1', 'prot2']].values
        pos_pairs_list = [(p[0], p[1], 1) for p in pos_pairs]
        if args.augment:  # also add swapped pairs
            pos_pairs_list += [(p[1], p[0], 1) for p in pos_pairs]
        print('# Computing GLIDER matrix...', file=output)
        glider_mat, glider_map = glide_compute_map(pos_pairs_list, thres_p=args.glider_thresh)
    else:
        glider_mat, glider_map = None, None

    # Model initialization
    if args.checkpoint is None:
        if args.dumb_embed_switch:
            print('# Using last 100 dim of embeddings (dumb embedding)', file=output)
            embedding = None
            projection_dim = 100
        else:
            projection_dim = args.projection_dim
            dropout_p = args.dropout_p
            embedding = FullyConnectedEmbed(6165, projection_dim, dropout=dropout_p)
            print('# Initializing embedding model:', file=output)
            print(f'\tprojection_dim: {projection_dim}', file=output)
            print(f'\tdropout_p: {dropout_p}', file=output)

        hidden_dim = args.hidden_dim
        kernel_width = args.kernel_width
        contact = ContactCNN(projection_dim, hidden_dim, kernel_width)
        print('# Initializing contact model:', file=output)
        print(f'\thidden_dim: {hidden_dim}', file=output)
        print(f'\tkernel_width: {kernel_width}', file=output)

        use_W = args.use_w
        do_pool = args.do_pool
        pool_width = args.pool_width
        sigmoid = args.sigmoid
        print('# Initializing interaction model:', file=output)
        if do_pool:
            print(f'\tdo_pool: {do_pool}', file=output)
            print(f'\tpool_width: {pool_width}', file=output)
        print(f'\tuse_w: {use_W}', file=output)
        print(f'\tsigmoid: {sigmoid}', file=output)

        model = ModelInteraction(embedding, contact, use_cuda,
                                 use_W=use_W, pool_size=pool_width, do_pool=do_pool,
                                 do_sigmoid=sigmoid, use_glider=args.use_glider,
                                 glider_map=glider_map, glider_mat=glider_mat,
                                 glider_param=args.glider_param)
        print(model, file=output)
    else:
        print(f'# Loading model from checkpoint {args.checkpoint}', file=output)
        model = torch.load(args.checkpoint, map_location='cpu')
        model.use_cuda = use_cuda

    if use_cuda:
        model.cuda()

    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    inter_weight = args.lambda_
    cmap_weight = 1 - inter_weight

    print(f'# Using save prefix "{args.save_prefix}"', file=output)
    print(f'# Training with Adam: lr={args.lr}, weight_decay={args.weight_decay}', file=output)
    print(f'\tnum_epochs: {args.num_epochs}', file=output)
    print(f'\tbatch_size: {args.batch_size}', file=output)
    print(f'\tinteraction weight: {inter_weight}', file=output)
    print(f'\tcontact map weight: {cmap_weight}', file=output)
    output.flush()

    digits = int(np.floor(np.log10(args.num_epochs))) + 1
    best_valid_aupr = -1
    best_epoch = -1

    for epoch in range(args.num_epochs):
        model.train()
        n = 0
        loss_accum = 0
        acc_accum = 0
        mse_accum = 0

        for z0, z1, y in train_loader:
            loss, correct, mse, b = interaction_grad(model, z0, z1, y, tensors, use_cuda,
                                                      weight=inter_weight)
            n += b
            delta = b * (loss - loss_accum)
            loss_accum += delta / n
            delta = correct - b * acc_accum
            acc_accum += delta / n
            delta = b * (mse - mse_accum)
            mse_accum += delta / n

            optim.step()
            optim.zero_grad()
            model.clip()

            # Progress report every 100 batches
            if ((n - b) // 100) < (n // 100):
                print(f'# [{epoch+1}/{args.num_epochs}] training {n / len(train_loader.dataset):.1%}: '
                      f'Loss={loss_accum:.6f}, Acc={acc_accum:.3%}, MSE={mse_accum:.6f}', file=output)
                output.flush()

        # Evaluation on validation and test (every epoch_scale epochs)
        if (epoch + 1) % args.epoch_scale == 0:
            model.eval()
            with torch.no_grad():
                # Validation
                if valid_loader is not None:
                    v_loss, v_correct, v_mse, v_pr, v_re, v_f1, v_aupr = interaction_eval(
                        model, valid_loader, tensors, use_cuda
                    )
                    v_acc = v_correct / len(valid_dataset)
                    print(f'# Epoch {epoch+1} Validation: Loss={v_loss:.6f}, Acc={v_acc:.3%}, '
                          f'Prec={v_pr:.6f}, Rec={v_re:.6f}, F1={v_f1:.6f}, AUPR={v_aupr:.6f}', file=output)
                    output.flush()

                    if v_aupr > best_valid_aupr:
                        best_valid_aupr = v_aupr
                        best_epoch = epoch + 1
                        best_path = args.save_prefix + '_best.sav'
                        print(f'# New best model (AUPR={v_aupr:.6f}) -> {best_path}', file=output)
                        model.cpu()
                        torch.save(model, best_path)
                        if use_cuda:
                            model.cuda()

                # Test
                t_loss, t_correct, t_mse, t_pr, t_re, t_f1, t_aupr = interaction_eval(
                    model, test_loader, tensors, use_cuda
                )
                t_acc = t_correct / len(test_dataset)
                print(f'# Epoch {epoch+1} Test: Loss={t_loss:.6f}, Acc={t_acc:.3%}, '
                      f'Prec={t_pr:.6f}, Rec={t_re:.6f}, F1={t_f1:.6f}, AUPR={t_aupr:.6f}', file=output)
                output.flush()

            # Save epoch checkpoint
            if args.save_prefix:
                epoch_path = args.save_prefix + f'_epoch{str(epoch+1).zfill(digits)}.sav'
                print(f'# Saving model to {epoch_path}', file=output)
                model.cpu()
                torch.save(model, epoch_path)
                if use_cuda:
                    model.cuda()

        output.flush()

    # Save final model
    if args.save_prefix:
        final_path = args.save_prefix + '_final.sav'
        print(f'# Saving final model to {final_path}', file=output)
        model.cpu()
        torch.save(model, final_path)
        if use_cuda:
            model.cuda()

    print('# Training completed.', file=output)
    output.close()


if __name__ == '__main__':
    main()
