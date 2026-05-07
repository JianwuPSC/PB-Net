"""
Train D-SCRIPT model with custom CSV data (train/valid/test) and sequence truncation.
Fully fixed device compatibility.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from typing import NamedTuple

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score as average_precision
from torch.autograd import Variable
from tqdm import tqdm

from dscript import __version__
from dscript.fasta import parse_dict
from dscript.foldseek import fold_vocab, get_foldseek_onehot
from dscript.glider import glide_compute_map, glider_score
from dscript.models.contact import ContactCNN
from dscript.models.embedding import FullyConnectedEmbed
from dscript.models.interaction import ModelInteraction
from dscript.utils import PairedDataset, collate_paired_sequences, log


class TrainArguments(NamedTuple):
    cmd: str
    device: int
    train_csv: str
    valid_csv: str | None
    test_csv: str
    embedding: str
    no_augment: bool
    input_dim: int
    projection_dim: int
    dropout: float
    hidden_dim: int
    kernel_width: int
    no_w: bool
    no_sigmoid: bool
    do_pool: bool
    pool_width: int
    num_epochs: int
    batch_size: int
    weight_decay: float
    lr: float
    interaction_weight: float
    run_tt: bool
    glider_weight: float
    glider_thresh: float
    outfile: str | None
    save_prefix: str | None
    checkpoint: str | None
    seed: int | None
    max_seq_len: int
    func: Callable[[TrainArguments], None]


def add_args(parser):
    """Add command line arguments."""
    data_grp = parser.add_argument_group("Data")
    proj_grp = parser.add_argument_group("Projection Module")
    contact_grp = parser.add_argument_group("Contact Module")
    inter_grp = parser.add_argument_group("Interaction Module")
    train_grp = parser.add_argument_group("Training")
    misc_grp = parser.add_argument_group("Output and Device")
    foldseek_grp = parser.add_argument_group("Foldseek related commands")

    data_grp.add_argument("--train-csv", required=True, help="training pairs (prot1, prot2, label)")
    data_grp.add_argument("--valid-csv", help="validation pairs (prot1, prot2, label)")
    data_grp.add_argument("--test-csv", required=True, help="test pairs (prot1, prot2, label)")
    data_grp.add_argument("--embedding", required=True, help="HDF5 embedding file")
    data_grp.add_argument("--no-augment", action="store_true", help="disable data augmentation")
    data_grp.add_argument("--max-seq-len", type=int, default=1500,
                          help="truncate sequences longer than this (default: 1500)")

    proj_grp.add_argument("--input-dim", type=int, default=6165, help="input embedding dimension")
    proj_grp.add_argument("--projection-dim", type=int, default=100, help="projection layer dimension")
    proj_grp.add_argument("--dropout-p", type=float, default=0.5, help="dropout probability")

    contact_grp.add_argument("--hidden-dim", type=int, default=50, help="contact CNN hidden dimension")
    contact_grp.add_argument("--kernel-width", type=int, default=7, help="contact CNN kernel width")

    inter_grp.add_argument("--no-w", action="store_true", help="disable weight matrix in interaction")
    inter_grp.add_argument("--no-sigmoid", action="store_true", help="disable sigmoid at end")
    inter_grp.add_argument("--do-pool", action="store_true", help="use max pooling in interaction")
    inter_grp.add_argument("--pool-width", type=int, default=15, help="max-pool width")

    train_grp.add_argument("--num-epochs", type=int, default=10)
    train_grp.add_argument("--batch-size", type=int, default=25)
    train_grp.add_argument("--weight-decay", type=float, default=0)
    train_grp.add_argument("--lr", type=float, default=0.001)
    train_grp.add_argument("--lambda", dest="interaction_weight", type=float, default=0.35,
                           help="weight on interaction loss")

    train_grp.add_argument("--topsy-turvy", dest="run_tt", action="store_true",
                           help="use GLIDER-derived supervision")
    train_grp.add_argument("--glider-weight", type=float, default=0.2,
                           help="GLIDER loss weight")
    train_grp.add_argument("--glider-thresh", type=float, default=0.925,
                           help="GLIDER positive threshold")

    misc_grp.add_argument("-o", "--outfile", help="output log file (default stdout)")
    misc_grp.add_argument("--save-prefix", help="prefix for saved models")
    misc_grp.add_argument("-d", "--device", type=int, default=-1, help="GPU device (-1 for CPU)")
    misc_grp.add_argument("--checkpoint", help="load model from checkpoint")
    misc_grp.add_argument("--seed", type=int, help="random seed")

    foldseek_grp.add_argument("--allow_foldseek", action="store_true", help="use foldseek one-hot encoding")
    foldseek_grp.add_argument("--foldseek_fasta", help="foldseek fasta file")

    return parser


def predict_cmap_interaction(model, n0, n1, tensors, use_cuda,
                             allow_foldseek=False, fold_record=None, fold_vocab=None, add_first=True):
    b = len(n0)
    p_hat = []
    c_map_mag = []
    for i in range(b):
        z_a = tensors[n0[i]]
        z_b = tensors[n1[i]]
        if use_cuda:
            z_a = z_a.cuda()
            z_b = z_b.cuda()
        if allow_foldseek:
            assert fold_record is not None and fold_vocab is not None
            f_a = get_foldseek_onehot(n0[i], z_a.shape[1], fold_record, fold_vocab).unsqueeze(0)
            f_b = get_foldseek_onehot(n1[i], z_b.shape[1], fold_record, fold_vocab).unsqueeze(0)
            if use_cuda:
                f_a = f_a.cuda()
                f_b = f_b.cuda()
            if add_first:
                z_a = torch.cat([z_a, f_a], dim=2)
                z_b = torch.cat([z_b, f_b], dim=2)
        if allow_foldseek and not add_first:
            cm, ph = model.map_predict(z_a, z_b, True, f_a, f_b)
        else:
            cm, ph = model.map_predict(z_a, z_b)
        p_hat.append(ph)
        c_map_mag.append(torch.mean(cm))
    p_hat = torch.stack(p_hat, 0)
    c_map_mag = torch.stack(c_map_mag, 0)
    return c_map_mag, p_hat


def predict_interaction(model, n0, n1, tensors, use_cuda,
                        allow_foldseek=False, fold_record=None, fold_vocab=None, add_first=True):
    _, p_hat = predict_cmap_interaction(model, n0, n1, tensors, use_cuda,
                                        allow_foldseek, fold_record, fold_vocab, add_first)
    return p_hat


def interaction_grad(model, n0, n1, y, tensors, accuracy_weight=0.35,
                     run_tt=False, glider_weight=0, glider_map=None, glider_mat=None,
                     use_cuda=True, allow_foldseek=False, fold_record=None, fold_vocab=None, add_first=True):
    c_map_mag, p_hat = predict_cmap_interaction(model, n0, n1, tensors, use_cuda,
                                                allow_foldseek, fold_record, fold_vocab, add_first)

    # Move y to same device as p_hat
    if use_cuda:
        y = y.cuda()
    y = Variable(y)

    p_hat = p_hat.float()
    bce_loss = F.binary_cross_entropy(p_hat.float(), y.float())

    if run_tt:
        g_score = []
        device = p_hat.device  # 统一设备
        for i in range(len(n0)):
            g_score.append(torch.tensor(glider_score(n0[i], n1[i], glider_map, glider_mat), dtype=torch.float64, device=device))
        g_score = torch.stack(g_score, 0)
        glider_loss = F.binary_cross_entropy(p_hat.float(), g_score.float())
        accuracy_loss = glider_weight * glider_loss + (1 - glider_weight) * bce_loss
    else:
        accuracy_loss = bce_loss

    representation_loss = torch.mean(c_map_mag)
    loss = accuracy_weight * accuracy_loss + (1 - accuracy_weight) * representation_loss
    b = len(p_hat)
    loss.backward()

    # 在计算 correct 和 mse 时，确保所有张量在同一设备
    with torch.no_grad():
        guess_cutoff = torch.tensor(0.5, device=p_hat.device).float()
        p_guess = (guess_cutoff > p_hat).float()  # 等价于 (guess_cutoff * torch.ones(b) < p_hat).float()
        y_float = y.float()  # y 已在 GPU
        correct = torch.sum(p_guess == y_float).item()
        mse = torch.mean((y_float - p_hat) ** 2).item()

    # 如果需要将张量移回 CPU 用于后续（可选，但保留原逻辑）
    if use_cuda:
        y = y.cpu()
        p_hat = p_hat.cpu()
        if run_tt:
            g_score = g_score.cpu()

    return loss, correct, mse, b


def interaction_eval(model, data_loader, tensors, use_cuda,
                     allow_foldseek=False, fold_record=None, fold_vocab=None, add_first=True):
    p_hat = []
    true_y = []
    device = next(model.parameters()).device  # 统一设备

    for n0, n1, y in data_loader:
        p_hat.append(predict_interaction(model, n0, n1, tensors, use_cuda,
                                         allow_foldseek, fold_record, fold_vocab, add_first))
        true_y.append(y)

    y = torch.cat(true_y, 0)
    p_hat = torch.cat(p_hat, 0)

    # 移动到模型设备
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

    # 移回 CPU 以计算 AUPR
    y_cpu = y.cpu().numpy()
    p_hat_cpu = p_hat.data.cpu().numpy()
    aupr = average_precision(y_cpu, p_hat_cpu)

    return loss, correct, mse, pr, re, f1, aupr


def train_model(args, output):
    batch_size = args.batch_size
    use_cuda = (args.device > -1) and torch.cuda.is_available()
    no_augment = args.no_augment

    # ---- Load CSV files ----
    train_df = pd.read_csv(args.train_csv, sep='\t', header=None)
    train_df.columns = ['prot1', 'prot2', 'label']
    if no_augment:
        train_p1 = train_df['prot1']
        train_p2 = train_df['prot2']
        train_y = torch.from_numpy(train_df['label'].values.copy())
    else:
        train_p1 = pd.concat([train_df['prot1'], train_df['prot2']], axis=0).reset_index(drop=True)
        train_p2 = pd.concat([train_df['prot2'], train_df['prot1']], axis=0).reset_index(drop=True)
        train_y = torch.from_numpy(pd.concat([train_df['label'], train_df['label']]).values.copy())

    train_dataset = PairedDataset(train_p1, train_p2, train_y)
    train_iterator = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                                 collate_fn=collate_paired_sequences, shuffle=True)
    log(f"Loaded {len(train_p1)} training pairs", file=output)
    output.flush()

    if args.valid_csv:
        valid_df = pd.read_csv(args.valid_csv, sep='\t', header=None)
        valid_df.columns = ['prot1', 'prot2', 'label']
        valid_p1 = valid_df['prot1']
        valid_p2 = valid_df['prot2']
        valid_y = torch.from_numpy(valid_df['label'].values.copy())
        valid_dataset = PairedDataset(valid_p1, valid_p2, valid_y)
        valid_iterator = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size,
                                                     collate_fn=collate_paired_sequences, shuffle=False)
        log(f"Loaded {len(valid_p1)} validation pairs", file=output)
        output.flush()
    else:
        valid_iterator = None

    test_df = pd.read_csv(args.test_csv, sep='\t', header=None)
    test_df.columns = ['prot1', 'prot2', 'label']
    test_p1 = test_df['prot1']
    test_p2 = test_df['prot2']
    test_y = torch.from_numpy(test_df['label'].values.copy())
    test_dataset = PairedDataset(test_p1, test_p2, test_y)
    test_iterator = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                                collate_fn=collate_paired_sequences, shuffle=False)
    log(f"Loaded {len(test_p1)} test pairs", file=output)
    output.flush()

    # ---- Determine all proteins ----
    all_proteins = set(train_p1).union(train_p2)
    if valid_iterator:
        all_proteins = all_proteins.union(valid_p1).union(valid_p2)
    all_proteins = all_proteins.union(test_p1).union(test_p2)
    log(f"Total unique proteins: {len(all_proteins)}", file=output)

    # ---- Load embeddings with truncation ----
    log(f"Loading embeddings (max length = {args.max_seq_len})...", file=output)
    output.flush()
    embeddings = {}
    truncated_count = 0
    with h5py.File(args.embedding, "r") as h5fi:
        for prot_name in tqdm(all_proteins):
            try:
                embed = h5fi[prot_name][:, :]
            except KeyError:
                log(f"Warning: {prot_name} not found in embedding file. Skipping.", file=output)
                continue
            original_len = embed.shape[0]
            if original_len > args.max_seq_len:
                embed = embed[:args.max_seq_len, :].copy()
                truncated_count += 1
                log(f"  Truncated {prot_name}: {original_len} -> {args.max_seq_len}", file=output)
            else:
                embed = embed.copy()
            embeddings[prot_name] = torch.from_numpy(embed)
            assert embeddings[prot_name].shape[0] <= args.max_seq_len, \
                f"Length {embeddings[prot_name].shape[0]} > {args.max_seq_len} for {prot_name}"

    log(f"Loaded embeddings for {len(embeddings)} proteins (truncated {truncated_count})", file=output)
    output.flush()

    # ---- Topsy-Turvy (optional) ----
    run_tt = args.run_tt
    glider_weight = args.glider_weight
    glider_thresh = args.glider_thresh * 100
    if run_tt:
        log("Running D-SCRIPT Topsy-Turvy mode...", file=output)
        log(f"  glider_weight = {glider_weight}", file=output)
        log(f"  glider_thresh = {glider_thresh}th percentile", file=output)
        log("Computing GLIDER matrix...", file=output)
        output.flush()
        glider_mat, glider_map = glide_compute_map(train_df[train_df['label'] == 1], thres_p=glider_thresh)
    else:
        glider_mat, glider_map = None, None

    # ---- Foldseek (optional) ----
    allow_foldseek = args.allow_foldseek
    fold_fasta_file = args.foldseek_fasta
    add_first = False
    fold_record = {}
    if allow_foldseek:
        assert fold_fasta_file is not None, "Foldseek fasta file required when --allow_foldseek is used"
        fold_fasta = parse_dict(fold_fasta_file)
        for rec_k, rec_v in fold_fasta.items():
            fold_record[rec_k] = rec_v
        log("Foldseek one-hot encoding enabled.", file=output)

    # ---- Model initialization ----
    if args.checkpoint is None:
        input_dim = args.input_dim
        if allow_foldseek and add_first:
            input_dim += len(fold_vocab)
        projection_dim = args.projection_dim
        dropout_p = args.dropout_p
        embedding_model = FullyConnectedEmbed(input_dim, projection_dim, dropout=dropout_p)
        log("Initializing embedding model:", file=output)
        log(f"  projection_dim = {projection_dim}", file=output)
        log(f"  dropout_p = {dropout_p}", file=output)

        hidden_dim = args.hidden_dim
        kernel_width = args.kernel_width
        log("Initializing contact model:", file=output)
        log(f"  hidden_dim = {hidden_dim}", file=output)
        log(f"  kernel_width = {kernel_width}", file=output)

        proj_dim = projection_dim
        if allow_foldseek and not add_first:
            proj_dim += len(fold_vocab)
        contact_model = ContactCNN(proj_dim, hidden_dim, kernel_width)

        do_w = not args.no_w
        do_pool = args.do_pool
        pool_width = args.pool_width
        do_sigmoid = not args.no_sigmoid
        log("Initializing interaction model:", file=output)
        log(f"  do_pool = {do_pool}", file=output)
        log(f"  pool_width = {pool_width}", file=output)
        log(f"  do_w = {do_w}", file=output)
        log(f"  do_sigmoid = {do_sigmoid}", file=output)
        model = ModelInteraction(embedding_model, contact_model, use_cuda,
                                 do_w=do_w, pool_size=pool_width, do_pool=do_pool, do_sigmoid=do_sigmoid)
    else:
        log(f"Loading model from checkpoint {args.checkpoint}", file=output)
        model = torch.load(args.checkpoint)
        model.use_cuda = use_cuda

    if use_cuda:
        model.cuda()
        log("Model moved to GPU.", file=output)

    # ---- Optimizer ----
    params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    log(f"Optimizer: Adam, lr={args.lr}, weight_decay={args.weight_decay}", file=output)

    # ---- Training loop ----
    digits = int(np.floor(np.log10(args.num_epochs))) + 1
    best_valid_aupr = -1
    best_epoch = -1

    for epoch in range(args.num_epochs):
        model.train()
        n = 0
        loss_accum = 0
        acc_accum = 0
        mse_accum = 0

        for z0, z1, y in train_iterator:
            loss, correct, mse, b = interaction_grad(
                model, z0, z1, y, embeddings,
                accuracy_weight=args.interaction_weight,
                run_tt=run_tt, glider_weight=glider_weight,
                glider_map=glider_map, glider_mat=glider_mat,
                use_cuda=use_cuda,
                allow_foldseek=allow_foldseek, fold_record=fold_record,
                fold_vocab=fold_vocab, add_first=add_first
            )
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

        # Evaluation after each epoch
        model.eval()
        with torch.no_grad():
            if valid_iterator is not None:
                valid_loss, valid_correct, valid_mse, valid_pr, valid_re, valid_f1, valid_aupr = interaction_eval(
                    model, valid_iterator, embeddings, use_cuda,
                    allow_foldseek, fold_record, fold_vocab, add_first
                )
                valid_acc = valid_correct / len(valid_dataset)
                log(f"Epoch {epoch+1} Validation: Loss={valid_loss:.6f}, Acc={valid_acc:.3%}, AUPR={valid_aupr:.6f}", file=output)
                output.flush()
                if valid_aupr > best_valid_aupr:
                    best_valid_aupr = valid_aupr
                    best_epoch = epoch + 1
                    if args.save_prefix:
                        best_path = args.save_prefix + "_best.sav"
                        log(f"New best model (AUPR={valid_aupr:.6f}) -> {best_path}", file=output)
                        model.cpu()
                        torch.save(model, best_path)
                        if use_cuda:
                            model.cuda()
            else:
                valid_aupr = 0

            test_loss, test_correct, test_mse, test_pr, test_re, test_f1, test_aupr = interaction_eval(
                model, test_iterator, embeddings, use_cuda,
                allow_foldseek, fold_record, fold_vocab, add_first
            )
            test_acc = test_correct / len(test_dataset)
            log(f"Epoch {epoch+1} Test: Loss={test_loss:.6f}, Acc={test_acc:.3%}, AUPR={test_aupr:.6f}", file=output)
            output.flush()

        # Save epoch checkpoint
        if args.save_prefix:
            save_path = args.save_prefix + "_epoch" + str(epoch + 1).zfill(digits) + ".sav"
            log(f"Saving model to {save_path}", file=output)
            model.cpu()
            torch.save(model, save_path)
            if use_cuda:
                model.cuda()

    # Save final model
    if args.save_prefix:
        final_path = args.save_prefix + "_final.sav"
        log(f"Saving final model to {final_path}", file=output)
        model.cpu()
        torch.save(model, final_path)
        if use_cuda:
            model.cuda()

    log("Training completed.", file=output)
    output.flush()


def main(args):
    output = args.outfile
    if output is None:
        output = sys.stdout
    else:
        output = open(output, "w")

    log(f"D-SCRIPT Version {__version__}", file=output, print_also=True)
    log(f"Called as: {' '.join(sys.argv)}", file=output, print_also=True)

    device = args.device
    use_cuda = (device > -1) and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(device)
        log(f"Using CUDA device {device} - {torch.cuda.get_device_name(device)}", file=output, print_also=True)
    else:
        log("Using CPU", file=output, print_also=True)
        device = "cpu"

    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        log(f"Random seed set to {args.seed}", file=output, print_also=True)

    train_model(args, output)

    output.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_args(parser)
    main(parser.parse_args())
