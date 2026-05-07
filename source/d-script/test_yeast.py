"""
Evaluate a trained model (compatible with train_yeast.py).
Supports multi-threshold metrics and PR curve data export.
"""

import sys
import os
import argparse
import datetime
import numpy as np
import pandas as pd
import torch
import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    roc_curve,
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from tqdm import tqdm

# 添加项目路径（如果需要）
sys.path.append("/home/wuj/data/tools/D-SCRIPT/")


def add_args(parser):
    """
    Create parser for command line utility.
    """
    parser.add_argument("--model", required=True, help="Trained prediction model (.sav file)")
    parser.add_argument("--test", required=True, help="Test data (TSV: prot1\tprot2\tlabel)")
    parser.add_argument("--embedding", required=True, help="HDF5 file with embedded sequences")
    parser.add_argument("--max-seq-len", type=int, default=2000, help="Maximum sequence length (must match training)")
    parser.add_argument("--manual-threshold", type=float, default=0.5, help="Threshold for manual classification (default: 0.5)")
    parser.add_argument("-o", "--outfile", help="Output file prefix (default: timestamp)")
    parser.add_argument("-d", "--device", type=int, default=-1, help="Compute device (-1 for CPU)")
    return parser


def compute_metrics_for_threshold(y_true, y_score, threshold):
    """Compute classification metrics at a given threshold."""
    y_pred = (y_score >= threshold).astype(int)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    rec = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    return acc, prec, rec, f1


def generate_threshold_metrics(y_true, y_score):
    """Generate metrics for thresholds from 0.0 to 1.0 (step 0.01)."""
    thresholds = np.arange(0.0, 1.01, 0.01)
    metrics_list = []
    for thr in thresholds:
        acc, prec, rec, f1 = compute_metrics_for_threshold(y_true, y_score, thr)
        metrics_list.append({
            'threshold': round(thr, 2),
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'accuracy': acc
        })
    metrics_df = pd.DataFrame(metrics_list)
    return metrics_df


def plot_eval_predictions(labels, predictions, path="figure"):
    """
    Plot histogram, precision-recall curve, and ROC curve.
    """
    pos_phat = predictions[labels == 1]
    neg_phat = predictions[labels == 0]

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle("Distribution of Predictions")
    ax1.hist(pos_phat, bins=50)
    ax1.set_xlim(0, 1)
    ax1.set_title("Positive")
    ax1.set_xlabel("p-hat")
    ax2.hist(neg_phat, bins=50)
    ax2.set_xlim(0, 1)
    ax2.set_title("Negative")
    ax2.set_xlabel("p-hat")
    plt.savefig(path + ".phat_dist.png")
    plt.close()

    precision, recall, _ = precision_recall_curve(labels, predictions)
    aupr = average_precision_score(labels, predictions)
    print("AUPR:", aupr)

    pr_df = pd.DataFrame({"recall": recall, "precision": precision})
    pr_df.to_csv(path + "_pr_curve.csv", index=False)

    plt.step(recall, precision, color="b", alpha=0.2, where="post")
    plt.fill_between(recall, precision, step="post", alpha=0.2, color="b")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title(f"Precision-Recall (AUPR: {aupr:.3f})")
    plt.savefig(path + ".aupr.png")
    plt.close()

    fpr, tpr, _ = roc_curve(labels, predictions)
    auroc = roc_auc_score(labels, predictions)
    print("AUROC:", auroc)

    plt.step(fpr, tpr, color="b", alpha=0.2, where="post")
    plt.fill_between(fpr, tpr, step="post", alpha=0.2, color="b")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title(f"ROC Curve (AUROC: {auroc:.3f})")
    plt.savefig(path + ".auroc.png")
    plt.close()


def main(args):
    # Set device
    device = args.device
    use_cuda = (device >= 0) and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(device)
        print(f"# Using CUDA device {device} - {torch.cuda.get_device_name(device)}")
    else:
        print("# Using CPU")

    # Load the entire model (trained with torch.save)
    print("Loading model...")
    model = torch.load(args.model, map_location="cpu")
    if use_cuda:
        model = model.cuda()
    else:
        model = model.cpu()
    model.eval()

    # Load embeddings
    print("Loading embeddings...")
    h5fi = h5py.File(args.embedding, "r")
    test_df = pd.read_csv(args.test, sep="\t", header=None)
    test_df.columns = ["prot1", "prot2", "label"]  # label is 0/1

    all_proteins = set(test_df["prot1"]).union(test_df["prot2"])
    seq_emb_dict = {}
    for prot in tqdm(all_proteins, desc="Loading embeddings"):
        embed = h5fi[prot][:, :]  # shape: (L, dim)
        # Truncate if needed (must match training)
        if embed.shape[0] > args.max_seq_len:
            embed = embed[:args.max_seq_len, :].copy()
        seq_emb_dict[prot] = torch.from_numpy(embed).float()

    # Prepare output
    if args.outfile is None:
        out_prefix = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
    else:
        out_prefix = args.outfile
    out_file = open(out_prefix + ".predictions.tsv", "w")

    predictions = []
    labels = []

    print("Predicting...")
    with torch.no_grad():
        for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Predicting pairs"):
            n0, n1, label = row["prot1"], row["prot2"], row["label"]
            try:
                p0 = seq_emb_dict[n0]
                p1 = seq_emb_dict[n1]
                if use_cuda:
                    p0 = p0.cuda()
                    p1 = p1.cuda()
                # Model's predict method returns scalar probability
                pred = model.predict(p0, p1).item()
                predictions.append(pred)
                labels.append(label)
                out_file.write(f"{n0}\t{n1}\t{pred:.6f}\t{label}\n")
            except Exception as e:
                sys.stderr.write(f"Error with {n0} x {n1}: {e}\n")

    out_file.close()
    h5fi.close()

    predictions = np.array(predictions)
    labels = np.array(labels)

    # ---- Multi-threshold analysis ----
    metrics_df = generate_threshold_metrics(labels, predictions)
    threshold_metrics_path = out_prefix + "_threshold_metrics.csv"
    metrics_df.to_csv(threshold_metrics_path, index=False)
    print(f"Threshold metrics saved to {threshold_metrics_path}")

    # ---- Manual threshold classification ----
    manual_thr = args.manual_threshold
    acc, prec, rec, f1 = compute_metrics_for_threshold(labels, predictions, manual_thr)
    print("\n========== Manual Threshold Classification ==========")
    print(f"Threshold = {manual_thr:.2f}")
    print(f"Accuracy  = {acc:.4f}")
    print(f"Precision = {prec:.4f}")
    print(f"Recall    = {rec:.4f}")
    print(f"F1 Score  = {f1:.4f}")

    # ---- Best F1 threshold ----
    best_idx = metrics_df['f1'].idxmax()
    best_row = metrics_df.loc[best_idx]
    print("\n========== Best F1 Threshold ==========")
    print(f"Threshold = {best_row['threshold']:.2f}")
    print(f"Precision = {best_row['precision']:.4f}")
    print(f"Recall    = {best_row['recall']:.4f}")
    print(f"F1 Score  = {best_row['f1']:.4f}")
    print(f"Accuracy  = {best_row['accuracy']:.4f}")

    # ---- Overall AUPR / AUROC (already printed inside plot function) ----
    # ---- Generate plots ----
    plot_eval_predictions(labels, predictions, out_prefix)

    # ---- Save numerical metrics to file ----
    aupr = average_precision_score(labels, predictions)
    auroc = roc_auc_score(labels, predictions)
    with open(out_prefix + ".metrics.txt", "w") as f:
        f.write(f"AUPR: {aupr}\nAUROC: {auroc}\n")
        f.write(f"Manual threshold ({manual_thr}): Acc={acc}, Prec={prec}, Rec={rec}, F1={f1}\n")
        f.write(f"Best F1 threshold ({best_row['threshold']}): Prec={best_row['precision']}, Rec={best_row['recall']}, F1={best_row['f1']}, Acc={best_row['accuracy']}\n")

    print(f"\nAll outputs saved with prefix: {out_prefix}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_args(parser)
    main(parser.parse_args())
