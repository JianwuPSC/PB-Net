import sys, os
import argparse
import torch
import h5py
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, average_precision_score, roc_curve, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')

def plot_eval_predictions(pos_phat, neg_phat, plot_prediction_distributions=True, plot_curves=True, path='figure'):
    """Plot distribution of predictions, PR curve, and ROC curve."""
    print('Plotting Curves')
    if plot_prediction_distributions:
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.suptitle('Distribution of Predictions')
        ax1.hist(pos_phat, bins=50)
        ax1.set_xlim(0,1)
        ax1.set_title("Positive")
        ax1.set_xlabel("p-hat")
        ax2.hist(neg_phat, bins=50)
        ax2.set_xlim(0,1)
        ax2.set_title("Negative")
        ax2.set_xlabel("p-hat")
        plt.savefig(path + '.phat_dist.png')
        plt.close()

    if plot_curves:
        all_phat = np.concatenate([pos_phat, neg_phat])
        all_y = [1]*len(pos_phat) + [0]*len(neg_phat)
        precision, recall, _ = precision_recall_curve(all_y, all_phat)
        aupr = average_precision_score(all_y, all_phat)

        pr_df = pd.DataFrame({"recall": recall, "precision": precision})
        pr_df.to_csv(path + "_pr_curve.csv", index=False)
        print("AUPR:", aupr)

        plt.step(recall, precision, color='b', alpha=0.2, where='post')
        plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('Precision-Recall (AUPR: {:.3})'.format(aupr))
        plt.savefig(path + '.aupr.png')
        plt.close()

        fpr, tpr, _ = roc_curve(all_y, all_phat)
        auroc = roc_auc_score(all_y, all_phat)
        print("AUROC:", auroc)

        plt.step(fpr, tpr, color='b', alpha=0.2, where='post')
        plt.fill_between(fpr, tpr, step='post', alpha=0.2, color='b')
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('Receiver Operating Characteristic (AUROC: {:.3})'.format(auroc))
        plt.savefig(path + '.auroc.png')
        plt.close()

def compute_metrics_for_threshold(y_true, y_score, threshold):
    """Compute classification metrics at a given threshold."""
    y_pred = (y_score >= threshold).astype(int)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    rec = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    return acc, prec, rec, f1

def generate_threshold_metrics(y_true, y_score):
    """Generate metrics for thresholds 0.0 to 1.0 step 0.01."""
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
    return pd.DataFrame(metrics_list)

def main():
    parser = argparse.ArgumentParser('Evaluate trained Topsy-Turvy model with TSV test set')
    parser.add_argument('--model', required=True, help='Trained model .sav file')
    parser.add_argument('--test-tsv', required=True, help='Test pairs TSV (prot1\tprot2\tlabel)')
    parser.add_argument('--embedding', required=True, help='HDF5 embedding file')
    parser.add_argument('--max-seq-len', type=int, default=2000, help='Sequence truncation length (must match training)')
    parser.add_argument('--manual-threshold', type=float, default=0.5, help='Threshold for manual classification')
    parser.add_argument('--outprefix', default='evaluation', help='Prefix for output files')
    parser.add_argument('--device', type=int, default=0, help='GPU device (-1 for CPU)')
    parser.add_argument('--plot-distributions', action='store_true', help='Plot histograms of predictions')
    parser.add_argument('--plot-curves', action='store_true', help='Plot AUPR and AUROC curves')
    args = parser.parse_args()

    device = args.device
    use_cuda = (device >= 0) and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(device)
        print(f'Using CUDA device {device} - {torch.cuda.get_device_name(device)}')
    else:
        print('Using CPU')

    # Load model
    print('Loading model...')
    model = torch.load(args.model, map_location='cpu')
    # Disable GLIDER (not needed for evaluation)
    model.glider_mat = None
    model.glider_map = None
    if use_cuda:
        model = model.cuda()
    else:
        model = model.cpu()
        model.use_cuda = False
    model.eval()

    # Load test data
    print('Loading test data...')
    test_df = pd.read_csv(args.test_tsv, sep='\t', header=None)
    test_df.columns = ['prot1', 'prot2', 'label']
    all_proteins = set(test_df['prot1']).union(set(test_df['prot2']))

    # Load embeddings with truncation
    print('Loading embeddings...')
    h5fi = h5py.File(args.embedding, 'r')
    seqEmbDict = {}
    for prot in tqdm(all_proteins, desc='Loading embeddings'):
        embed = h5fi[prot][:, :]
        if embed.shape[0] > args.max_seq_len:
            embed = embed[:args.max_seq_len, :].copy()
        seqEmbDict[prot] = torch.from_numpy(embed).float()

    # Predict
    predictions = []
    labels = []
    outfile = open(args.outprefix + '.predictions.tsv', 'w')
    print("protein1\tprotein2\tlabel\tprobability", file=outfile)

    with torch.no_grad():
        for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc='Predicting pairs'):
            p1 = seqEmbDict[row['prot1']]
            p2 = seqEmbDict[row['prot2']]
            if use_cuda:
                p1 = p1.cuda()
                p2 = p2.cuda()
            _, pred = model.map_predict(p1, p2)
            prob = pred.item()
            predictions.append(prob)
            labels.append(row['label'])
            print(f"{row['prot1']}\t{row['prot2']}\t{row['label']}\t{prob:.6f}", file=outfile)
    outfile.close()
    h5fi.close()

    predictions = np.array(predictions)
    labels = np.array(labels)

    # Separate positive and negative for plots
    pos_idx = labels == 1
    neg_idx = labels == 0
    pos_phat = predictions[pos_idx]
    neg_phat = predictions[neg_idx]

    # Generate threshold metrics (for PR curve)
    metrics_df = generate_threshold_metrics(labels, predictions)
    metrics_df.to_csv(args.outprefix + '_threshold_metrics.csv', index=False)
    print(f"Threshold metrics saved to {args.outprefix}_threshold_metrics.csv")

    # Manual threshold classification
    acc, prec, rec, f1 = compute_metrics_for_threshold(labels, predictions, args.manual_threshold)
    print(f"\n========== Manual Threshold (={args.manual_threshold}) ==========")
    print(f"Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")

    # Best F1 threshold
    best_idx = metrics_df['f1'].idxmax()
    best_row = metrics_df.loc[best_idx]
    print(f"\n========== Best F1 Threshold ==========")
    print(f"Threshold: {best_row['threshold']:.2f}, Precision: {best_row['precision']:.4f}, Recall: {best_row['recall']:.4f}, F1: {best_row['f1']:.4f}, Accuracy: {best_row['accuracy']:.4f}")

    # Overall AUPR/AUROC
    aupr = average_precision_score(labels, predictions)
    auroc = roc_auc_score(labels, predictions)
    print(f"\nOverall AUPR: {aupr:.4f}, AUROC: {auroc:.4f}")

    # Plot distributions and curves if requested
    if args.plot_distributions or args.plot_curves:
        plot_eval_predictions(pos_phat, neg_phat,
                              plot_prediction_distributions=args.plot_distributions,
                              plot_curves=args.plot_curves,
                              path=args.outprefix)

    # Save metrics summary
    with open(args.outprefix + '_metrics.txt', 'w') as f:
        f.write(f"Manual threshold ({args.manual_threshold}): Acc={acc:.4f}, Prec={prec:.4f}, Rec={rec:.4f}, F1={f1:.4f}\n")
        f.write(f"Best F1 threshold ({best_row['threshold']}): Prec={best_row['precision']:.4f}, Rec={best_row['recall']:.4f}, F1={best_row['f1']:.4f}, Acc={best_row['accuracy']:.4f}\n")
        f.write(f"AUPR={aupr:.4f}, AUROC={auroc:.4f}\n")

    print("Done.")

if __name__ == "__main__":
    main()
