#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Independent test script for SimpleMLP model (item-level, with binder).
Completely consistent with training evaluation, including deterministic settings.
Supports item-level and pair-level metrics, multi-threshold evaluation.
Additionally saves standardized test features to CSV.
"""

import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import random
import pickle
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score,precision_recall_curve
from sklearn.preprocessing import StandardScaler

# ---------------------------- Set Random Seed ----------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ---------------------------- Model Definition ----------------------------
class SimpleMLP(nn.Module):
    def __init__(self, input_size, hidden_size=128, dropout_rate=0.5):
        super(SimpleMLP, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_size, hidden_size * 4),
            nn.BatchNorm1d(hidden_size * 4),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, 2)
        )
    def forward(self, x):
        return self.classifier(x)

# ---------------------------- Feature Loading ----------------------------
def load_features(source_file, binder_file, target_file, saprot_source_file, saprot_binder_file, saprot_target_file):
    esm_source = pd.read_csv(source_file, header=None)
    esm_binder = pd.read_csv(binder_file, header=None)
    esm_target = pd.read_csv(target_file, header=None)

    saprot_source = pd.read_csv(saprot_source_file, header=None)
    saprot_binder = pd.read_csv(saprot_binder_file, header=None)
    saprot_target = pd.read_csv(saprot_target_file, header=None)

    name_df = saprot_source[[0]].copy()
    name_df.columns = ['name']
    class_series = saprot_source[1281].copy()
    class_df = pd.DataFrame(class_series)
    class_df.columns = ['class']

    esm_source = esm_source.iloc[:, 1:-1]
    esm_binder = esm_binder.iloc[:, 1:-1]
    esm_target = esm_target.iloc[:, 1:-1]

    saprot_source = saprot_source.iloc[:, 1:-1]
    saprot_binder = saprot_binder.iloc[:, 1:-1]
    saprot_target = saprot_target.iloc[:, 1:-1]

    return {
        'esm_s': esm_source, 'esm_b': esm_binder, 'esm_t': esm_target,
        'sap_s': saprot_source, 'sap_b': saprot_binder, 'sap_t': saprot_target,
        'name': name_df, 'class': class_df
    }

def build_features(feature_dict, mode):
    esm_s = feature_dict['esm_s']
    esm_b = feature_dict['esm_b']
    esm_t = feature_dict['esm_t']
    sap_s = feature_dict['sap_s']
    sap_b = feature_dict['sap_b']
    sap_t = feature_dict['sap_t']

    if mode == 'esm2_source_binder_target':
        X = pd.concat([esm_s, esm_b, esm_t], axis=1)
    elif mode == 'saprot_source_binder_target':
        X = pd.concat([sap_s, sap_b, sap_t], axis=1)
    elif mode == 'esm2_match_source_binder_target':
        X = esm_s - esm_b - esm_t
    elif mode == 'saprot_match_source_binder_target':
        X = sap_s - sap_b - sap_t
    elif mode == 'cat_esm2_match_source_binder_target_saprot_match_source_binder_target':
        X1 = esm_s - esm_b - esm_t
        X2 = sap_s - sap_b - sap_t
        X = pd.concat([X1, X2], axis=1)
    elif mode == 'cat_esm2_source_binder_target_saprot_match_source_binder_target':
        X1 = pd.concat([esm_s, esm_b, esm_t], axis=1)
        X2 = sap_s - sap_b - sap_t
        X = pd.concat([X1, X2], axis=1)
    elif mode == 'cat_saprot_source_binder_target_esm2_match_source_binder_target':
        X1 = pd.concat([sap_s, sap_b, sap_t], axis=1)
        X2 = esm_s - esm_b - esm_t
        X = pd.concat([X1, X2], axis=1)
    elif mode == 'cat_esm_source_target':
        X = pd.concat([esm_s, esm_t], axis=1)
    elif mode == 'cat_saprot_source_target':
        X = pd.concat([sap_s, sap_t], axis=1)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    X = X.fillna(0)
    return X.values.astype(np.float32)

# ---------------------------- Metrics (pos_label=1) ----------------------------
def compute_metrics(y_true, y_pred, y_score, pos_label=1):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
    rec = recall_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
    roc_auc = roc_auc_score(y_true, y_score)
    auprc = average_precision_score(y_true, y_score, pos_label=pos_label)

    return {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'roc_auc': roc_auc,
        'auprc': auprc
    }

# ---------------------------- Protein Pair Aggregation ----------------------------
def compute_pair_metrics_from_items(item_df, manual_threshold=0.5):
    pair_groups = item_df.groupby(['bait', 'target'])
    pair_records = []
    for (bait, target), group in pair_groups:
        true_label = group['true_label'].iloc[0] if 'true_label' in group else None
        any_positive = (group['prob_positive'] >= manual_threshold).any()
        pair_pred_label = 1 if any_positive else 0
        max_prob = group['prob_positive'].max()
        record = {
            'bait': bait, 'target': target,
            'pred_label': pair_pred_label,
            'max_prob': max_prob, 'n_items': len(group)
        }
        if true_label is not None:
            record['true_label'] = true_label
        pair_records.append(record)
    return pd.DataFrame(pair_records)

def extract_bait_target(name):
    parts = name.split('_')
    bait = parts[0]
    target = parts[-2]
    return bait, target

# ---------------------------- Main ----------------------------
def main():
    parser = argparse.ArgumentParser(description='Test SimpleMLP model (fully consistent with training)')
    # Test data
    parser.add_argument('--test_source_esm', type=str, required=True)
    parser.add_argument('--test_binder_esm', type=str, required=True)
    parser.add_argument('--test_target_esm', type=str, required=True)
    parser.add_argument('--test_source_saprot', type=str, required=True)
    parser.add_argument('--test_binder_saprot', type=str, required=True)
    parser.add_argument('--test_target_saprot', type=str, required=True)

    # Model, scaler, mode
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--scaler_path', type=str, required=True)
    parser.add_argument('--mode', type=str, required=True,
                        choices=['esm2_source_binder_target',
                                 'saprot_source_binder_target',
                                 'esm2_match_source_binder_target',
                                 'saprot_match_source_binder_target',
                                 'cat_esm2_match_source_binder_target_saprot_match_source_binder_target',
                                 'cat_esm2_source_binder_target_saprot_match_source_binder_target',
                                 'cat_saprot_source_binder_target_esm2_match_source_binder_target',
                                 'cat_esm_source_target',
                                 'cat_saprot_source_target'])
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--seed', type=int, default=12)

    # Labels and output control
    parser.add_argument('--with_labels', action='store_true',
                        help='Assume test data contains labels and compute metrics')
    parser.add_argument('--compute_pair_metrics', action='store_true',
                        help='Aggregate to protein-pair level and compute pair metrics')
    parser.add_argument('--manual_threshold', type=float, default=0.5,
                        help='Default threshold for pair-level positive prediction')
    parser.add_argument('--manual_thresholds', type=str, default=None,
                        help='Comma-separated thresholds for pair aggregation evaluation (e.g. 0.3,0.5,0.7)')
    parser.add_argument('--output_prefix', type=str, default='test_predictions',
                        help='Prefix for all output CSV files')

    args = parser.parse_args()

    # Set deterministic seed
    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load test features
    print("Loading test data...")
    test_feat = load_features(args.test_source_esm, args.test_binder_esm, args.test_target_esm,
                              args.test_source_saprot, args.test_binder_saprot, args.test_target_saprot)
    X_test = build_features(test_feat, args.mode)
    names = test_feat['name'].values.ravel()
    print(f"Test data shape: {X_test.shape}")

    # Load scaler and apply
    with open(args.scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    X_test = scaler.transform(X_test)
    print("First test sample after scaler:", X_test[0, :5])

    # ========== 新增：输出标准化后的测试特征 CSV ==========
    test_feat_df = pd.DataFrame(X_test, columns=[f'feat_{i}' for i in range(X_test.shape[1])])
    test_feat_df.insert(0, 'name', names)
    # 添加标签列（无论是否 with_labels，标签都存在于原始数据中）
    test_feat_df['label'] = test_feat['class'].values.ravel().astype(np.int64)
    std_csv_path = f"{args.output_prefix}_test_standardized.csv"
    test_feat_df.to_csv(std_csv_path, index=False)
    print(f"Standardized test features saved to {std_csv_path}")
    # =====================================================

    # Load model
    input_size = X_test.shape[1]
    model = SimpleMLP(input_size, hidden_size=args.hidden_size, dropout_rate=args.dropout).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    print(f"Model loaded from {args.model_path}")

    # Prepare DataLoader
    X_tensor = torch.tensor(X_test, dtype=torch.float32)
    if args.with_labels:
        y_test = test_feat['class'].values.ravel().astype(np.int64)
        y_tensor = torch.tensor(y_test, dtype=torch.long)
        test_dataset = TensorDataset(X_tensor, y_tensor)
    else:
        test_dataset = TensorDataset(X_tensor)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Predict
    all_probs = []
    all_preds = []
    all_true = [] if args.with_labels else None
    with torch.no_grad():
        for batch in test_loader:
            if args.with_labels:
                batch_x, batch_y = batch
                batch_x = batch_x.to(device)
                all_true.extend(batch_y.numpy())
            else:
                batch_x = batch[0].to(device)
            outputs = model(batch_x)
            prob_class1 = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            pred = (prob_class1 >= 0.5).astype(int)
            all_probs.extend(prob_class1)
            all_preds.extend(pred)

    # Item-level results
    item_df = pd.DataFrame({
        'name': names,
        'pred_label': all_preds,
        'prob_positive': all_probs
    })
    if args.with_labels:
        y_true_item = np.array(all_true)
        item_df['true_label'] = y_true_item
        # Compute metrics at threshold 0.5
        item_metrics = compute_metrics(y_true_item, all_preds, all_probs, pos_label=1)
        precision, recall, _ = precision_recall_curve(y_true_item, all_probs, pos_label=1)
        pr_df = pd.DataFrame({"recall": recall, "precision": precision})
        pr_df.to_csv("item_pr_curve.csv", index=False)

        print("\n========== Test Set Metrics (Item-level, threshold=0.5) ==========")
        for k, v in item_metrics.items():
            print(f"{k}: {v:.4f}")

        # Multi-threshold scanning for item-level
        thresholds = np.arange(0.0, 1.01, 0.01)
        item_metrics_list = []
        for thr in thresholds:
            y_pred = (all_probs >= thr).astype(int)
            prec = precision_score(y_true_item, y_pred, pos_label=1, zero_division=0)
            rec = recall_score(y_true_item, y_pred, pos_label=1, zero_division=0)
            f1 = f1_score(y_true_item, y_pred, pos_label=1, zero_division=0)
            acc = accuracy_score(y_true_item, y_pred)
            item_metrics_list.append({'threshold': thr, 'precision': prec, 'recall': rec, 'f1': f1, 'accuracy': acc})
        item_metrics_df = pd.DataFrame(item_metrics_list)
        best_idx = item_metrics_df['f1'].idxmax()
        best_item_row = item_metrics_df.loc[best_idx]
        best_item_info = {
            'threshold': best_item_row['threshold'],
            'precision': best_item_row['precision'],
            'recall': best_item_row['recall'],
            'f1': best_item_row['f1'],
            'accuracy': best_item_row['accuracy'],
            'roc_auc': item_metrics['roc_auc'],
            'auprc': item_metrics['auprc']
        }
        print(f"\n=== Item-level Best F1 @ threshold {best_item_info['threshold']:.2f} ===")
        print(f"Precision={best_item_info['precision']:.4f}, Recall={best_item_info['recall']:.4f}, F1={best_item_info['f1']:.4f}, Accuracy={best_item_info['accuracy']:.4f}")

        # Save item-level files
        item_df.to_csv(f"{args.output_prefix}_item_predictions.csv", index=False)
        pd.DataFrame([item_metrics]).to_csv(f"{args.output_prefix}_item_overall_metrics.csv", index=False)
        pd.DataFrame([best_item_info]).to_csv(f"{args.output_prefix}_item_best_f1.csv", index=False)

        # Pair-level if requested
        if args.compute_pair_metrics:
            # Add bait/target columns
            bait_list = [extract_bait_target(n)[0] for n in names]
            target_list = [extract_bait_target(n)[1] for n in names]
            item_df['bait'] = bait_list
            item_df['target'] = target_list

            # --- Default single threshold evaluation ---
            pair_df = compute_pair_metrics_from_items(item_df, manual_threshold=args.manual_threshold)
            pair_df.to_csv(f"{args.output_prefix}_pair_results.csv", index=False)

            if 'true_label' in pair_df.columns:
                y_true_pair = pair_df['true_label'].values.ravel()
                y_score_pair = pair_df['max_prob'].values
                pred_pair = pair_df['pred_label'].values
                pair_metrics = compute_metrics(y_true_pair, pred_pair, y_score_pair, pos_label=1)
                precision, recall, _ = precision_recall_curve(y_true_pair, y_score_pair, pos_label=1)
                pr_df = pd.DataFrame({"recall": recall, "precision": precision})
                pr_df.to_csv("pair_pr_curve.csv", index=False)

                roc_auc = pair_metrics['roc_auc']
                auprc = pair_metrics['auprc']
                print(f"\n=== Protein Pair Level (manual threshold = {args.manual_threshold}) ===")
                print(f"Precision={pair_metrics['precision']:.4f}, Recall={pair_metrics['recall']:.4f}, F1={pair_metrics['f1']:.4f}, Accuracy={pair_metrics['accuracy']:.4f}, ROC AUC={roc_auc:.4f}, AUPRC={auprc:.4f}")

                # Multi-threshold scanning (fixed aggregation threshold)
                thresholds = np.arange(0.0, 1.01, 0.01)
                metrics_list = []
                for thr in thresholds:
                    y_pred = np.where(y_score_pair >= thr, 1, 0)
                    tp = np.sum((y_true_pair == 1) & (y_pred == 1))
                    fp = np.sum((y_true_pair != 1) & (y_pred == 1))
                    fn = np.sum((y_true_pair == 1) & (y_pred != 1))
                    tn = np.sum((y_true_pair != 1) & (y_pred != 1))
                    precision = tp / (tp+fp) if (tp+fp)>0 else np.nan
                    recall = tp / (tp+fn) if (tp+fn)>0 else np.nan
                    f1_thr = 2 * precision * recall / (precision+recall) if (precision+recall)>0 else np.nan
                    accuracy = (tp+tn) / (tp+tn+fp+fn) if (tp+tn+fp+fn)>0 else np.nan
                    metrics_list.append({
                        'threshold': thr, 'precision': precision, 'recall': recall,
                        'f1': f1_thr, 'accuracy': accuracy,
                        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn
                    })
                pair_metrics_df = pd.DataFrame(metrics_list)
                best_idx = pair_metrics_df['f1'].idxmax()
                best_pair_row = pair_metrics_df.loc[best_idx]
                print(f"\n=== Pair-level Best F1 (score threshold) @ {best_pair_row['threshold']:.2f} ===")
                print(f"Precision={best_pair_row['precision']:.4f}, Recall={best_pair_row['recall']:.4f}, F1={best_pair_row['f1']:.4f}, Accuracy={best_pair_row['accuracy']:.4f}")

                pair_metrics_df.to_csv(f"{args.output_prefix}_pair_threshold_metrics.csv", index=False)
                pd.DataFrame([{
                    'threshold': best_pair_row['threshold'],
                    'precision': best_pair_row['precision'],
                    'recall': best_pair_row['recall'],
                    'f1': best_pair_row['f1'],
                    'accuracy': best_pair_row['accuracy'],
                    'roc_auc': roc_auc,
                    'auprc': auprc
                }]).to_csv(f"{args.output_prefix}_pair_best_f1.csv", index=False)

            # --- NEW: Multi-threshold aggregation evaluation ---
            if args.manual_thresholds:
                thr_list = [float(x.strip()) for x in args.manual_thresholds.split(',')]
                multi_metrics = []
                print("\n--- Multi-threshold pair aggregation evaluation ---")
                for thr in thr_list:
                    pair_tmp = compute_pair_metrics_from_items(item_df, manual_threshold=thr)
                    if 'true_label' in pair_tmp.columns:
                        y_t = pair_tmp['true_label'].values.ravel()
                        y_s = pair_tmp['max_prob'].values
                        y_p = pair_tmp['pred_label'].values
                        m = compute_metrics(y_t, y_p, y_s, pos_label=1)
                        m['aggregation_threshold'] = thr
                        multi_metrics.append(m)
                        print(f"Thr={thr:.2f}: P={m['precision']:.4f}, R={m['recall']:.4f}, F1={m['f1']:.4f}")
                if multi_metrics:
                    multi_df = pd.DataFrame(multi_metrics)
                    multi_out = f"{args.output_prefix}_pair_multi_threshold_metrics.csv"
                    multi_df.to_csv(multi_out, index=False)
                    print(f"Multi-threshold pair metrics saved to {multi_out}")

        print(f"\nAll results saved with prefix: {args.output_prefix}")
    else:
        # No labels, just save predictions
        item_df.to_csv(f"{args.output_prefix}_item_predictions.csv", index=False)
        print(f"Item-level predictions saved to {args.output_prefix}_item_predictions.csv")

    print("Done.")

if __name__ == '__main__':
    main()
