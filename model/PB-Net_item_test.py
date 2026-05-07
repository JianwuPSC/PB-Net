#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for SimpleMLP model (item-level, with binder).
Compatible with models trained using PB-Net_item_train.py.
Supports the same feature modes, model architecture, and can load a saved StandardScaler.
"""

import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pickle
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score

# ---------------------------- Model Definition (must match training) ----------------------------
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


# ---------------------------- Feature Loading (reused from training) ----------------------------
def load_features(source_file, binder_file, target_file, saprot_source_file, saprot_binder_file, saprot_target_file):
    """Load all feature files and return DataFrames."""
    esm_source = pd.read_csv(source_file, header=None)
    esm_binder = pd.read_csv(binder_file, header=None)
    esm_target = pd.read_csv(target_file, header=None)

    saprot_source = pd.read_csv(saprot_source_file, header=None)
    saprot_binder = pd.read_csv(saprot_binder_file, header=None)
    saprot_target = pd.read_csv(saprot_target_file, header=None)

    # Extract name (first column) and label (column 1281) from saprot_source
    name_df = saprot_source[[0]].copy()
    name_df.columns = ['name']
    class_df = saprot_source[[1281]].copy()
    class_df.columns = ['class']

    # Drop name and label columns
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
    """Build feature matrix according to the specified mode (must match training)."""
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


# ---------------------------- Pair Aggregation (optional) ----------------------------
def aggregate_pairs(item_df, manual_threshold=0.5):
    """Aggregate item-level predictions to protein-pair level (positive class=1)."""
    pair_groups = item_df.groupby(['bait', 'target'])
    pair_records = []
    for (bait, target), group in pair_groups:
        true_label = group['true_label'].iloc[0]
        any_positive = (group['prob_positive'] >= manual_threshold).any()
        pair_pred_label = 1 if any_positive else 0
        max_prob = group['prob_positive'].max()
        pair_records.append({
            'bait': bait, 'target': target,
            'true_label': true_label, 'pred_label_manual': pair_pred_label,
            'max_prob': max_prob, 'n_items': len(group)
        })
    return pd.DataFrame(pair_records)


def extract_bait_target(name):
    parts = name.split('_')
    bait = parts[0]
    target = parts[-2]
    return bait, target


# ---------------------------- Main ----------------------------
def main():
    parser = argparse.ArgumentParser(description='Test SimpleMLP model for PPI prediction (item-level)')
    # Test data files (same format as training)
    parser.add_argument('--test_source_esm', type=str, required=True)
    parser.add_argument('--test_binder_esm', type=str, required=True)
    parser.add_argument('--test_target_esm', type=str, required=True)
    parser.add_argument('--test_source_saprot', type=str, required=True)
    parser.add_argument('--test_binder_saprot', type=str, required=True)
    parser.add_argument('--test_target_saprot', type=str, required=True)

    # Model and mode
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model .pt file')
    parser.add_argument('--mode', type=str, required=True,
                        choices=[
                            'esm2_source_binder_target',
                            'saprot_source_binder_target',
                            'esm2_match_source_binder_target',
                            'saprot_match_source_binder_target',
                            'cat_esm2_match_source_binder_target_saprot_match_source_binder_target',
                            'cat_esm2_source_binder_target_saprot_match_source_binder_target',
                            'cat_saprot_source_binder_target_esm2_match_source_binder_target',
                            'cat_esm_source_target',
                            'cat_saprot_source_target'
                        ],
                        help='Feature concatenation mode (must match training)')
    parser.add_argument('--hidden_size', type=int, default=128, help='Hidden size of the model')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for prediction')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use (cuda:0 or cpu)')
    parser.add_argument('--seed', type=int, default=12, help='Random seed (for reproducibility)')
    parser.add_argument('--scaler_path', type=str, default=None, help='Path to saved StandardScaler (pickle file)')

    # Output
    parser.add_argument('--output_csv', type=str, default='test_predictions.csv', help='Output CSV file path')
    parser.add_argument('--with_labels', action='store_true', help='If set, assume test data has labels and compute metrics')
    parser.add_argument('--manual_threshold', type=float, default=0.5, help='Threshold for pair-level aggregation (ignored if not computing pairs)')
    parser.add_argument('--compute_pair_metrics', action='store_true', help='Also compute and save protein-pair level metrics')

    args = parser.parse_args()

    # Set random seed (optional, for reproducibility)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load test data
    print("Loading test data...")
    test_feat = load_features(args.test_source_esm, args.test_binder_esm, args.test_target_esm,
                              args.test_source_saprot, args.test_binder_saprot, args.test_target_saprot)

    X_test = build_features(test_feat, args.mode)
    names = test_feat['name'].values.ravel()
    print(f"Test data shape: {X_test.shape}")

    # Apply StandardScaler if provided
    if args.scaler_path is not None:
        with open(args.scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        X_test = scaler.transform(X_test)
        print("Scaler mean (first 5):", scaler.mean_[:5])
        print("Scaler scale (first 5):", scaler.scale_[:5])
        print("First test sample after scaler:", X_test[0, :5])
        print(f"Applied scaler from {args.scaler_path}")

    # Determine input size
    input_size = X_test.shape[1]

    # Load model
    model = SimpleMLP(input_size, hidden_size=args.hidden_size, dropout_rate=args.dropout).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    print(f"Model loaded from {args.model_path}")

    # DataLoader
    X_tensor = torch.tensor(X_test, dtype=torch.float32)
    test_dataset = TensorDataset(X_tensor)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Predict
    all_probs = []
    all_preds = []
    with torch.no_grad():
        for batch in test_loader:
            batch_x = batch[0].to(device)
            outputs = model(batch_x)
            prob = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()  # positive class probability (label=1)
            pred = (prob >= args.manual_threshold).astype(int)
            all_probs.extend(prob)
            all_preds.extend(pred)

    # Prepare output DataFrame (item-level)
    output_df = pd.DataFrame({
        'name': names,
        'pred_label': all_preds,
        'prob_positive': all_probs
    })

    # If labels are available, compute metrics
    if args.with_labels:
        y_true = test_feat['class'].values.ravel().astype(int)
        output_df['true_label'] = y_true
        metrics = compute_metrics(y_true, all_preds, all_probs, pos_label=1)
        print("\n========== Test Set Metrics (Item-level) ==========")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")
        # Save metrics to CSV
        metrics_df = pd.DataFrame([metrics])
        metrics_df.to_csv(args.output_csv.replace('.csv', '_item_metrics.csv'), index=False)

    # Save item-level predictions
    output_df.to_csv(args.output_csv, index=False)
    print(f"Item-level predictions saved to {args.output_csv}")

    # Optional: pair-level aggregation
    if args.compute_pair_metrics:
        print("\nComputing protein-pair level metrics...")
        # Parse bait and target from item names
        bait_list = [extract_bait_target(n)[0] for n in names]
        target_list = [extract_bait_target(n)[1] for n in names]
        item_df = pd.DataFrame({
            'name': names,
            'bait': bait_list,
            'target': target_list,
            'true_label': y_true if args.with_labels else None,
            'prob_positive': all_probs
        })
        if args.with_labels:
            item_df['true_label'] = y_true
        # Aggregate
        pair_df = aggregate_pairs(item_df, manual_threshold=args.manual_threshold)
        pair_df.to_csv(args.output_csv.replace('.csv', '_pair_results.csv'), index=False)
        print(f"Pair-level results saved to {args.output_csv.replace('.csv', '_pair_results.csv')}")

        # Compute pair-level metrics if labels exist
        if args.with_labels:
            y_true_pair = pair_df['true_label'].values
            y_score_pair = pair_df['max_prob'].values
            pred_pair = pair_df['pred_label_manual'].values
            pair_metrics = compute_metrics(y_true_pair, pred_pair, y_score_pair, pos_label=1)
            print("\n========== Test Set Metrics (Pair-level) ==========")
            for k, v in pair_metrics.items():
                print(f"{k}: {v:.4f}")
            # Save pair metrics
            pd.DataFrame([pair_metrics]).to_csv(args.output_csv.replace('.csv', '_pair_metrics.csv'), index=False)

    print("Done.")


if __name__ == '__main__':
    main()
