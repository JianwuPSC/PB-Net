#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rule‑based PPI classifier (protein‑pair level).
Applies user‑defined thresholds on selected features to flag individual items,
then aggregates to protein pairs: a pair is positive if any of its items is positive.
Metrics are computed at the pair level.
"""

import argparse
import numpy as np
import pandas as pd
import re
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# ======================== All available features ========================
FEATURE_COLS = {
    'nearest_dis':  1,
    'counts_8A':    2,
    'binder_PAE':   3,
    'binder_RMSD':  4,
    'inter_PAE':    5,
    'query_PAE':    6,
    'binder_pLDDT': 7,
    'query_pLDDT':  8,
    'total_pLDDT':  9,
    'query_RMSD':  10,
    '3d_score':    11,
    'ss_score':    12,
    'blast_score': 13,
    'f1':          14,
    'f2':          15,
    'f3':          16,
    'f4':          17,
    'f5':          18,
}

def load_test_file(file_path):
    """Load the tab‑separated file, return names, features DataFrame, true labels."""
    data = np.loadtxt(file_path, delimiter='\t', dtype=str)
    names = data[:, 0]
    true_label = (data[:, -1].astype(str) == '1').astype(int)   # positive = 1

    # Build feature DataFrame
    feat_dict = {}
    for fname, col_idx in FEATURE_COLS.items():
        feat_dict[fname] = data[:, col_idx].astype(float)
    df_feat = pd.DataFrame(feat_dict)
    # Clip f1–f5 as in original code
    for f in ['f1','f2','f3','f4','f5']:
        df_feat[f] = df_feat[f].clip(upper=10)
    return names, df_feat, true_label

def apply_rules(df, rules, logic='AND'):
    """
    Apply a list of rules to flag samples as positive.
    Each rule is a tuple: (feature_name, operator_str, threshold)
    logic: 'AND' (all rules must be satisfied) or 'OR' (at least one rule satisfied).
    Returns a boolean array (True = positive).
    """
    if not rules:
        return np.ones(len(df), dtype=bool)

    mask_list = []
    for feat, op, thr in rules:
        vals = df[feat].values
        if op == '>':
            m = vals > thr
        elif op == '>=':
            m = vals >= thr
        elif op == '<':
            m = vals < thr
        elif op == '<=':
            m = vals <= thr
        elif op == '==':
            m = vals == thr
        else:
            raise ValueError(f"Unsupported operator: {op}")
        mask_list.append(m)

    if logic.upper() == 'AND':
        return np.logical_and.reduce(mask_list)
    else:  # 'OR'
        return np.logical_or.reduce(mask_list)

def extract_bait_target(name):
    """Extract bait and target protein identifiers from the item name."""
    parts = name.split('_')
    bait = parts[0]
    target = parts[-2]
    return bait, target

def evaluate_pair(y_true, y_pred):
    """Return dict of metrics for pair-level (pos_label=1)."""
    prec = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    rec  = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1   = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    acc  = accuracy_score(y_true, y_pred)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    return {'precision': prec, 'recall': rec, 'f1': f1, 'accuracy': acc,
            'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn}

def main():
    parser = argparse.ArgumentParser(description='Rule‑based PPI classifier (pair level)')
    parser.add_argument('--test_file', type=str, required=True,
                        help='Path to test feature file (tab‑separated)')
    parser.add_argument('--rules', type=str, required=True,
                        help='Comma‑separated rules, e.g. "nearest_dis>5,f1<2"')
    parser.add_argument('--logic', type=str, default='AND', choices=['AND','OR'],
                        help='How to combine multiple rules (default: AND)')
    parser.add_argument('--output_prefix', type=str, default='rule_based_pair',
                        help='Prefix for output files')
    args = parser.parse_args()

    # --- Load data and apply rules on items ---
    names, df_feat, y_true = load_test_file(args.test_file)
    print(f"Loaded {len(df_feat)} items.")

    raw_rules = [r.strip() for r in args.rules.split(',') if r.strip()]
    parsed_rules = []
    for rule_str in raw_rules:
        m = re.match(r'(\w+)\s*(>=|<=|!=|==|>|<)\s*(.+)', rule_str)
        if not m:
            raise ValueError(f"Cannot parse rule: {rule_str}. Expected format like 'nearest_dis>5'")
        feat, op, val = m.groups()
        if feat not in FEATURE_COLS:
            raise ValueError(f"Unknown feature '{feat}'. Available: {list(FEATURE_COLS.keys())}")
        thr = float(val)
        parsed_rules.append((feat, op, thr))
    print(f"Rules: {parsed_rules}, logic = {args.logic}")

    pred_mask = apply_rules(df_feat, parsed_rules, args.logic)
    y_pred_item = pred_mask.astype(int)

    # --- Build item-level DataFrame for aggregation ---
    item_df = pd.DataFrame({
        'name': names,
        'true_label': y_true,
        'pred_label': y_pred_item
    })

    # Extract bait and target
    bait_list, target_list = [], []
    for n in names:
        b, t = extract_bait_target(n)
        bait_list.append(b)
        target_list.append(t)
    item_df['bait'] = bait_list
    item_df['target'] = target_list

    # --- Aggregate to protein pairs ---
    pair_groups = item_df.groupby(['bait', 'target'])
    pair_records = []
    for (bait, target), group in pair_groups:
        # True label should be identical for all items of the same pair
        true_label = group['true_label'].iloc[0]
        # Pair is positive if ANY item is predicted positive
        pair_pred = 1 if (group['pred_label'] == 1).any() else 0
        pair_records.append({
            'bait': bait,
            'target': target,
            'true_label': true_label,
            'pred_label': pair_pred
        })
    pair_df = pd.DataFrame(pair_records)
    print(f"Generated {len(pair_df)} protein pairs.")

    # --- Evaluate at pair level ---
    y_true_pair = pair_df['true_label'].values.astype(int)
    y_pred_pair = pair_df['pred_label'].values.astype(int)
    metrics = evaluate_pair(y_true_pair, y_pred_pair)

    print("\n========== Pair‑level Rule‑based Results ==========")
    print(f"Logic: {args.logic}")
    for rule in parsed_rules:
        print(f"  {rule[0]} {rule[1]} {rule[2]}")
    print(f"\nTP = {metrics['TP']}, FP = {metrics['FP']}, FN = {metrics['FN']}, TN = {metrics['TN']}")
    print(f"Precision = {metrics['precision']:.4f}")
    print(f"Recall    = {metrics['recall']:.4f}")
    print(f"F1        = {metrics['f1']:.4f}")
    print(f"Accuracy  = {metrics['accuracy']:.4f}")

    # --- Save pair predictions and metrics ---
    pair_out = f"{args.output_prefix}_pair_predictions.csv"
    pair_df.to_csv(pair_out, index=False)
    print(f"\nPair‑level predictions saved to {pair_out}")

    metrics_out = f"{args.output_prefix}_pair_metrics.csv"
    pd.DataFrame([metrics]).to_csv(metrics_out, index=False)
    print(f"Pair‑level metrics saved to {metrics_out}")

if __name__ == '__main__':
    main()
