#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rule‑based classifier for PPI prediction.
Applies user‑defined thresholds on selected features (from the test feature file)
to flag samples as positive. Supports AND / OR combination of multiple conditions.
Metrics (precision, recall, F1, accuracy) are computed against the true label
(last column, 1 = positive, 0 = negative).
"""

import argparse
import numpy as np
import pandas as pd
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
      e.g. ('nearest_dis', '>', 5.0)  or  ('counts_8A', '<=', 2)
    logic: 'AND' (all rules must be satisfied) or 'OR' (at least one rule satisfied).

    Returns a boolean array (True = positive).
    """
    if not rules:
        return np.ones(len(df), dtype=bool)  # no rule -> all positive

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

def evaluate(y_true, y_pred):
    """Return dict of metrics (pos_label=1)."""
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
    parser = argparse.ArgumentParser(description='Rule‑based PPI classifier using feature thresholds.')
    parser.add_argument('--test_file', type=str, required=True,
                        help='Path to test feature file (tab‑separated)')
    parser.add_argument('--rules', type=str, required=True,
                        help='Comma‑separated rules, e.g. "nearest_dis>5,f1<2"')
    parser.add_argument('--logic', type=str, default='AND', choices=['AND','OR'],
                        help='How to combine multiple rules (default: AND)')
    parser.add_argument('--output_prefix', type=str, default='rule_based',
                        help='Prefix for output files')
    args = parser.parse_args()

    # Load data
    names, df_feat, y_true = load_test_file(args.test_file)
    print(f"Loaded {len(df_feat)} samples.")

    # Parse rules
    raw_rules = [r.strip() for r in args.rules.split(',') if r.strip()]
    parsed_rules = []
    for rule_str in raw_rules:
        # Split by operator signs
        import re
        m = re.match(r'(\w+)\s*(>=|<=|!=|==|>|<)\s*(.+)', rule_str)
        if not m:
            raise ValueError(f"Cannot parse rule: {rule_str}. Expected format like 'nearest_dis>5'")
        feat, op, val = m.groups()
        if feat not in FEATURE_COLS:
            raise ValueError(f"Unknown feature '{feat}'. Available: {list(FEATURE_COLS.keys())}")
        thr = float(val)
        parsed_rules.append((feat, op, thr))
    print(f"Rules: {parsed_rules}, logic = {args.logic}")

    # Predict
    pred_mask = apply_rules(df_feat, parsed_rules, args.logic)
    y_pred = pred_mask.astype(int)

    # Evaluate
    metrics = evaluate(y_true, y_pred)
    print("\n========== Rule‑based Results ==========")
    print(f"Logic: {args.logic}")
    for rule in parsed_rules:
        print(f"  {rule[0]} {rule[1]} {rule[2]}")
    print(f"\nTP = {metrics['TP']}, FP = {metrics['FP']}, FN = {metrics['FN']}, TN = {metrics['TN']}")
    print(f"Precision = {metrics['precision']:.4f}")
    print(f"Recall    = {metrics['recall']:.4f}")
    print(f"F1        = {metrics['f1']:.4f}")
    print(f"Accuracy  = {metrics['accuracy']:.4f}")

    # Save predictions
    res = pd.DataFrame({
        'name': names,
        'true_label': y_true,
        'pred_label': y_pred
    })
    out_csv = f"{args.output_prefix}_predictions.csv"
    res.to_csv(out_csv, index=False)
    print(f"\nPredictions saved to {out_csv}")

    # Save metrics
    pd.DataFrame([metrics]).to_csv(f"{args.output_prefix}_metrics.csv", index=False)
    print(f"Metrics saved to {args.output_prefix}_metrics.csv")

if __name__ == '__main__':
    main()
